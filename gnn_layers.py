from typing import Optional, Tuple
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax


class RelAttnRGCNLayer(MessagePassing):
    def __init__(
        self,
        d: int,
        num_relations: int,
        num_node_types: int,
        basis: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.d = d
        self.R = num_relations
        self.T = num_node_types
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        self.type_emb = nn.Embedding(num_node_types, d)
        self.W0 = nn.Linear(d, d, bias=False)

        self.use_basis = basis > 0
        if self.use_basis:
            self.B = basis
            self.basis = nn.Parameter(torch.empty(self.B, d, d))
            nn.init.xavier_uniform_(self.basis)
            self.coeff = nn.Parameter(torch.empty(num_relations, self.B))
            nn.init.xavier_uniform_(self.coeff)
        else:
            self.Wr = nn.Parameter(torch.empty(num_relations, d, d))
            nn.init.xavier_uniform_(self.Wr)

        self.attn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4 * d, d),
                nn.GELU(),
                nn.Linear(d, 1),
            )
            for _ in range(num_relations)
        ])

        self.ln = nn.LayerNorm(d)
        self._cached = {}

    def _rel_weight(self, r: torch.Tensor) -> torch.Tensor:
        if not self.use_basis:
            return self.Wr[r]
        c = self.coeff[r]  
        return torch.einsum("eb,bdh->edh", c, self.basis) 
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_type: torch.Tensor,
        edge_prior_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self._cached = {
            "edge_type": edge_type,
            "node_type": node_type,
            "edge_prior_mask": edge_prior_mask,
            "edge_index": edge_index,
            "tau_prior": getattr(self, "tau_prior", 0.2),
        }

        out = self.propagate(edge_index=edge_index, x=x)
        out = self.W0(x) + out
        out = self.act(self.ln(out))

        self._compute_attention_stats(x)

        return out, self._cached["alpha"]

    def message(self, x_j, x_i, index):
        edge_type = self._cached["edge_type"]
        node_type = self._cached["node_type"]
        edge_prior_mask = self._cached.get("edge_prior_mask", None)

        src = self._cached["edge_index"][0]
        dst = self._cached["edge_index"][1]

        t_j = self.type_emb(node_type[src])
        t_i = self.type_emb(node_type[dst])

        E = x_j.size(0)
        e = None

        for r in range(self.R):
            m = (edge_type == r)
            if m.any():
                inp = torch.cat([x_j[m], x_i[m], t_j[m], t_i[m]], dim=-1)
                out = self.attn[r](inp)
                if e is None:
                    e = out.new_zeros((E, 1))
                e[m] = out

        if e is None:
            e = x_j.new_zeros((E, 1))

        e = e.squeeze(-1)

        if edge_prior_mask is not None:
            eps = 1e-12
            prior = edge_prior_mask.view(-1).clamp(min=eps).to(e.dtype)
            tau_prior = self._cached.get("tau_prior", 0.2)
            e = e + tau_prior * prior.log()

        alpha = pyg_softmax(e, index)
        alpha = self.dropout(alpha)

        Wr = self._rel_weight(edge_type)
        msg = torch.bmm(Wr, x_j.unsqueeze(-1)).squeeze(-1)

        self._cached["alpha"] = alpha
        return msg * alpha.unsqueeze(-1)


    def _compute_attention_stats(self, x: torch.Tensor):
        """
        Compute (no grad):
          - mean attention entropy over nodes
          - hard-edge attention ratio
        """
        if "alpha" not in self._cached:
            return

        with torch.no_grad():
            alpha = self._cached["alpha"]               
            edge_index = self._cached["edge_index"]
            dst = edge_index[1]                        

            eps = 1e-12
            num_nodes = x.size(0)

            ent_edge = -alpha * torch.log(alpha.clamp(min=eps))
            ent_per_node = torch.zeros(
                num_nodes, device=alpha.device, dtype=alpha.dtype
            )
            ent_per_node.scatter_add_(0, dst, ent_edge)
            self._cached["attn_entropy"] = ent_per_node.mean().item()

            edge_prior_mask = self._cached.get("edge_prior_mask", None)
            if edge_prior_mask is not None:
                hard_mask = edge_prior_mask.view(-1) > 1.5  # hard=2.0
                hard_alpha = alpha[hard_mask].sum()
                total_alpha = alpha.sum()
                self._cached["hard_alpha_ratio"] = (
                    hard_alpha / (total_alpha + eps)
                ).item()
            else:
                self._cached["hard_alpha_ratio"] = None
