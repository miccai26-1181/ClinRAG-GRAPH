from typing import Dict, List, Tuple, Optional

import math

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from config import ModelConfig, LossWeights
from grl import GRL
from encoders import ImageEncoder3D, TabularVarEncoder  # no text concepts
from gnn_layers import RelAttnRGCNLayer
from pooling import AttnPool, pool_graph


def _inv_sigmoid(p: float, eps: float = 1e-4) -> float:
    p = float(max(eps, min(1.0 - eps, p)))
    return math.log(p / (1.0 - p))


def _mix_detach(x: torch.Tensor, grad_mask: torch.Tensor) -> torch.Tensor:
    return x * grad_mask + x.detach() * (1.0 - grad_mask)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PCRCenterAgnosticGraphModel(nn.Module):
    def __init__(self, cfg: ModelConfig, loss_w: LossWeights):
        super().__init__()
        self.cfg = cfg
        self.loss_w = loss_w

        self.f_adv_img = nn.Sequential(nn.Linear(cfg.d, cfg.dz), nn.GELU(), nn.LayerNorm(cfg.dz))

        self.img_enc = ImageEncoder3D(
            d=cfg.d, in_channels=1, use_monai=getattr(cfg, "use_monai", False)
        )

        self.enc_age = TabularVarEncoder(d=cfg.d, kind="scalar")

        self.enc_ER = TabularVarEncoder(d=cfg.d, kind="onehot", num_classes=2)
        self.enc_PR = TabularVarEncoder(d=cfg.d, kind="onehot", num_classes=2)
        self.enc_HER2 = TabularVarEncoder(d=cfg.d, kind="onehot", num_classes=2)
        self.enc_Ki67 = TabularVarEncoder(d=cfg.d, kind="scalar")

        self.enc_subtype = nn.Embedding(4, cfg.d)
        self.rel2id = {
            # Strong prior (fixed 2.0)
            "strong_subtype_ER": 0,
            "strong_subtype_PR": 1,
            "strong_subtype_HER2": 2,
            "strong_Ki67_HER2": 3,

            # Soft prior (learnable, initialized per edge type)
            "soft_img_Ki67": 4,
            "soft_img_HER2": 5,
            "soft_img_subtype": 6,
            "soft_img_ER": 7,
            "soft_img_PR": 8,
            "soft_age_Ki67": 9,
            "soft_age_subtype": 10,

            # Learnable prior (learnable)
            "learnable_ER_PR": 11,
            "learnable_ER_HER2": 12,
            "learnable_PR_HER2": 13,
            "learnable_age_ER": 14,
            "learnable_age_PR": 15,
            "learnable_age_HER2": 16,
        }

        self.register_buffer("strong_prior_weight", torch.tensor(2.0, dtype=torch.float32), persistent=False)

        soft_init = {
            "soft_img_Ki67": 0.80,
            "soft_img_HER2": 0.8,
            "soft_img_subtype": 0.80,
            "soft_img_ER": 0.80,
            "soft_img_PR": 0.80,
            "soft_age_Ki67": 0.80,
            "soft_age_subtype": 0.80,
        }
        self.soft_prior_logit = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(_inv_sigmoid(v), dtype=torch.float32))
            for k, v in soft_init.items()
        })

        learnable_init = {
            "learnable_ER_PR": 0.2,
            "learnable_ER_HER2": 0.2,
            "learnable_PR_HER2": 0.2,
            "learnable_age_ER": 0.2,
            "learnable_age_PR": 0.2,
            "learnable_age_HER2": 0.2,
        }
        self.learnable_prior_logit = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(_inv_sigmoid(v), dtype=torch.float32))
            for k, v in learnable_init.items()
        })

        # node types
        self.TYPE_IMG = 0
        self.TYPE_TAB = 1
        # tab node keys
        self.tab_keys = ["age", "ER", "PR", "HER2", "Ki67", "subtype"]
        self.biomarker_keys = ["ER", "PR", "HER2", "Ki67"]

        # --- GNN stack ---
        num_rel = len(self.rel2id)
        num_types = 2
        self.gnn = nn.ModuleList([
            RelAttnRGCNLayer(
                d=cfg.d,
                num_relations=num_rel,
                num_node_types=num_types,
                basis=getattr(cfg, "basis", 0),
                dropout=getattr(cfg, "gnn_dropout", 0.1),
            )
            for _ in range(cfg.rgcn_layers)
        ])

        self.use_attn_pool = getattr(cfg, "use_attn_pool", True)
        self.attn_pool = AttnPool(cfg.d) if self.use_attn_pool else None

        self.f_ca = nn.Sequential(nn.Linear(cfg.d, cfg.dz), nn.GELU(), nn.LayerNorm(cfg.dz))

        self.pcr_head = MLP(cfg.dz, 1, hidden=256)
        self.adv_head = MLP(cfg.dz, cfg.num_centers, hidden=256)
        self.grl = GRL(lambd=cfg.grl_lambda)

        # ============================================================
        #                IMG DOMINANCE
        # ============================================================
        self.enable_img_dominance = bool(getattr(cfg, "enable_img_dominance", True))

        # --- Path 1: Feature dominance (img residual / gated update) ---
        self.img_residual_gating = bool(getattr(cfg, "img_residual_gating", True))
        mp_gate_init = float(getattr(cfg, "img_mp_gate_init", 0.30))
        self.img_mp_gate_logit = nn.Parameter(
            torch.full((len(self.gnn),), _inv_sigmoid(mp_gate_init), dtype=torch.float32)
        )
        self.img_ln = nn.LayerNorm(cfg.d)
        img_scale_init = float(getattr(cfg, "img_scale_init", 1.50))
        self.img_scale = nn.Parameter(torch.full((cfg.d,), img_scale_init, dtype=torch.float32))

        # --- Path 2: Readout dominance (img-only pCR branch + fusion) ---
        self.img_readout_dominance = bool(getattr(cfg, "img_readout_dominance", True))
        self.f_img = nn.Sequential(nn.Linear(cfg.d, cfg.dz), nn.GELU(), nn.LayerNorm(cfg.dz))
        self.pcr_head_img = MLP(cfg.dz, 1, hidden=256)

        w_img_init = float(getattr(cfg, "pcr_img_weight_init", 0.15))
        self.pcr_img_weight_logit = nn.Parameter(
            torch.tensor(_inv_sigmoid(w_img_init), dtype=torch.float32)
        )

        # --- Path 3: Optimization dominance (route gradients through img) ---
        self.pcr_detach_nonimg_readout = bool(getattr(cfg, "pcr_detach_nonimg_readout", False))
        self.pcr_other_grad_scale = float(getattr(cfg, "pcr_other_grad_scale", 1))
        self.adv_from_img = bool(getattr(cfg, "adv_from_img", True))

    @staticmethod
    def _add_bidir(
        edges: List[Tuple[int, int]],
        rels: List[int],
        masks: List[torch.Tensor],
        u: int,
        v: int,
        rid: int,
        mask_val: torch.Tensor,
    ):
        mv = mask_val if torch.is_tensor(mask_val) else torch.tensor(float(mask_val))
        edges.append((u, v)); rels.append(rid); masks.append(mv)
        edges.append((v, u)); rels.append(rid); masks.append(mv)

    def _encode_subtype(self, subtype_value: torch.Tensor) -> torch.Tensor:
        if subtype_value.dim() > 0 and subtype_value.numel() > 1:
            idx = subtype_value.argmax(dim=-1).long()
        else:
            idx = subtype_value.long()
        idx = idx.clamp(min=0, max=3)
        return self.enc_subtype(idx).view(-1)

    def _prior_strong(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.strong_prior_weight.to(device=device, dtype=dtype)

    def _prior_soft(self, rel_key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.sigmoid(self.soft_prior_logit[rel_key]).to(device=device, dtype=dtype)

    def _prior_learnable(self, rel_key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.sigmoid(self.learnable_prior_logit[rel_key]).to(device=device, dtype=dtype)

    def build_patient_graph(
        self,
        img_emb: torch.Tensor,
        tab_embs: Dict[str, torch.Tensor],
    ) -> Data:
        device = img_emb.device
        dtype = img_emb.dtype

        x_list = [img_emb]
        node_type = [self.TYPE_IMG]

        tab_idx: Dict[str, int] = {}
        for k in self.tab_keys:
            tab_idx[k] = len(x_list)
            x_list.append(tab_embs[k])
            node_type.append(self.TYPE_TAB)

        x = torch.stack(x_list, dim=0)
        node_type = torch.tensor(node_type, device=device, dtype=torch.long)

        edges: List[Tuple[int, int]] = []
        rels: List[int] = []
        masks: List[torch.Tensor] = []

        IMG = 0

        w_strong = self._prior_strong(device=device, dtype=dtype)
        self._add_bidir(edges, rels, masks, tab_idx["subtype"], tab_idx["ER"], self.rel2id["strong_subtype_ER"], w_strong)
        self._add_bidir(edges, rels, masks, tab_idx["subtype"], tab_idx["PR"], self.rel2id["strong_subtype_PR"], w_strong)
        self._add_bidir(edges, rels, masks, tab_idx["subtype"], tab_idx["HER2"], self.rel2id["strong_subtype_HER2"], w_strong)
        self._add_bidir(edges, rels, masks, tab_idx["Ki67"], tab_idx["HER2"], self.rel2id["strong_Ki67_HER2"], w_strong)
        
        
        self._add_bidir(edges, rels, masks, IMG, tab_idx["Ki67"], self.rel2id["soft_img_Ki67"],
                        self._prior_soft("soft_img_Ki67", device=device, dtype=dtype))
        self._add_bidir(edges, rels, masks, IMG, tab_idx["HER2"], self.rel2id["soft_img_HER2"],
                        self._prior_soft("soft_img_HER2", device=device, dtype=dtype))
        self._add_bidir(edges, rels, masks, IMG, tab_idx["subtype"], self.rel2id["soft_img_subtype"],
                        self._prior_soft("soft_img_subtype", device=device, dtype=dtype))
        self._add_bidir(edges, rels, masks, IMG, tab_idx["ER"], self.rel2id["soft_img_ER"],
                        self._prior_soft("soft_img_ER", device=device, dtype=dtype))
        self._add_bidir(edges, rels, masks, IMG, tab_idx["PR"], self.rel2id["soft_img_PR"],
                        self._prior_soft("soft_img_PR", device=device, dtype=dtype))

        self._add_bidir(edges, rels, masks, tab_idx["age"], tab_idx["Ki67"], self.rel2id["soft_age_Ki67"],
                        self._prior_soft("soft_age_Ki67", device=device, dtype=dtype))
        self._add_bidir(edges, rels, masks, tab_idx["age"], tab_idx["subtype"], self.rel2id["soft_age_subtype"],
                        self._prior_soft("soft_age_subtype", device=device, dtype=dtype))

        self._add_bidir(edges, rels, masks, tab_idx["ER"], tab_idx["PR"], self.rel2id["learnable_ER_PR"],
                        self._prior_learnable("learnable_ER_PR", device=device, dtype=dtype))
        self._add_bidir(edges, rels, masks, tab_idx["ER"], tab_idx["HER2"], self.rel2id["learnable_ER_HER2"],
                        self._prior_learnable("learnable_ER_HER2", device=device, dtype=dtype))
        self._add_bidir(edges, rels, masks, tab_idx["PR"], tab_idx["HER2"], self.rel2id["learnable_PR_HER2"],
                        self._prior_learnable("learnable_PR_HER2", device=device, dtype=dtype))

        self._add_bidir(edges, rels, masks, tab_idx["age"], tab_idx["ER"], self.rel2id["learnable_age_ER"],
                        self._prior_learnable("learnable_age_ER", device=device, dtype=dtype))
        self._add_bidir(edges, rels, masks, tab_idx["age"], tab_idx["PR"], self.rel2id["learnable_age_PR"],
                        self._prior_learnable("learnable_age_PR", device=device, dtype=dtype))
        self._add_bidir(edges, rels, masks, tab_idx["age"], tab_idx["HER2"], self.rel2id["learnable_age_HER2"],
                        self._prior_learnable("learnable_age_HER2", device=device, dtype=dtype))

        edge_index = torch.tensor(edges, device=device, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(rels, device=device, dtype=torch.long)
        edge_prior_mask = torch.stack(masks, dim=0).to(device=device, dtype=dtype)

        return Data(
            x=x,
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_prior_mask=edge_prior_mask,
        )


    def _pool_img_nodes(self, x: torch.Tensor, node_type: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int) -> torch.Tensor:
        img_mask = (node_type == self.TYPE_IMG)
        x_img = x[img_mask]
        b_img = batch_idx[img_mask].long()
        if x_img.numel() == 0:
            return x.new_zeros((num_graphs, x.size(-1)))

        g_img = x.new_zeros((num_graphs, x.size(-1)))
        g_img.index_add_(0, b_img, x_img)
        cnt = x.new_zeros((num_graphs,))
        ones = x.new_ones((b_img.numel(),))
        cnt.index_add_(0, b_img, ones)
        g_img = g_img / cnt.clamp_min(1.0).unsqueeze(-1)
        return g_img

    def forward_graph(self, batch: Batch) -> Dict[str, torch.Tensor]:
        x0 = batch.x
        node_type = batch.node_type
        edge_index = batch.edge_index
        edge_type = batch.edge_type
        edge_prior_mask = getattr(batch, "edge_prior_mask", None)

        x = x0
        attn_list = []
        for li, layer in enumerate(self.gnn):
            use_prior = (li == 0)
            x_prev = x
            x, alpha = layer(
                x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                node_type=node_type,
                edge_prior_mask=edge_prior_mask if (use_prior and edge_prior_mask is not None) else None,
            )
            
            if self.enable_img_dominance and self.img_residual_gating:
                img_mask = (node_type == self.TYPE_IMG)
                if img_mask.any():
                    mp_gate = torch.sigmoid(self.img_mp_gate_logit[li]).to(x.dtype).view(1, 1)
                    x_img = (1.0 - mp_gate) * x_prev[img_mask] + mp_gate * x[img_mask]
                    x_img = self.img_ln(x_img) * self.img_scale.view(1, -1)
                    x = x.clone()
                    x[img_mask] = x_img

            attn_list.append(alpha)

        num_graphs = int(batch.num_graphs)

        # img-only
        g_img = self._pool_img_nodes(x, node_type, batch.batch, num_graphs=num_graphs)
        z_img = self.f_img(g_img)
        y_logit_img = self.pcr_head_img(z_img).squeeze(-1)

        # pCR mixed
        if self.enable_img_dominance and self.pcr_detach_nonimg_readout:
            grad_mask = (node_type == self.TYPE_IMG).to(x.dtype).unsqueeze(-1)
            x_readout = _mix_detach(x, grad_mask)
        else:
            x_readout = x

        g_pcr = pool_graph(x_readout, batch.batch, self.use_attn_pool, self.attn_pool)
        z_ca = self.f_ca(g_pcr)
        y_logit_gnn = self.pcr_head(z_ca).squeeze(-1)

        if self.enable_img_dominance and self.img_readout_dominance:
            w_img = torch.sigmoid(self.pcr_img_weight_logit).to(y_logit_img.dtype)
            y_logit = (1.0 - w_img) * y_logit_gnn + w_img * y_logit_img
        else:
            w_img = y_logit_img.new_tensor(0.0)
            y_logit = y_logit_gnn

        return {
            "y_logit": y_logit,
            "y_logit_img": y_logit_img,
            "y_logit_gnn": y_logit_gnn,
            "attn_list": attn_list,
            "attn_list_full": attn_list,
            "edge_prior_mask": edge_prior_mask,
            "edge_dst": edge_index[1],
            "num_nodes": batch.num_nodes,
            "num_edges": batch.edge_index.size(1),
            "num_edges_pcr": batch.edge_index.size(1),
            "pcr_img_weight": w_img.detach(),
            "g_img": g_img.detach(),
            "g_pcr": g_pcr.detach(),
            "edge_index": edge_index,   # [2, E_batch]
            "edge_type": edge_type,     # [E_batch]
            "node_type": node_type,     # [N_batch]
            "batch_idx": batch.batch,   # [N_batch]
        }


    def forward(self, x_dce, tab, use_adv: bool = True) -> Dict[str, torch.Tensor]:
        device = x_dce.device
        B = x_dce.size(0)

        x_in = x_dce
        if x_in.dim() == 4:  # [B, D, H, W] -> [B, 1, D, H, W]
            x_in = x_in.unsqueeze(1)
        img_emb = self.img_enc(x_in)  # [B, d]

        tab_embs_batch: List[Dict[str, torch.Tensor]] = []
        for i in range(B):
            te = {
                "age": self.enc_age(tab["age"][i]).view(-1),
                "ER": self.enc_ER(tab["ER"][i]).view(-1),
                "PR": self.enc_PR(tab["PR"][i]).view(-1),
                "HER2": self.enc_HER2(tab["HER2"][i]).view(-1),
                "Ki67": self.enc_Ki67(tab["Ki67"][i]).view(-1),
                "subtype": self._encode_subtype(tab["subtype"][i]),
            }
            tab_embs_batch.append(te)

        graphs = []
        for i in range(B):
            graphs.append(self.build_patient_graph(
                img_emb=img_emb[i],
                tab_embs=tab_embs_batch[i],
            ))

        batch_graph = Batch.from_data_list(graphs).to(device)
        out = self.forward_graph(batch_graph)

        if use_adv:
            z_adv = self.f_adv_img(img_emb)        
            adv_logit = self.adv_head(self.grl(z_adv)) 
        else:
            adv_logit = out["y_logit"].new_zeros((B, self.cfg.num_centers))

        out["adv_logit"] = adv_logit
        return out