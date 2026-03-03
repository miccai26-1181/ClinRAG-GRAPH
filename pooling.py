import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.utils import scatter

class AttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / (d ** 0.5))

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        score = (x * self.q).sum(dim=-1)
        w = pyg_softmax(score, batch)
        g = scatter(x * w.unsqueeze(-1), batch, dim=0, reduce="sum")
        return g

def pool_graph(x: torch.Tensor, batch: torch.Tensor, use_attn_pool: bool, attn_pool: AttnPool):
    if use_attn_pool:
        return attn_pool(x, batch)
    return global_mean_pool(x, batch)
