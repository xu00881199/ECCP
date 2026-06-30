import torch
from torch import nn


class AnatomyEvidenceTokenizer(nn.Module):
    def __init__(self, num_anatomy_regions=7, hidden_dim=256, num_heads=8):
        super().__init__()
        self.anatomy_queries = nn.Parameter(torch.randn(num_anatomy_regions, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, visual_tokens):
        batch_size = visual_tokens.size(0)
        queries = self.anatomy_queries.unsqueeze(0).expand(batch_size, -1, -1)
        anatomy_tokens, attention_maps = self.cross_attn(
            query=queries,
            key=visual_tokens,
            value=visual_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        anatomy_tokens = self.norm(anatomy_tokens + queries)
        return anatomy_tokens, attention_maps
