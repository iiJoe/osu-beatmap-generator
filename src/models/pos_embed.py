import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, model_dim):
        super().__init__()
        self.position_embedding = nn.Embedding(seq_length, model_dim)
        self.seq_length = seq_length

    def forward(self, x):
        positions = torch.arange(0, self.seq_length, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pos_embeds = self.position_embedding(positions)
        return x + pos_embeds
