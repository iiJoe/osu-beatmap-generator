import torch.nn as nn
import constants

class FineTunedModel(nn.Module):

    def __init__(self, pos_embed, transformer):
        super(FineTunedModel, self).__init__()
        self.pos_embed = pos_embed
        self.transformer = transformer
        self.mlp = nn.Sequential(
            nn.Linear(constants.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        pos_embed_output = self.pos_embed(x)
        transformer_output = self.transformer(pos_embed_output)
        mlp_output = self.mlp(transformer_output)
        output = mlp_output.squeeze(-1)
        return output
