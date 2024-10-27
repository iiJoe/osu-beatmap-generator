import torch, torch.nn as nn
import constants
import fast_transformers

class FineTunedModel(nn.Module):

    def __init__(self, model, transformer):
        super(FineTunedModel, self).__init__()
        self.model = model
        self.transformer = transformer
        self.mlp = nn.Sequential(
            nn.Linear(constants.label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        model_output = self.model(x)
        transformer_output = self.transformer(model_output)
        mlp_output = self.mlp(transformer_output)
        return mlp_output
