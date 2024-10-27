import torch, torch.nn as nn
import constants

class FineTunedModel(nn.Module):

    def __init__(self, model):
        super(FineTunedModel, self).__init__()
        self.model = model
        self.mlp = nn.Sequential(
            nn.Linear(constants.label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        model_output = self.model(x)
        mlp_output = self.mlp(model_output)
        return mlp_output.squeeze(-1)
