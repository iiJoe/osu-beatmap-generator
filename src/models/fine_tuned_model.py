import torch, torch.nn as nn
import constants

class FineTunedModel(nn.Module):

    def __init__(self, model):
        super(FineTunedModel, self).__init__()
        self.model = model
        self.mlp = nn.Linear(constants.label_dim, 1)

    def forward(self, x):
        model_output = self.model(x).last_hidden_state
        mlp_output = self.mlp(model_output)
        return mlp_output.squeeze(-1)
