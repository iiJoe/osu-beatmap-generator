import os
from dataloader import AudioDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import constants
from utils import get_model, collate_fn


def validate(model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = constants.batch_size

    if model is None:
        if not os.path.exists(constants.trained_model_path):
            raise FileNotFoundError(f"The trained model should be placed at {constants.trained_model_path}")

            model = get_model()

    model = model.to(device)
    model.eval()

    dataset = AudioDataset(constants.test_labels_file)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    bce_criterion = nn.BCELoss(reduction="mean")
    l1_criterion = nn.L1Loss(reduction="mean")
    total_loss = 0.0

    # Evaluation
    with torch.no_grad():
        print("=== Evaluation Start ===")
        for i, (src, tgt) in enumerate(data_loader):
            src = src.to(device)
            tgt = tgt.to(device)

            spec_pad_mask = (src.abs().sum(dim=1) == 0)
            hit_pad_mask = (tgt == 0).all(dim=2)
            hit_pad_mask[:, 0] = False

            output = model(src, tgt, spec_pad_mask=spec_pad_mask, hit_pad_mask=hit_pad_mask)

            hit_pad_mask = (~hit_pad_mask).float().unsqueeze(-1)
            output = output * hit_pad_mask
            tgt = tgt * hit_pad_mask

            end_token_mask = torch.where(tgt[:, 1:, 0] == 0, constants.end_token_weight, 1.0).to(device)
            output[:, :-1, 0] *= end_token_mask
            tgt[:, 1:, 0] *= end_token_mask

            p_weights = torch.tensor(constants.predictions_weights, dtype=torch.float32).to(device)
            output[:, :-1, 1:] *= p_weights
            tgt[:, 1:, 1:] *= p_weights

            bce_loss = bce_criterion(output[:, :-1, 0], tgt[:, 1:, 0])
            l1_loss = l1_criterion(output[:, :-1, 1:], tgt[:, 1:, 1:])

            total_loss += l1_loss.item()

    print("=== Evaluation End ===")

    return total_loss / len(data_loader)
