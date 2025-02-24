import os
from dataloader import AudioDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import constants
from utils import labels_from_json, get_model, causal_mask


def validate(model=None):
    file_paths, file_labels = labels_from_json(constants.test_labels_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = constants.batch_size

    if model is None:
        if not os.path.exists(constants.trained_model_path):
            raise FileNotFoundError(f"The trained model should be placed at {constants.trained_model_path}")

            model = get_model()

    model.to(device)
    model.eval()

    all_preds, all_labels, all_reg_preds, all_reg_tgt = [], [], [], []

    mse_criterion = nn.MSELoss(reduction="none")
    total_loss = 0.0

    # Evaluation
    with torch.no_grad():
        print("=== Evaluation Start ===")
        for i in range(0, len(file_paths), batch_size):
            print(f"Batch {i // batch_size + 1} Start")
            end_idx = min(i+batch_size, len(file_paths))
            print(f"Processing records from index {i} to {end_idx}")

            dataset = AudioDataset(file_paths[i:end_idx], file_labels[i:end_idx])
            data_loader = DataLoader(dataset, batch_size=end_idx - i, shuffle=False)

            for src, tgt in data_loader:
                src, tgt = src.to(device, dtype=torch.float32, non_blocking=True), tgt.to(device, dtype=torch.float32, non_blocking=True)
                bce_weights = torch.where(tgt[:, :, 0] != 2, 1, 0).to(device)  # Check for those labelled as 2, set weights to 0
                mse_weights = bce_weights.unsqueeze(-1).expand_as(tgt[:, :, 1:])
                tgt[:, :, 0][tgt[:, :, 0] == 2] = 0  # Change those with exists values of 2 and change to 0 for bce

                _, target_len, _ = tgt.shape
                padding_mask = (tgt[:, :, 0] != 2)
                tgt_mask = causal_mask(target_len)
                predictions = model(src, tgt, tgt_mask=tgt_mask)

                optimal_prediction = 0.5
                exists_pred = (predictions[:, :, 0][padding_mask] > optimal_prediction).int()
                exists_tgt = tgt[:, :, 0][padding_mask]
                all_preds.append(exists_pred.cpu())
                all_labels.append(exists_tgt.cpu())

                regression_mask = padding_mask.unsqueeze(-1).expand_as(tgt[:, :, 1:])
                reg_preds = predictions[:, :, 1:][regression_mask]
                reg_tgt = tgt[:, :, 1:][regression_mask]
                all_reg_preds.append(reg_preds.cpu())
                all_reg_tgt.append(reg_tgt.cpu())

                bce_criterion = nn.BCELoss(weight=bce_weights, reduction="mean")
                bce_loss = bce_criterion(predictions[:, :, 0], tgt[:, :, 0])
                mse_loss = (mse_criterion(predictions[:, :, 1:], tgt[:, :, 1:]) * mse_weights).sum() / mse_weights.sum()
                total_loss += (bce_loss + mse_loss).item()

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_reg_preds = torch.cat(all_reg_preds)
    all_reg_tgt = torch.cat(all_reg_tgt)

    print("=== Evaluation End ===")

    return total_loss / len(data_loader)
