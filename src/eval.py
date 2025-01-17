import os
from dataloader import AudioDataset
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error

import constants
from utils import labels_from_json, get_model, causal_mask

def eval():

    if not os.path.exists(constants.trained_model_path):
        raise FileNotFoundError(f"The trained model should be placed at {constants.trained_model_path}")

    file_paths, file_labels = labels_from_json(constants.test_labels_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = constants.batch_size

    model = get_model()
    model.to(device)
    model.eval()

    all_preds, all_labels, all_reg_preds, all_reg_tgt = [], [], [], []

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

                _, target_len, _ = tgt.shape
                padding_mask = (tgt[:, :, 0] != 2)
                tgt_mask = causal_mask(target_len, target_len)
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


    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_reg_preds = torch.cat(all_reg_preds)
    all_reg_tgt = torch.cat(all_reg_tgt)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)

    mae = mean_absolute_error(all_reg_tgt, all_reg_preds)
    mse = mean_squared_error(all_reg_tgt, all_reg_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print("=== Evaluation End ===")

eval()
