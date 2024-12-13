import os
from dataloader import AudioDataset
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import constants
from utils import labels_from_csv, get_model

def eval():
    if not os.path.exists(constants.trained_model_path):
       raise FileNotFoundError(f"The trained model should be placed at {constants.trained_model_path}")
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    file_paths, labels = labels_from_csv(constants.test_labels_file)
    dataset = AudioDataset(file_paths, labels)
    batch_size = len(file_paths)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    test_loss = 0
    all_preds, all_labels = [], []

    # Evaluation
    with torch.no_grad():
        print("=== Evaluation Start ===")
        for data, labels in data_loader:
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            weights = torch.where(labels == 1, constants.label_weight, 1.0)
            print(f"weights: {weights}")
            criterion = nn.BCELoss(weight=weights)

            logits = model(data)
            loss = criterion(logits, labels.float())
            test_loss += loss.item()
            optimal_prediction = 0.5
            predictions = (logits > optimal_prediction).int()
            print(f"Size: {predictions.size()}")
            print(f"Sum of predictions: {torch.sum(predictions)}")
            print(f"Sum of labels: {torch.sum(labels)}")
            print(predictions)
            print(f"Logits: {logits}")
            print(f"Labels: {labels}")

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = [x for xs in all_preds for x in xs]
    all_labels = [x for xs in all_labels for x in xs]
    avg_loss = test_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print("=== Evaluation End ===")

eval()
