from dataloader import AudioDataset
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import constants
from utils import labels_from_csv, get_model

def eval():
    model = get_model()
    model.load_state_dict(torch.load(constants.trained_model_path))
    model.eval()

    file_paths, labels = labels_from_csv(constants.test_labels_file)
    dataset = AudioDataset(file_paths, labels)
    data_loader = DataLoader(dataset, batch_size=len(file_paths), shuffle=True)

    test_loss = 0
    all_preds, all_labels = [], []
    criterion = nn.BCELoss()

    # Evaluation
    with torch.no_grad():

        is_cuda = torch.cuda.is_available()
        if is_cuda:
            print("CUDA is available")
            model.cuda()

        print("=== Evaluation Begin ===")
        for data, labels in data_loader:
            if is_cuda:
                data, labels = data.cuda(), labels.cuda()

            logits = model(data)

            loss = criterion(logits, labels.float())
            test_loss += loss.item()

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).int()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


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
    print("=== Evaluation Begin ===")

eval()
