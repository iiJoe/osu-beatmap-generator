from dataloader import AudioDataset
import torch, torch.nn as nn
from torch.utils.data import DataLoader

import constants
from utils import get_model, causal_mask, labels_from_json
from eval import validate


def evaluate(model, best_val_loss):
    avg_val_loss = validate(model)
    print(f"Validation Loss: {avg_val_loss}\n")
    if (avg_val_loss < best_val_loss):
        torch.save(model.state_dict(), constants.best_model_path)
        best_val_loss = avg_val_loss
        model.train()
    return best_val_loss

def train():
    file_paths, file_labels = labels_from_json(constants.training_labels_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = constants.batch_size

    print("=== Training Start ===")

    model = get_model()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    mse_criterion = nn.MSELoss(reduction="none")
    num_epochs = 5
    best_val_loss = float("inf")

    # Split data set to train batch by batch
    for epoch in range(num_epochs):
        for i in range(0, len(file_paths), batch_size):
            print(f"Epoch {epoch + 1} - Batch {i // batch_size + 1} Start")
            end_idx = min(i+batch_size, len(file_paths))
            print(f"Processing records from index {i} to {end_idx}")

            dataset = AudioDataset(file_paths[i:end_idx], file_labels[i:end_idx])
            data_loader = DataLoader(dataset, batch_size=end_idx - i, shuffle=True)

            running_loss = 0.0
            for src, tgt in data_loader:
                src, tgt = src.to(device, dtype=torch.float32, non_blocking=True), tgt.to(device, dtype=torch.float32, non_blocking=True)
                bce_weights = torch.where(tgt[:, :, 0] != 2, 1, 0).to(device)  # Check for those labelled as 2, set weights to 0
                mse_weights = bce_weights.unsqueeze(-1).expand_as(tgt[:, :, 1:])
                mse_weights[:, :, 2] = mse_weights[:, :, 2] * constants.time_weight
                bce_weights[tgt[:, :, 0] == 0] = constants.end_token_weight
                tgt[:, :, 0][tgt[:, :, 0] == 2] = 0  # Change those with exists values of 2 and change to 0 for bce
                normalize_bounds = torch.tensor(constants.predictions_normalize, dtype=torch.float32)

                _, target_len, _ = tgt.shape
                tgt_mask = causal_mask(target_len)
                output = model(src, tgt, tgt_mask=tgt_mask)
                tgt = tgt.clone()
                tgt[:, :, 1:] = tgt[:, :, 1:] / normalize_bounds

                bce_criterion = nn.BCELoss(weight=bce_weights, reduction="mean")
                bce_loss = bce_criterion(output[:, :, 0], tgt[:, :, 0])
                mse_loss = (mse_criterion(output[:, :, 1:], tgt[:, :, 1:]) * mse_weights).sum() / mse_weights.sum()
                total_loss = bce_loss + mse_loss
                print(f"BCE Loss: {bce_loss}")
                print(f"MSE Loss: {mse_loss}")
                print(f"Total Loss: {total_loss}")

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()

            if (i // batch_size + 1) % 20 == 0:
                best_val_loss = evaluate(model, best_val_loss)

            print(f"Batch {i // batch_size + 1} End")

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}")

    best_val_loss = evaluate(model, best_val_loss)
    torch.save(model.state_dict(), constants.trained_model_path)
    print("=== Training End ===")


train()
