from dataloader import AudioDataset
import torch, torch.nn as nn
from torch.utils.data import DataLoader

import constants
from utils import get_model, causal_mask, labels_from_json

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
    mse_scale = 1e5
    num_epochs = 10

    # Split data set to train batch by batch
    for i in range(0, len(file_paths), batch_size):
        print(f"Batch {i // batch_size + 1} Start")
        end_idx = min(i+batch_size, len(file_paths))
        print(f"Processing records from index {i} to {end_idx}")

        dataset = AudioDataset(file_paths[i:end_idx], file_labels[i:end_idx])
        data_loader = DataLoader(dataset, batch_size=end_idx - i, shuffle=True)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for src, tgt in data_loader:
                src, tgt = src.to(device, dtype=torch.float32, non_blocking=True), tgt.to(device, dtype=torch.float32, non_blocking=True)
                bce_weights = torch.where(tgt[:, :, 0] == 1, 1.0, 0).to(device)
                mse_weights = bce_weights.unsqueeze(-1).expand_as(tgt[:, :, 1:])

                _, target_len, _ = tgt.shape
                tgt_mask = causal_mask(target_len, target_len)
                output = model(src, tgt, tgt_mask=tgt_mask)

                bce_criterion = nn.BCELoss(weight=bce_weights, reduction="mean")
                bce_loss = bce_criterion(output[:, :, 0], tgt[:, :, 0])
                mse_loss = (mse_criterion(output[:, :, 1:], tgt[:, :, 1:]) * mse_weights).sum() / (mse_weights.sum() * mse_scale + 1e-8)
                total_loss = bce_loss + mse_loss
                print(f"BCE Loss: {bce_loss}")
                print(f"MSE Loss: {mse_loss}")
                print(f"Total Loss: {total_loss}")

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}")

        print(f"Batch {i // batch_size + 1} End")

    torch.save(model.state_dict(), constants.trained_model_path)
    print("=== Training End ===")

train()
