from dataloader import AudioDataset
import torch, torch.nn as nn
from torch.utils.data import DataLoader

import constants
from utils import labels_from_csv, get_model

def train():
    file_paths, file_labels = labels_from_csv(constants.training_labels_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = constants.batch_size

    print("=== Training Start ===")

    print(f"Using weight of {constants.label_weight}")
    # Split data set to train batch by batch
    for i in range(0, len(file_paths), batch_size):
        print(f"Batch {i // batch_size + 1} Start")
        end_idx = min(i+batch_size, len(file_paths))
        print(f"Processing records from index {i} to {end_idx}")

        dataset = AudioDataset(file_paths[i:end_idx], file_labels[i:end_idx])
        data_loader = DataLoader(dataset, batch_size=end_idx - i, shuffle=True)

        num_epochs = 20
        model = get_model()
        model.to(device)
        model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            for data, labels in data_loader:
                data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                weights = torch.where(labels == 1, constants.label_weight, 1.0)
                criterion = nn.BCELoss(weight=weights)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels.float())

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}')

            torch.save(model.state_dict(), constants.trained_model_path)
        print(f"Batch {i // batch_size + 1} End")

    print("=== Training End ===")

train()
