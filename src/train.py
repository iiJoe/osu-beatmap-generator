from dataloader import AudioDataset
import torch, torch.nn as nn
from torch.utils.data import DataLoader

import constants
from utils import labels_from_csv, get_model

def train():
    model = get_model()
    file_paths, labels = labels_from_csv(constants.training_labels_file)
    dataset = AudioDataset(file_paths, labels)
    data_loader = DataLoader(dataset, batch_size=len(file_paths), shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print("CUDA is available")
        model.cuda()

    print("=== Training Start ===")
    model.train()
    for epoch in range(num_epochs):

        running_loss = 0.0
        for data, labels in data_loader:
            if is_cuda:
                data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}')

        torch.save(model.state_dict(), constants.trained_model_path)

    print("=== Training End ===")

train()
