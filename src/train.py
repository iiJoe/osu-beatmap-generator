import numpy as np
from dataloader import AudioDataset
import torch, torch.nn as nn
from torch.utils.data import DataLoader

import constants
from models.fast_transformers import fast_transformer
from models.ast_model import ASTModel
from models.fine_tuned_model import FineTunedModel

def labels_from_csv(csv_file):
    data = np.genfromtxt(csv_file, delimiter=",", dtype=str, encoding='utf-8')
    file_paths = data[:, 0]
    labels = data[:, 1:].astype(float)

    return file_paths, labels

def train():
    file_paths, labels = labels_from_csv(constants.labels_file)
    dataset = AudioDataset(file_paths, labels)
    data_loader = DataLoader(dataset, batch_size=len(file_paths), shuffle=True)

    ast_mdl = ASTModel(label_dim=constants.label_dim, audioset_pretrain=True)
    fine_tuned_model = FineTunedModel(ast_mdl, fast_transformer)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(fine_tuned_model.parameters(), lr=0.0001)

    num_epochs = 10
    fine_tuned_model.train()
    for epoch in range(num_epochs):

        running_loss = 0.0
        for batch_spectrograms, batch_labels in data_loader:
            optimizer.zero_grad()
            outputs = fine_tuned_model(batch_spectrograms)
            loss = criterion(outputs, batch_labels.float())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}')

        torch.save(fine_tuned_model.state_dict(), "../pretrained_models/trained_model.pth")

train()
