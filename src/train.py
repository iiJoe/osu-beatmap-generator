import numpy as np
from dataloader import AudioDataset
import torch, torch.nn as nn
from torch.utils.data import DataLoader

import constants
from transformers import ASTModel
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

    ast_mdl = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=constants.label_dim)
    fine_tuned_model = FineTunedModel(ast_mdl)
    criterion = nn.BCEWithLogitsLoss() # Applies sigmoid internally
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

        torch.save(fine_tuned_model.state_dict(),     "ast/pretrained_models/trained_model.pth")

train()
