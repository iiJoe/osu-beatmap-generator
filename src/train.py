from dataloader import AudioDataset
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from prodigyopt import Prodigy

import constants
from utils import get_model, collate_fn
from eval import validate


def evaluate(model, best_val_loss):
    avg_val_loss = validate(model)
    print(f"Validation Loss: {avg_val_loss}\n")
    if (avg_val_loss < best_val_loss):
        torch.save(model.state_dict(), constants.best_model_path)
        best_val_loss = avg_val_loss

    return best_val_loss


def train():
    batch_size = constants.batch_size
    best_val_loss = float("inf")
    num_epochs = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Training Start ===")
    print(f"Device used is : {device}")

    model = get_model().to(device)
    model.train()
    optimizer = Prodigy(model.parameters(), lr=1.,slice_p=1, weight_decay=0, d_coef=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    bce_criterion = nn.BCELoss(reduction="mean")
    l1_criterion = nn.L1Loss(reduction="mean")

    dataset = AudioDataset(constants.training_labels_file)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    # Split data set to train batch by batch
    for epoch in range(num_epochs):
        for i, (src, tgt) in enumerate(data_loader):
            print(f"Epoch {epoch + 1} - Batch {i + 1} Start")
            src = src.to(device)
            tgt = tgt.to(device)

            spec_pad_mask = (src.abs().sum(dim=1) == 0)
            hit_pad_mask = (tgt == 0).all(dim=2)
            hit_pad_mask[:, 0] = False

            optimizer.zero_grad()
            output = model(src, tgt, spec_pad_mask=spec_pad_mask, hit_pad_mask=hit_pad_mask)

            hit_pad_mask = (~hit_pad_mask).float().unsqueeze(-1)
            output = output * hit_pad_mask
            tgt = tgt * hit_pad_mask

            end_token_mask = torch.where(tgt[:, 1:, 0] == 0, constants.end_token_weight, 1.0).to(device)
            output[:, :-1, 0] *= end_token_mask
            tgt[:, 1:, 0] *= end_token_mask

            p_weights = torch.tensor(constants.predictions_weights, dtype=torch.float32).to(device)
            output[:, :-1, 1:] *= p_weights
            tgt[:, 1:, 1:] *= p_weights

            bce_loss = bce_criterion(output[:, :-1, 0], tgt[:, 1:, 0])
            l1_loss = l1_criterion(output[:, :-1, 1:], tgt[:, 1:, 1:])

            print(f"BCE Loss: {bce_loss}")
            print(f"L1 Loss: {l1_loss}")

            bce_loss.backward(retain_graph=True)
            l1_loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                best_val_loss = evaluate(model, best_val_loss)
                model.train()

            print(f"Epoch {epoch + 1} - Batch {i + 1} End")

        scheduler.step()

    best_val_loss = evaluate(model, best_val_loss)
    torch.save(model.state_dict(), constants.trained_model_path)
    print("=== Training End ===")


train()
