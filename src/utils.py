import numpy as np

from models.osu_model import OsuModel
import os
import torch
import json
import constants

# Deprecated, used for old model
def labels_from_csv(csv_file):
    data = np.genfromtxt(csv_file, delimiter=",", dtype=str, encoding='utf-8')
    file_paths = data[:, 0]
    labels = data[:, 1:].astype(float)

    return file_paths, labels

def labels_from_json(json_file):
    with open(json_file) as json_data:
        max_len = 0
        file_paths = []
        attributes_list = []
        data = json.load(json_data)
        for splice in data:
            attributes = splice[constants.json_attributes_key]
            if not attributes:
                continue

            file_paths.append(splice[constants.json_file_path_key])
            for i in range(len(attributes)):
                attributes[i][constants.exists_key] = 1 if i < len(attributes) - 1 else 0
            max_len = max(max_len, len(attributes))
            attributes_list.append(attributes)

        padded_attributes_list = []
        for attributes in attributes_list:
            padded_attributes = attributes + [{constants.exists_key: 0, **{k: 0 for k in attributes[0] if k != constants.exists_key}}] * (max_len - len(attributes))
            padded_attributes_list.append(padded_attributes)

        final_attributes_list = np.array([
            [[
                attr[constants.exists_key],
                attr["x"],
                attr["y"],
                attr["time"],
                attr["type"],
                attr["hitSound"],
            ]
             for attr in attrs]
            for attrs in padded_attributes_list])
        return file_paths, final_attributes_list


def get_model():
    print(f"Retrieving model...")
    model = OsuModel(
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )
    if os.path.exists(constants.trained_model_path):
        print(f"Trained model exists. Loading saved state")
        checkpoint = torch.load(constants.trained_model_path)
        model.load_state_dict(checkpoint)

    return model

def causal_mask(batch_size, seq_len):
    mask = torch.triu(torch.ones(batch_size, seq_len, dtype=torch.bool), diagonal=1)
    return mask
