import torch
import constants
from torch.utils.data import Dataset
from utils import audio_to_spectrogram_tensor, labels_from_json


class AudioDataset(Dataset):
    def __init__(self, labels_file):
        file_paths, file_labels = labels_from_json(labels_file)
        self.file_paths = file_paths
        self.labels = file_labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        spectrogram = audio_to_spectrogram_tensor(self.file_paths[idx])
        start_token = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.float32)
        normalized_factor = [1] + constants.predictions_normalize
        end_token = torch.tensor([normalized_factor], dtype=torch.float32)

        label = torch.cat((start_token,
                          torch.tensor(self.labels[idx], dtype=torch.float32),
                          end_token)) if len(self.labels[idx]) > 0 else torch.empty(0, 6)
        label = label / torch.tensor(normalized_factor, dtype=torch.float32)

        return spectrogram, label
