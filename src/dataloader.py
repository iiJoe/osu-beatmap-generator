import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import librosa
import numpy as np

def audio_to_spectrogram_tensor(audio_file):
  y, sr = librosa.load(audio_file, sr=None, duration=10.24)
  n_fft = 256
  hop_length = 441
  S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
  S_db = librosa.amplitude_to_db(np.abs(S))

  spectrogram = S_db[:128, :1024].T

  spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)
  padded_tensor = pad_spectrogram_tensor(spectrogram_tensor)
  normalized_tensor = normalize_tensor(padded_tensor)

  return normalized_tensor

def pad_spectrogram_tensor(tensor, target_shape=(1024, 128)):
    current_shape = tensor.shape

    if current_shape == target_shape:
        return tensor

    padding_height = target_shape[0] - current_shape[0]

    if padding_height > 0:
        tensor = F.pad(tensor, (0, 0, 0, padding_height), mode='constant', value=0)

    return tensor

# Z-score Normalization
def normalize_tensor(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std * 2)


class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        spectrogram = audio_to_spectrogram_tensor(self.file_paths[idx])
        normalized_spectrogram = normalize_tensor(spectrogram)

        label = self.labels[idx]
        return normalized_spectrogram, label
