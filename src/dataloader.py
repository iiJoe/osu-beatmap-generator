from torch.utils.data import Dataset
from utils import audio_to_spectrogram_tensor

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        spectrogram = audio_to_spectrogram_tensor(self.file_paths[idx])

        label = self.labels[idx]
        return spectrogram, label
