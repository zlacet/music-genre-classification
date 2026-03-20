import os
import librosa
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset
import kagglehub


TARGET_SR = 16000
MAX_LEN = 80000
N_MELS = 128


def get_transform():
    return T.MelSpectrogram(sample_rate=TARGET_SR, n_mels=N_MELS)


def load_gtzan_metadata():
    path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
    data_dir = os.path.join(path, "Data", "genres_original")

    classes = sorted(
        [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    )[:10]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    file_paths = []
    labels = []

    for cls in classes:
        cls_folder = os.path.join(data_dir, cls)
        for file in os.listdir(cls_folder):
            if file.lower().endswith((".wav", ".mp3")):
                file_paths.append(os.path.join(cls_folder, file))
                labels.append(class_to_idx[cls])

    clean_file_paths = []
    clean_labels = []

    for fp, lb in zip(file_paths, labels):
        try:
            y, _ = librosa.load(fp, sr=TARGET_SR, mono=True)
            if len(y) > 0:
                clean_file_paths.append(fp)
                clean_labels.append(lb)
        except Exception:
            pass

    return clean_file_paths, clean_labels, classes


class GTZANDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, target_sr=TARGET_SR, max_len=MAX_LEN):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.target_sr = target_sr
        self.max_len = max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            waveform, _ = librosa.load(file_path, sr=self.target_sr, mono=True)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.file_paths))

        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        elif waveform.shape[1] < self.max_len:
            waveform = F.pad(waveform, (0, self.max_len - waveform.shape[1]))

        if self.transform:
            waveform = self.transform(waveform)
            waveform = torch.log(waveform + 1e-6)
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)

        return waveform, label