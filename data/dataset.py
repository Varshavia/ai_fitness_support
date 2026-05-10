import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

SEQ_LEN = 30
INPUT_DIM = 34

class KeypointDataset(Dataset):
    def __init__(self, keypoint_root, movements=None, seq_len=SEQ_LEN, input_dim=INPUT_DIM):
        self.data = []
        self.labels = []
        self.meta = []
        self.seq_len = seq_len
        self.input_dim = input_dim

        if movements is None:
            movements = [
                d for d in os.listdir(keypoint_root)
                if os.path.isdir(os.path.join(keypoint_root, d))
            ]

        for movement in movements:
            for label_dir, label_value in [("correct", 1), ("incorrect", 0)]:
                folder = os.path.join(keypoint_root, movement, label_dir)
                if not os.path.exists(folder):
                    continue
                for filepath in glob.glob(os.path.join(folder, "*.npy")):
                    keypoints = np.load(filepath)
                    if keypoints.shape[0] == 0:
                        continue

                    sequence = keypoints[:, :self.input_dim]
                    if len(sequence) >= self.seq_len:
                        sequence = sequence[:self.seq_len]
                    else:
                        pad = np.zeros((self.seq_len - len(sequence), self.input_dim))
                        sequence = np.vstack([sequence, pad])

                    self.data.append(torch.tensor(sequence, dtype=torch.float32))
                    self.labels.append(label_value)
                    self.meta.append({
                        "movement": movement,
                        "label": label_dir,
                        "filename": os.path.basename(filepath)
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Geriye donuk uyumluluk
SquatKeypointDataset = KeypointDataset
