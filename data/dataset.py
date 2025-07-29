import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SquatKeypointDataset(Dataset):
    def __init__(self, keypoint_root):
        self.data = []
        self.labels = []
        for label_dir in ['correct', 'incorrect']:
            folder = os.path.join(keypoint_root, label_dir)
            for file in os.listdir(folder):
                if file.endswith('.npy'):
                    keypoints = np.load(os.path.join(folder, file))
                    if keypoints.shape[0] == 0:
                        continue
                    keypoints = torch.tensor(keypoints, dtype=torch.float32)
                    label = 1 if label_dir == 'correct' else 0
                    self.data.append(keypoints)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
