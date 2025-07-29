import glob
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

SEQ_LEN = 30
INPUT_DIM = 34

def load_and_preprocess_data(keypoint_dir="keypoints"):
    data = []
    labels = []

    correct_files = glob.glob(f"{keypoint_dir}/correct/*.npy")
    incorrect_files = glob.glob(f"{keypoint_dir}/incorrect/*.npy")

    for file in correct_files:
        keypoints = np.load(file)
        if keypoints.shape[0] > 0:
            data.append(keypoints)
            labels.append(1)

    for file in incorrect_files:
        keypoints = np.load(file)
        if keypoints.shape[0] > 0:
            data.append(keypoints)
            labels.append(0)

    processed_data = []
    for keypoint_seq in data:
        flattened = keypoint_seq[:, :34]
        if len(flattened) >= SEQ_LEN:
            flattened = flattened[:SEQ_LEN]
        else:
            pad = np.zeros((SEQ_LEN - len(flattened), INPUT_DIM))
            flattened = np.vstack([flattened, pad])
        processed_data.append(flattened)

    X = torch.tensor(processed_data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=4)

    return train_loader, test_loader