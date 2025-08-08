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

    # Tüm hareketleri al (örneğin squat, pushup)
    movements = [d for d in glob.glob(f"{keypoint_dir}/*") if not d.endswith(".DS_Store")]

    for movement_path in movements:
        for label_type, label_value in [("correct", 1), ("incorrect", 0)]:
            files = glob.glob(f"{movement_path}/{label_type}/*.npy")
            for file in files:
                keypoints = np.load(file)
                if keypoints.shape[0] > 0:
                    sequence = keypoints[:, :INPUT_DIM]
                    if len(sequence) >= SEQ_LEN:
                        sequence = sequence[:SEQ_LEN]
                    else:
                        pad = np.zeros((SEQ_LEN - len(sequence), INPUT_DIM))
                        sequence = np.vstack([sequence, pad])
                    data.append(sequence)
                    labels.append(label_value)

    X = torch.tensor(np.array(data), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=4)

    return train_loader, test_loader
