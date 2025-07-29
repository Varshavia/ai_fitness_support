import numpy as np
import torch

SEQ_LEN = 30
INPUT_DIM = 34

def predict_single_sample(model, npy_path, device):
    model.eval()
    keypoints = np.load(npy_path)

    if len(keypoints) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(keypoints), INPUT_DIM))
        keypoints = np.vstack([keypoints[:, :34], pad])
    else:
        keypoints = keypoints[:SEQ_LEN, :34]

    keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(keypoints_tensor)
        if isinstance(output, tuple):
            output = output[0]

        prediction = torch.argmax(output, dim=1).item()

    return prediction