import numpy as np
import torch
from model.abse_model import ABSEModel


SEQ_LEN = 30
INPUT_DIM = 34

def extract_embedding(npy_path, model, device):
    keypoints = np.load(npy_path)
    
    if len(keypoints) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(keypoints), INPUT_DIM))
        keypoints = np.vstack([keypoints[:, :34], pad])
    else:
        keypoints = keypoints[:SEQ_LEN, :34]

    keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output, attn_weights = model(keypoints_tensor)
        lstm_out, _ = model.lstm(keypoints_tensor)
        attn_scores = torch.softmax(model.attn(lstm_out), dim=1)
        context = torch.sum(attn_scores * lstm_out, dim=1)  # [1, hidden_size*2]

    return context.squeeze(0).cpu().numpy()  # [128] shape


