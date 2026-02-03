import os
import numpy as np
import faiss
import torch
from model.abse_model import ABSEModel
from embedding_extractor import extract_embedding

DATA_DIR = "keypoints"
MODEL_PATH = "model/abse_model.pth"
INDEX_PATH = "faiss_index.index"
LABELS_PATH = "faiss_labels.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ABSEModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

index = faiss.IndexFlatL2(128)  
labels = []

for label_name, label_value in [("correct", 1), ("incorrect", 0)]:
    folder = os.path.join(DATA_DIR, label_name)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            path = os.path.join(folder, file)
            embedding = extract_embedding(path, model, device)
            index.add(np.expand_dims(embedding, axis=0))  # (1, 128)
            labels.append(label_value)


faiss.write_index(index, INDEX_PATH)
np.save(LABELS_PATH, np.array(labels))
print("FAISS index and labels saved.")