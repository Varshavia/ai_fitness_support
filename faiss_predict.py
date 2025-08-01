import faiss
import numpy as np
import torch
from model.abse_model import ABSEModel
from embedding_extractor import extract_embedding

MODEL_PATH = "model/abse_model.pth"
INDEX_PATH = "faiss_index.index"
LABELS_PATH = "faiss_labels.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ABSEModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


index = faiss.read_index(INDEX_PATH)
labels = np.load(LABELS_PATH)

def predict_with_faiss(npy_path, k=3):
    embedding = extract_embedding(npy_path, model, device)
    distances, indices = index.search(np.expand_dims(embedding, axis=0), k)

    neighbor_labels = labels[indices[0]]
    predicted_label = int(np.round(np.mean(neighbor_labels)))  

    return {
        "prediction": "correct" if predicted_label == 1 else "incorrect",
        "neighbors": neighbor_labels.tolist(),
        "distances": distances[0].tolist()
    }
