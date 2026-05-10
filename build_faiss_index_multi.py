import os
import numpy as np
import faiss
import json
import torch

from model.abse_model import ABSEModel
from embedding_extractor import extract_embedding
from supabase_client import supabase


DATA_DIR = "keypoints"
INDEX_PATH = "faiss_index.index"
LABELS_PATH = "faiss_labels.npy"
META_PATH = "faiss_metadata.json"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ABSEModel()
model.load_state_dict(torch.load("model/abse_model.pth", map_location=device))
model.to(device)
model.eval()


index = faiss.IndexFlatL2(128)
labels = []
metadata = []


for movement in os.listdir(DATA_DIR):
    movement_path = os.path.join(DATA_DIR, movement)
    if not os.path.isdir(movement_path):
        continue

    for label_type in ['correct', 'incorrect']:
        label_path = os.path.join(movement_path, label_type)
        if not os.path.exists(label_path):
            continue

        label_value = 1 if label_type == "correct" else 0

        for file in os.listdir(label_path):
            if not file.endswith(".npy"):
                continue

            filepath = os.path.join(label_path, file)
            try:
                embedding = extract_embedding(filepath, model, device)
                index.add(np.expand_dims(embedding, axis=0))
                labels.append(label_value)

               
                entry = {
                    "filename": file,
                    "exercise_type": movement,
                    "prediction": label_type,
                    "score": None,  # skor yoksa boş geçilebilir
                    "feedback": {},  # boş feedback
                    "keypoints": None  # istersen: np.load(filepath).tolist()
                }
                metadata.append(entry)

                # 🔁 Supabase'e kaydet
                supabase.table("predictions").insert(entry).execute()
                print(f" {file} başarıyla eklendi ve Supabase'e yazıldı.")

            except Exception as e:
                print(f" {file} hata: {e}")


faiss.write_index(index, INDEX_PATH)
np.save(LABELS_PATH, np.array(labels))
with open(META_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n FAISS index, labels ve metadata kaydedildi.")


