from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import numpy as np
import os
import uvicorn  # GEREKLİ: uvicorn modülünü içe aktar

from model.abse_model import ABSEModel
from predict.inference import predict_single_sample
from feedback.generator import generate_feedback

app = FastAPI()

# Model parametreleri
INPUT_SIZE = 34
HIDDEN_SIZE = 64
NUM_LAYERS = 1
MODEL_PATH = "C:/Workshop/Ar-Ge/ysf2/model/abse_model.pth"
SEQ_LEN = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ABSEModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully and is in evaluation mode.")

@app.post("/predict")
async def predict_squat(file: UploadFile = File(...), lang: str = "en"):

    contents = await file.read()
    keypoints = np.load(file.file)
    if keypoints.shape[0] < 1 or keypoints.shape[1] < 34:
        raise ValueError("Invalid keypoint shape")
    single_frame = keypoints[0][:34].reshape(17, 2)
    feedback = generate_feedback(single_frame, lang=lang)
    prediction = predict_single_sample(model, file.file.name, device)
    predicted_label = "correct" if prediction == 1 else "incorrect"
    return JSONResponse({
        "prediction": predicted_label,
        "feedback": feedback
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
