from model.abse_model import ABSEModel
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from predict.inference import predict_single_sample
from feedback.generator import generate_feedback
import numpy as np



# Model parametreleri
INPUT_SIZE = 34
HIDDEN_SIZE = 64
NUM_LAYERS = 1
MODEL_PATH = "C:/Workshop/Ar-Ge/ysf2/model/abse_model.pth"
SEQ_LEN = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ABSEModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)

print("Loading model from:", MODEL_PATH)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print("Model loaded successfully.")
model.to(device)
model.eval()
print("Model is in evaluation mode.")
print("Device:", device)
contents = file_path = "C:\Workshop\Ar-Ge\ysf2\keypoints\incorrect\squat_incorrect_1.npy"  # Example file path
print("Reading file contents from:", file_path)
contents = np.load(file_path)
print("File contents loaded from:", contents)

keypoints = np.load(file_path)
print("Keypoints shape:", keypoints.shape)

if keypoints.shape[0] < 1 or keypoints.shape[1] < 34:
    raise ValueError("Invalid keypoint shape")
single_frame = keypoints[0][:34].reshape(17, 2)
feedback = generate_feedback(single_frame, lang="tr")
prediction = predict_single_sample(model, file_path, device)
predicted_label = "correct" if prediction == 1 else "incorrect"
print("Prediction:", predicted_label)
print("Feedback:", feedback)