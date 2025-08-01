# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import torch
import io

from model.abse_model import ABSEModel
from feedback.generator import generate_feedback

app = FastAPI(title="Squat Feedback API")

# Model config
MODEL_PATH = "model/abse_model.pth"
INPUT_SIZE = 34
HIDDEN_SIZE = 64
NUM_LAYERS = 1
SEQ_LEN = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ABSEModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = "en"):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files are supported")

    contents = await file.read()

    try:
        keypoints = np.load(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid .npy file: {str(e)}")

    if keypoints.shape[0] < 1 or keypoints.shape[1] < 34:
        raise HTTPException(status_code=400, detail="Invalid keypoint shape")

    # Model input prepare
    from predict.inference import SEQ_LEN, INPUT_DIM

    if len(keypoints) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(keypoints), INPUT_DIM))
        keypoints = np.vstack([keypoints[:, :INPUT_DIM], pad])
    else:
        keypoints = keypoints[:SEQ_LEN, :INPUT_DIM]

    input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        prediction = torch.argmax(output, dim=1).item()

    predicted_label = "correct" if prediction == 1 else "incorrect"

    first_frame = keypoints[0][:34].reshape(17, 2)
    feedback = generate_feedback(first_frame, lang)

    return JSONResponse(content={
        "prediction": predicted_label,
        "feedback": feedback
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
