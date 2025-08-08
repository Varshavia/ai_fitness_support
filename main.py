from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import torch
import io

from model.abse_model import ABSEModel
from feedback.generator import generate_feedback
from log_to_supabase import log_to_supabase
from ollama_client import generate_llm_feedback


app = FastAPI(title="Fitness Feedback API")

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
async def predict(file: UploadFile = File(...), lang: str = "en", exercise: str = "squat"):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files are supported")

    contents = await file.read()
    try:
        keypoints = np.load(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid .npy file: {str(e)}")

    if keypoints.shape[0] < 1 or keypoints.shape[1] < 34:
        raise HTTPException(status_code=400, detail="Invalid keypoint shape")

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

    # 3) kural tabanlı açılar (ilk frame)
    first_frame = keypoints[0][:34].reshape(17, 2)
    rule_based = generate_feedback(first_frame, lang=lang, movement=exercise)

    angles = {
        "knee_angle": rule_based.get("knee_angle"),
        "torso_angle": rule_based.get("torso_angle"),
        "body_angle": rule_based.get("body_angle"),
        "elbow_angle": rule_based.get("elbow_angle"),
    }

    # 4) LLM feedback
    llm = generate_llm_feedback(
        movement=exercise,
        lang=lang,
        angles=angles,
        score=rule_based.get("score", 0.0),
        predicted_label=predicted_label
    )

    # 5) Supabase log
    payload_feedback = {"rule_based": rule_based, "llm": llm}
    log_to_supabase(file.filename, predicted_label, payload_feedback, exercise)

    # 6) Response
    return JSONResponse(content={
        "prediction": predicted_label,
        "angles": angles,
        "score": rule_based.get("score"),
        "feedback": {
            "rule_based": rule_based.get("feedbacks", []),
            "llm": llm
        }
    })  

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
