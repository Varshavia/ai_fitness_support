import os
import cv2
import numpy as np
import mediapipe as mp

# Giriş ve çıkış klasörleri
VIDEO_ROOT = "pushup_videos"
SAVE_ROOT = "keypoints/pushup"

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Her video için ana işlem
def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Görüntüyü RGB'ye çevir
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            frame_keypoints = []
            for lm in landmarks:
                frame_keypoints.extend([lm.x, lm.y])  # (x, y)
            keypoints_sequence.append(frame_keypoints)

    cap.release()
    return np.array(keypoints_sequence)

# Ana döngü
for label in ["correct", "incorrect"]:
    input_folder = os.path.join(VIDEO_ROOT, label)
    output_folder = os.path.join(SAVE_ROOT, label)
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".mp4"):
            video_path = os.path.join(input_folder, file)
            print(f"🎥 İşleniyor: {file}")
            keypoints = extract_keypoints_from_video(video_path)

            if keypoints.size == 0:
                print(f"⚠️ Keypoint çıkarılamadı: {file}")
                continue

            output_filename = os.path.splitext(file)[0] + ".npy"
            output_path = os.path.join(output_folder, output_filename)
            np.save(output_path, keypoints)
            print(f"Registered: {output_filename}")
           
