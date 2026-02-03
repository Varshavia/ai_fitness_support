import os
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

class KeypointExtractor:
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def extract_keypoints_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        keypoints_sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(frame_rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                frame_keypoints = []
                for lm in landmarks:
                    frame_keypoints.extend([lm.x, lm.y])
                keypoints_sequence.append(frame_keypoints)

        cap.release()
        return np.array(keypoints_sequence)

    def extract_keypoints_from_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(frame_rgb)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            frame_keypoints = []
            for lm in landmarks:
                frame_keypoints.extend([lm.x, lm.y])
            return np.array(frame_keypoints)
        return None
