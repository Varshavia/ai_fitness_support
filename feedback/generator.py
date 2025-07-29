import numpy as np
import random
from feedback.messages import FEEDBACK_MESSAGES

def calculate_angle(x, y, z):
    ba = x - y
    bc = z - y
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def generate_feedback(keypoints, lang="en"):
    feedbacks = []
    rules_triggered = 0

    shoulder = keypoints[2]
    hip = keypoints[8]
    knee = keypoints[9]
    ankle = keypoints[10]

    knee_angle = calculate_angle(hip, knee, ankle)
    torso_angle = calculate_angle(shoulder, hip, np.array([hip[0], hip[1]-100]))

    if knee_angle < 30:
        feedbacks.append(random.choice(FEEDBACK_MESSAGES[lang]["knee_too_much"]))
        rules_triggered += 1
    elif knee_angle > 130:
        feedbacks.append(random.choice(FEEDBACK_MESSAGES[lang]["knee_too_little"]))
        rules_triggered += 1

    if torso_angle < 60:
        feedbacks.append(random.choice(FEEDBACK_MESSAGES[lang]["torso_forward"]))
        rules_triggered += 1
    elif torso_angle > 100:
        feedbacks.append(random.choice(FEEDBACK_MESSAGES[lang]["torso_upright"]))
        rules_triggered += 1

    score = round(100 * (4 - rules_triggered) / 4, 2)

    return {
        "knee_angle": round(knee_angle, 2),
        "torso_angle": round(torso_angle, 2),
        "score": score,
        "feedbacks": feedbacks,
        "perfect": rules_triggered == 0,
        "summary": FEEDBACK_MESSAGES[lang]["perfect"] if rules_triggered == 0 else FEEDBACK_MESSAGES[lang]["title"],
        "score_message": FEEDBACK_MESSAGES[lang]["score"].format(score)
    }