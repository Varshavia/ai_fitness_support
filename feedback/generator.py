import numpy as np
import random
from feedback.messages import FEEDBACK_MESSAGES

def calculate_angle(x, y, z):
    ba = x - y
    bc = z - y
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def generate_feedback(keypoints, lang="en", movement="squat"):
    feedbacks = []
    rules_triggered = 0
    keypoints = keypoints.reshape(17, 2)

    # Tüm açılar default None
    knee_angle = None
    torso_angle = None
    body_angle = None
    elbow_angle = None

    if movement == "squat":
        shoulder = keypoints[5]
        hip = keypoints[11]
        knee = keypoints[13]
        ankle = keypoints[15]

        knee_angle = calculate_angle(hip, knee, ankle)
        torso_angle = calculate_angle(shoulder, hip, np.array([hip[0], hip[1]-100]))

        if knee_angle < 60:
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

    elif movement == "pushup":
        shoulder = keypoints[5]
        hip = keypoints[11]
        ankle = keypoints[15]
        elbow = keypoints[13]
        wrist = keypoints[15]

        body_angle = calculate_angle(shoulder, hip, ankle)
        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        if body_angle < 160:
            feedbacks.append(random.choice(FEEDBACK_MESSAGES[lang]["pushup_sag"]))
            rules_triggered += 1

        if elbow_angle > 130:
            feedbacks.append(random.choice(FEEDBACK_MESSAGES[lang]["pushup_shallow"]))
            rules_triggered += 1

    score = round(100 * (4 - rules_triggered) / 4, 2)

    return {
        "knee_angle": round(knee_angle, 2) if knee_angle is not None else None,
        "torso_angle": round(torso_angle, 2) if torso_angle is not None else None,
        "body_angle": round(body_angle, 2) if body_angle is not None else None,
        "elbow_angle": round(elbow_angle, 2) if elbow_angle is not None else None,
        "score": score,
        "feedbacks": feedbacks,
        "perfect": rules_triggered == 0,
        "summary": FEEDBACK_MESSAGES[lang]["perfect"] if rules_triggered == 0 else FEEDBACK_MESSAGES[lang]["title"],
        "score_message": FEEDBACK_MESSAGES[lang]["score"].format(score)
    }
