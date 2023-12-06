import mediapipe as mp
import numpy as np
import cv2
import math

# Model params
MAX_NUM_HANDS = 2
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
# Render params
DRAW_OVERLAY_PALM = True
DRAW_OVERLAY_FINGERS = False
DRAW_LANDMARKS = False

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Circle to drag
circle_radius = 30
circle_position = (300, 300)
dragging = False

vec = lambda x: np.array([x.x, x.y, x.z])

TIPS = [
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
]

MCPS = [
    mp_hands.HandLandmark.INDEX_FINGER_MCP,
    mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
    mp_hands.HandLandmark.PINKY_MCP,
    mp_hands.HandLandmark.RING_FINGER_MCP,
]

def draw_circle(img, v, r, color):
    h, w, _ = image.shape
    x, y = v[0] * w, v[1] * h
    return cv2.circle(image, (int(x), int(y)), r, color, 2)

# Video capturing
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    ) as hands:

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        hand_closed = dict()

        # Draw circle
        cv2.circle(image, circle_position, circle_radius, (0, 0, 255), -1)

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if DRAW_LANDMARKS:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extracting landmarks
                landmarks = hand_landmarks.landmark

                # Get the landmarks for the fingertips and the base of the hand (wrist)
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                middle_finger_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]

                # Wrist landmark
                wrist = vec(landmarks[mp_hands.HandLandmark.WRIST])

                # Middle of the palm from vector average
                palm_mid = np.copy(wrist)

                # Finger distances
                d_tips = list()
                d_mcps = list()

                for mcp_i, tip_i in zip(MCPS, TIPS):
                    mcp = vec(landmarks[mcp_i])
                    tip = vec(landmarks[tip_i])
                    d_tips.append(np.linalg.norm(tip - wrist))
                    d_mcps.append(np.linalg.norm(mcp - wrist))

                    palm_mid += mcp

                    if DRAW_OVERLAY_FINGERS:
                        draw_circle(image, mcp, 12, (0, 0, 200))
                        draw_circle(image, tip, 12, (0, 200, 0))
                        draw_circle(image, wrist, 12, (200, 0, 0))

                palm_mid /= (len(MCPS) + 1)
                d_tips = np.array(d_tips)
                d_mcps = np.array(d_mcps)

                if DRAW_OVERLAY_PALM:
                    draw_circle(image, palm_mid, 20, (200, 200, 200))

                # Determine if the hand is open or closed based on the distance
                hand_closed[i] = (d_tips.sum() < d_mcps.sum())

                # Calculate the Euclidean distance between the hand and circle center
                hand_circle_distance = math.sqrt(
                    (circle_position[0] - palm_mid[0] * image.shape[1]) ** 2
                    + (circle_position[1] - palm_mid[1] * image.shape[0]) ** 2)

                # Checking if hand is at the circle 
                if hand_closed[i]:
                    if hand_circle_distance <= circle_radius:
                        dragging = True
                else:
                    dragging = False

                if dragging: 
                    circle_position = (
                        int(middle_finger_dip.x * image.shape[1]),
                        int(middle_finger_dip.y * image.shape[0]))
                
                #hand_size = 10 * Euclidean(middle_finger_mcp, wrist)

                # Define a threshold for hand openness based on the hand size
                #hand_open_threshold = hand_size * 0.75  # Adjust the multiplier to suit your needs

                # Determine if the hand is open or closed based on the distance
                # Estimating the distance to the camera BETA TODO
                #distance_to_camera = round((hand_size)**-2.1 * 0.6 + 0.4, 2)
                #distance_str=f"{distance_to_camera:.2f} m"
                # Display the distance of hand from the monitor

        for i, is_closed in hand_closed.items():
            text = f'{i} Closed' if is_closed else f'{i} Open'
            color = (0, 0, 255)
            cv2.putText(image, text, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
