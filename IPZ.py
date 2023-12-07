import mediapipe as mp
import cv2
import math
import time  
from collections import deque



#Time for hand distance
last_update_time = time.time()
#Time for push button
last_update_time2 = time.time()
#Time fo hand state
last_update_time3 = time.time()


# Naming functions for drawing and for detecting hand
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# Circle to drag
circle_radius = 30
circle_position = (300, 300)
dragging = False


# Hand state
hand_state=False
prev_states = deque(maxlen=4)


prev_hand_size = 0


#Push button
push_button_state = False


# Function to calculate Euclidean distance
def Euclidean(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


# Video capturing
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        # value determining strictness of hand recognition (higher=more strict)
        min_detection_confidence=0.6,
        # value determining strictness of hand following (higher=more strict)
        min_tracking_confidence=0.2) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        #Offset for text diplay
        x_offset = 0


        # Draw circle
        cv2.circle(image, circle_position, circle_radius, (0, 0, 255), -1)

        # Draw landmarks on hand
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


                # Extracting landmarks
                landmarks = hand_landmarks.landmark


                # Get the landmarks for the fingertips and the base of the hand (wrist)
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
                index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                wrist = landmarks[mp_hands.HandLandmark.WRIST]
                middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_finger_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
                middle_finger_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                ring_finger_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_finger_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
                pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
                pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]


                # Calculate the Euclidean distance between the fingertips and the wrist
                index_to_wrist_length = Euclidean(index_finger_tip, wrist)
                index_mcp_to_wrist_length = Euclidean(index_finger_mcp, wrist)
                middle_to_wrist_length = Euclidean(middle_finger_tip, wrist)
                middle_mcp_to_wrist_length = Euclidean(middle_finger_mcp, wrist)
                ring_to_wrist_length = Euclidean(ring_finger_tip, wrist)
                ring_mcp_to_wrist_length = Euclidean(ring_finger_mcp, wrist)
                pinky_to_wrist_length = Euclidean(pinky_tip, wrist)
                pinky_mcp_to_wrist_length = Euclidean(pinky_mcp, wrist)
                thumb_to_wrist_length = Euclidean(thumb_tip, wrist)


                # distance - pinch
                index_to_thumb_length = Euclidean(index_finger_tip,thumb_tip)
                thumb_tip_to_ip_length = Euclidean(thumb_tip,thumb_ip)


                # Calculate the Euclidean distance between the hand and circle center
                hand_circle_distance = math.sqrt(
                    (circle_position[0] - middle_finger_dip.x * image.shape[1]) ** 2
                    + (circle_position[1] - middle_finger_dip.y * image.shape[0]) ** 2)


                # Determine if the hand is open or closed
                cur_hand_state = not (
                        index_mcp_to_wrist_length > index_to_wrist_length and
                        middle_mcp_to_wrist_length > middle_to_wrist_length and
                        ring_mcp_to_wrist_length > ring_to_wrist_length and
                        pinky_mcp_to_wrist_length > pinky_to_wrist_length
                )

                # Opened/closed - state in time
                if time.time() - last_update_time3 >= 0.005:
                    prev_states.append(cur_hand_state)
                    if all(prev_states):
                        hand_state=True
                    else:
                        hand_state=False
                        
                    last_update_time3 = time.time()
                

                # Hand sate
                if not hand_state:
                    # Checking if hand is at the circle
                    if hand_circle_distance <= 1.5*circle_radius:
                        dragging = True

                    hand_str = 'Closed'
                    color = (0, 0, 255)
                else:
                    dragging = False

                    hand_str = 'Open'
                    color = (0, 255, 0)

                if dragging:
                    circle_position = (
                        int(middle_finger_dip.x * image.shape[1]),
                        int(middle_finger_dip.y * image.shape[0]))


                # Hand state display
                cv2.putText(image, hand_str, (x_offset + 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                
                #Detecting pinch
                if (index_to_thumb_length < thumb_tip_to_ip_length) and hand_state:
                    pinch_str = 'Pinch'
                    color=(255, 0, 0)
                    cv2.putText(image, pinch_str, (x_offset+10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2,cv2.LINE_AA)


                hand_size = 10 * (Euclidean(middle_finger_mcp, wrist))

                
                # Estimating the distance to the camera BETA TODO
                distance_to_camera = round((hand_size) ** -2.1 * 0.6 + 0.4, 2)
                distance_str = f"{distance_to_camera:.2f} m"

                if time.time() - last_update_time >= 0.4:
                    prev_hand_size = hand_size
                    last_update_time = time.time()

                #cv2.putText(image,"A" +  str(prev_hand_size/10), (x_offset + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                #            cv2.LINE_AA)

                #cv2.putText(image, "B"  + str(hand_size/10), (x_offset + 10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                #            cv2.LINE_AA)


                #Push button
                if not dragging:
                    pushText = "Nie klikniety"
                    buttonColor = (0,255,0)

                    if hand_size-prev_hand_size > 0.2*hand_size:
                        pushText = "Przycisk wcisniety"
                        buttonColor = (0, 255, 0)
                        push_button_state = True
                        last_update_time2 = time.time()
                    else:
                        pushText = "Nie jest"
                        buttonColor = (0, 0, 255)

                    if push_button_state:
                        if time.time() - last_update_time2 <= 0.7:
                            pushText = "Przycisk wcisniety"
                            buttonColor = (0, 255, 0)
                        else:
                            push_button_state = False

                    cv2.putText(image, pushText, (x_offset + 10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                buttonColor, 2,
                                cv2.LINE_AA)


                # Display the distance of hand from the monitor
                cv2.putText(image, distance_str, (x_offset + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,cv2.LINE_AA)

                x_offset += 150

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
