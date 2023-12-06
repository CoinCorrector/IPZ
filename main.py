#Importing libraries
import mediapipe as mp
import cv2
import math

#Naming functions for drawing and for detecting hand
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


#Circle to drag
circle_radius = 30
circle_position = (300, 300)
dragging = False


# Function to calculate Euclidean distance
def Euclidean(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


#Video capturing
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        #value determining strictness of hand recognition (higher=more strict)
        min_detection_confidence=0.6,
        #value determining strictness of hand following (higher=more strict)
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

        x_offset=0 

        # Draw circle
        cv2.circle(image, circle_position, circle_radius, (0, 0, 255), -1)


        #Draw landmarks on hand
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extracting landmarks
                landmarks = hand_landmarks.landmark

                # Get the landmarks for the fingertips and the base of the hand (wrist)
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
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


                # Calculate the Euclidean distance between the hand and circle center
                hand_circle_distance = math.sqrt(
                    (circle_position[0] - middle_finger_dip.x * image.shape[1]) ** 2
                    + (circle_position[1] - middle_finger_dip.y * image.shape[0]) ** 2)


                # Determine if the hand is open or closed
                hand_state = not (
                    index_mcp_to_wrist_length > index_to_wrist_length and
                    middle_mcp_to_wrist_length > middle_to_wrist_length and
                    ring_mcp_to_wrist_length > ring_to_wrist_length and
                    pinky_mcp_to_wrist_length > pinky_to_wrist_length
                )

                
                #Open/closed    
                if hand_state:
                    hand_str='Open'
                    color=(0, 255, 0)
                else:
                    hand_str='Closed'
                    color=(0, 0, 255)

                    

                #Hand state display
                cv2.putText(image, hand_str, (x_offset+10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2,cv2.LINE_AA)

                #Checking if hand is at the circle 
                if hand_state==False:
                    if hand_circle_distance <= circle_radius:
                        dragging = True
                else:
                    dragging = False

                if dragging: 
                    circle_position = (
                        int(middle_finger_dip.x * image.shape[1]),
                        int(middle_finger_dip.y * image.shape[0]))

               

                
                hand_size =10*( Euclidean(middle_finger_mcp, wrist))

                # Define a threshold for hand openness based on the hand size
                hand_open_threshold = hand_size * 0.75  # Adjust the multiplier to suit your needs

                # Determine if the hand is open or closed based on the distance
                # Estimating the distance to the camera BETA TODO
                distance_to_camera = round((hand_size)**-2.1 * 0.6 + 0.4, 2)
                distance_str=f"{distance_to_camera:.2f} m"
                # Display the distance of hand from the monitor
                if thumb_to_wrist_length > hand_open_threshold and index_to_wrist_length > hand_open_threshold:
                    cv2.putText(image, distance_str, (x_offset+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                else:
                    cv2.putText(image, distance_str, (x_offset+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                x_offset+=150

        cv2.imshow('Hand Tracking', image)



        if cv2.waitKey(5) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()
