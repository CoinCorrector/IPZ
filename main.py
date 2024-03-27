import mediapipe as mp
import numpy as np
from numpy import linalg
import cv2
import math
from collections import deque
import json
import socket
from threading import Thread
from queue import Queue
from select import select
from pathlib import Path
from os import getenv

# Model params
MAX_NUM_HANDS = 4
DETECTION_CONFIDENCE = 0.4
TRACKING_CONFIDENCE = 0.5
# Render params
DRAW_OVERLAY_PALM = True
DRAW_OVERLAY_FINGERS = False
DRAW_LANDMARKS = False
# Event triggering params
PINCH_REQUIRE_LAST = 4

LAST_DETECTION = None

class GestureProcessor:
    class CaptureState:
        def __init__(self, label, hand_id):
            self.label = label
            self.hand_id = hand_id
            self.points = list()

    def __init__(self):
        self.capture_dir = Path('capture')
        self.captures = dict()
        self.current_capture = None
        self.load_captures()

    def load_captures(self):
        self.capture_dir.mkdir(exist_ok=True)
        for f in self.capture_dir.glob('*.npy'):
            self.captures[f.stem] = np.load(str(f))

    def begin_capture(self, label, hand_id):
        self.current_capture = GestureProcessor.CaptureState(label, hand_id)
        print('=' * 80)
        print(f'CAPTURING {label} ON HAND {hand_id}')
        print('=' * 80)

    def end_capture(self, commit_d=0.01):
        cap = self.current_capture
        points = np.array(cap.points)
        out_points = np.zeros((0, 2))

        # Space points equally
        p = np.diff(points, axis=0)
        d = np.hypot(*p.T)
        total_d = 0

        for i, p in enumerate(points[:-1]):
            if total_d >= commit_d:
                out_points = np.r_[out_points, [p]]
                total_d = 0
            else:
                total_d += d[i]

        cap_path = self.capture_dir / f'{cap.label}.npy'
        print('=' * 80)
        print(f'SAVING {cap.label} TO {str(cap_path)}')
        np.save(str(cap_path), out_points)
        print('=' * 80)

        self.current_capture = None

    def get_capturing_hand_id(self):
        if cap := self.current_capture:
            return cap.hand_id

    def is_capturing(self):
        if cap := self.current_capture:
            return cap.label is not None

    def begin_detection(self, hand_id):
        self.current_capture = GestureProcessor.CaptureState(None, hand_id)
        print('=' * 80)
        print(f'DETECTING ON HAND {hand_id}')
        print('=' * 80)
    
    def end_detection(self):
        p = np.array(self.current_capture.points)
        z = (p.T[0] + 1j * p.T[1])
        w = np.fft.fft(z)

        d_best = 1e9
        label_best = None

        for label, target in self.captures.items():
            z2_ = target.T[0] + 1j * target.T[1]
            z2 = np.interp(np.linspace(0, 1, len(z)), np.linspace(0, 1, len(z2_)), z2_)
            w2 = np.fft.fft(z2)

            d = np.abs(w - w2).sum()
            if d < d_best:
                d_best = d
                label_best = label

        print('=' * 80)
        print(f'DETECTION RESULT {label_best}')
        print('=' * 80)

        global LAST_DETECTION
        LAST_DETECTION = label_best

    def update(self, hand):
        if cap := self.current_capture:
            if cap.hand_id == hand.idx:
                cap.points.append(hand.pos()[:2])

gesture_processor = GestureProcessor()

# Server sends two kinds of messages
# - Hand state changed ('hand_id': int, 'state': str)
# - Hand position update ('hand_id': int, 'pos': [float x 3])
class EventServer:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('0.0.0.0', port))
        self.sock.listen(1)

        self.event_queue = Queue()
        self.clients = list()
        self.accept_thread = Thread(target=self._accept_loop)
        self.send_thread = Thread(target=self._send_loop)

    def _accept_loop(self):
        while self.running:
            readfd, _, _ = select([self.sock], [], [], 0.5)
            if self.sock in readfd:
                client_sock, _ = self.sock.accept()
                self.clients.append(client_sock)

    def _send_loop(self):
        while self.running:
            if msg := self.event_queue.get():
                bad_clients = []
                for sock in self.clients:
                    try:
                        sock.send(msg + b'\x00')
                    except Exception:
                        bad_clients.append(sock)
                for sock in bad_clients:
                    self.clients.remove(sock)

    def start(self):
        self.running = True
        self.accept_thread.start()
        self.send_thread.start()

    def stop(self):
        self.running = False
        self.event_queue.put(None)

    def send(self, event):
        msg = json.dumps(event).encode()
        self.event_queue.put(msg)

# Create server for pushing events to client
server = EventServer(9111)
server.start()

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

def iter_mp_hands(multi_hand_landmarks):
    for hand_landmarks in multi_hand_landmarks[:MAX_NUM_HANDS]:
        # Extracting landmarks
        landmarks = hand_landmarks.landmark

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
            d_tips.append(linalg.norm(tip - wrist))
            d_mcps.append(linalg.norm(mcp - wrist))

            palm_mid += mcp

            if DRAW_OVERLAY_FINGERS:
                draw_circle(image, mcp, 12, (0, 0, 200))
                draw_circle(image, tip, 12, (0, 200, 0))
                draw_circle(image, wrist, 12, (200, 0, 0))

        palm_mid /= (len(MCPS) + 1)
        d_tips = np.array(d_tips)
        d_mcps = np.array(d_mcps)

        yield (landmarks, palm_mid, (d_tips, d_mcps))

class HandState:
    OPEN = 0
    CLOSED = 1
    PINCH = 2

    def name(state):
        return ["Open", "Closed", "Pinch"][state]

class History:
    def __init__(self, histlen):
        self.hist = deque(maxlen=histlen)

    def push(self, value):
        self.hist.append(value)

    def last_n(self, n):
        return list(self.hist)[-n:]

    def __getitem__(self, i):
        return self.hist[i]

class Hand:
    def __init__(self, idx):
        self.idx = idx
        self.visible = False
        self.state = HandState.OPEN
        self.pos_hist = History(120)
        self.pinch_hist = History(30)
        self.close_hist = History(30)
        self.state_changed_cbs = list()
        # Default position
        self.pos_hist.push(np.array([-1,-1,0]))

    def pos(self):
        return self.pos_hist[-1]

    def median_pos(self, n):
        pos = np.array(self.pos_hist.last_n(n))
        return np.median(pos, axis=0)

    def mean_velocity(self, n):
        pos = np.array(self.pos_hist.last_n(n))
        vel = (pos[1:] - pos[:-1]) * 30
        return vel.sum(axis=0) / n

    def register_state_callback(self, cb):
        self.state_changed_cbs.append(cb)

    def update_state(self):
        notify_changed = lambda pstate, state: [cb(pstate, state) for cb in self.state_changed_cbs]

        if self.state == HandState.OPEN:
            if sum(self.pinch_hist.last_n(5)) >= 3:
                self.state = HandState.PINCH
                notify_changed('OPEN', 'PINCH')
            elif all(self.close_hist.last_n(3)):
                self.state = HandState.CLOSED
                notify_changed('OPEN', 'CLOSED')
        elif self.state == HandState.CLOSED:
            if not any(self.close_hist.last_n(5)):
                self.state = HandState.OPEN
                notify_changed('CLOSED', 'OPEN')
        elif self.state == HandState.PINCH:
            if sum(self.close_hist.last_n(5)) >= 4:
                self.state = HandState.CLOSED
                notify_changed('PINCH', 'CLOSED')
            elif not any(self.pinch_hist.last_n(4)):
                self.state = HandState.OPEN
                notify_changed('PINCH', 'OPEN')

def draw_text(img, text, v, color):
    h, w, _ = img.shape
    x, y = v[0] * w, v[1] * h
    margin = 10
    return cv2.putText(img, text, (int(x)-margin, int(y)+margin), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

def draw_line(img, p0, p1, color, thickness=2):
    h, w, _ = img.shape
    x0, y0 = p0[0] * w, p0[1] * h
    x1, y1 = p1[0] * w, p1[1] * h
    return cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)

def draw_circle(img, v, r, color):
    h, w, _ = img.shape
    x, y = v[0] * w, v[1] * h
    return cv2.circle(img, (int(x), int(y)), r, color, 2)

# Video capturing
cap = cv2.VideoCapture(0)

# List of all hands to keep track of
hands = [Hand(i) for i in range(MAX_NUM_HANDS)]
# Table mapping model index to hand index, None means hands_present < MAX_NUM_HANDS
hand_mapping = [None for i in range(MAX_NUM_HANDS)]
# State for mapping renewal, happens only when number of hands reported by model changes
renew_hand_mapping = False
prev_hands_present = 0

gestures_to_capture = deque([x for x in getenv('CAPTURE_LIST', '').split(':') if x])

# Register hand callbacks
for hand in hands:
    class Handlers:
        def __init__(self, idx):
            self.idx = idx

        def send_message(self, pstate, state):
            server.send({'hand_id': self.idx, 'pstate': pstate, 'state': state})

        def end_capture(self, pstate, state):
            if pstate == 'PINCH' and gesture_processor.get_capturing_hand_id() == self.idx:
                if gesture_processor.is_capturing():
                    gesture_processor.end_capture()
                else:
                    gesture_processor.end_detection()

        def begin_capture(self,pstate, state):
            if state == 'PINCH':
                if len(gestures_to_capture) > 0:
                    label = gestures_to_capture.popleft()
                    gesture_processor.begin_capture(label, self.idx)
                else:
                    gesture_processor.begin_detection(self.idx)

    handlers = Handlers(hand.idx)
    hand.register_state_callback(handlers.send_message)
    hand.register_state_callback(handlers.end_capture)
    hand.register_state_callback(handlers.begin_capture)

with mp_hands.Hands(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    ) as hand_detector:

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hand_detector.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        hands_present = min(len(results.multi_hand_landmarks if results.multi_hand_landmarks else []), MAX_NUM_HANDS)

        # Number of hands changed, renew mapping
        if prev_hands_present != hands_present:
            #print(f'Change in num of hands: {prev_hands_present} -> {hands_present}')
            prev_hands_present = hands_present
            renew_hand_mapping = True

        # Hide visibility of all hands, if hand is present it will be set to True
        for i in range(MAX_NUM_HANDS):
            hands[i].visible = False
            # Clear hand mapping if regenerating
            if renew_hand_mapping:
                hand_mapping[i] = None

        if hands_present > 0:
            # (landmarks, pos, (d_tips, d_mcps))
            model_hands = list(iter_mp_hands(results.multi_hand_landmarks))

            if renew_hand_mapping:
                HAND_POS_CMP_MEDIAN = 8
                # Distance cross matrix between model reported hand and internal state hand
                model_to_marker_dist = np.zeros((hands_present, MAX_NUM_HANDS))

                # Populate distances between model hand and all MAX_NUM_HANDS hands
                for i in range(hands_present):
                    _, pos, _ = model_hands[i]
                    dist_to_model = lambda x: linalg.norm(pos - x.median_pos(HAND_POS_CMP_MEDIAN))
                    model_to_marker_dist[i, :] = list(map(dist_to_model, hands))

                # Finds the model hand closest to some internal hand and assigns the correct mapping,
                # currently assigned hand distances get masked out
                MASKED_DISTANCE = 1e9
                for i in range(hands_present):
                    m2m = model_to_marker_dist[:, :]
                    #print(f'm2m[{i}]', m2m)

                    d_min = m2m.argmin(axis=1)
                    dist_map = m2m[range(hands_present), d_min]
                    closest_model = dist_map.argmin()

                    hand_idx = d_min[closest_model]
                    hand_mapping[closest_model] = hand_idx
                    #print(f'Map model[{closest_model}] -> hand[{hand_idx}]')

                    # Mask out the hand which was just mapped
                    model_to_marker_dist[:, hand_idx] = MASKED_DISTANCE
                    model_to_marker_dist[closest_model, :] = MASKED_DISTANCE

            # Old logic implementing per hand behavior
            for i in range(hands_present):
                current_hand = hands[hand_mapping[i]]
                current_hand.visible = True
                landmarks, model_pos, (d_tips, d_mcps) = model_hands[i]

                # Update hand position
                current_hand.pos_hist.push(model_pos)

                index_finger_tip = vec(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP])
                thumb_tip = vec(landmarks[mp_hands.HandLandmark.THUMB_TIP])
                thumb_ip = vec(landmarks[mp_hands.HandLandmark.THUMB_IP])

                finger_tip_thumb_tip = linalg.norm(index_finger_tip - thumb_tip)
                thumb_ip_thumb_tip = linalg.norm(thumb_tip - thumb_ip)

                # Update hand pinch and close
                current_hand.pinch_hist.push(finger_tip_thumb_tip < thumb_ip_thumb_tip)
                current_hand.close_hist.push(d_tips.sum() < d_mcps.sum())
                current_hand.update_state()

                # Calculate the Euclidean distance between the hand and circle center
                hand_circle_distance = math.sqrt(
                    (circle_position[0] - model_pos[0] * image.shape[1]) ** 2
                    + (circle_position[1] - model_pos[1] * image.shape[0]) ** 2)

                # Checking if hand is at the circle 
                if current_hand.state == HandState.PINCH:
                    gesture_processor.update(current_hand)

                    if hand_circle_distance <= circle_radius:
                        dragging = True
                else:
                    dragging = False

                # FIXME: Currently broken when >1 hand is present, dragging should keep track of dragging hand
                # FIXME: Change circle_position to use normalized coordinates?
                if dragging: 
                    circle_position = (
                        int(model_pos[0] * image.shape[1]),
                        int(model_pos[1] * image.shape[0]))

        # Draw circle
        #cv2.circle(image, circle_position, circle_radius, (0, 0, 255), -1)

        for i, hand in enumerate(hands):
            if hand.visible:
                pos = hand.pos()

                # Send out updated position for i'th hand
                server.send({'hand_id': i, 'pos': pos.tolist()})

                if DRAW_OVERLAY_PALM:
                    draw_circle(image, pos, 20, (200, 200, 200))

                    v = hand.mean_velocity(5)
                    draw_line(image, pos, pos + v/30, (200, 0, 0))
                    draw_text(image, f'{i}', pos, (200, 200, 200))

                text = f'{i} ({pos[0]:.2f}, {pos[1]:.2f}) {HandState.name(hand.state)}'
                color = (0, 0, 255)
                cv2.putText(image, text, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        if LAST_DETECTION:
            cv2.putText(image, LAST_DETECTION, (10, 30 * 5), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)

        if renew_hand_mapping:
            #print('Hand mapping table [mp] -> [hand_idx]')
            for mp_i, i in enumerate(hand_mapping):
                #print(f'[{mp_i}] -> [{i}]')
                pass

        renew_hand_mapping = False

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

# Cleanup
server.stop()
cap.release()
cv2.destroyAllWindows()
