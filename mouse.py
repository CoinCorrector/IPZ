import pyautogui as pag
import time
import socket
import sys
import json

def run(addr, width, height):
    addr, port = addr.split(':')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((addr, int(port)))

    current_hand = None
    track_cursor = False

    while True:
        buf = sock.recv(1000)
        eom = buf.index(b'\x00')
        buf = buf[:eom]
        
        msg = json.loads(buf.decode())

        hand_id = msg['hand_id']

        if current_hand == None:
            current_hand = hand_id
            track_cursor = False # Assume hand is open when first tracking

        if 'state' in msg:
            if msg['state'] == 'PINCH':
                track_cursor = True
            else:
                track_cursor = False
        elif 'pos' in msg:
            if current_hand == hand_id and track_cursor:
                px, py, pz = msg['pos']
                px, py = int(px * width), int(py * height)
                pag.moveTo((px, py), duration=0.01)
                print('move', hand_id, '(', px, py, ')')

if __name__ == '__main__':
    w, h = pag.resolution()
    addr = sys.argv[-1] if len(sys.argv) > 1 else 'localhost:9111'

    run(addr, w, h)

