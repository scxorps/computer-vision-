import math
import numpy as np
import cv2
import time
import os
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import mediapipe as mp

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set the environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

pTime = 0
cap = cv2.VideoCapture(0)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

vol4bar = 400
vol = 0
volper = 0
button_on = False
prev_hand_presence = False

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'
            
            if hand_label == 'Left':
                # Track distance between index and thumb fingers
                x1, y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])
                x2, y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                index_thumb_distance = math.hypot(x2 - x1, y2 - y1)
                
                if index_thumb_distance < 20:  # Adjust threshold as needed
                    button_on = True
                else:
                    button_on = False
                
                cv2.circle(frame, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 7, (255, 0, 255), cv2.FILLED)

            if hand_label == 'Right' and button_on:
                # Control volume with right hand
                x1, y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])
                x2, y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                length = math.hypot(x2 - x1, y2 - y1)
                
                vol = np.interp(length, [30, 200], [minVol, maxVol])
                vol4bar = np.interp(length, [30, 200], [400, 150])
                volper = np.interp(length, [30, 200], [0, 100])
                
                volume.SetMasterVolumeLevel(vol, None)
                
                cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if length < 30:
                    cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 15, (0, 255, 0), cv2.FILLED)
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 10), 3)
    cv2.rectangle(frame, (50, int(vol4bar)), (85, 400), (0, 255, 10), cv2.FILLED)
    cv2.putText(frame, f'{int(volper)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 0, 0), 3)
    
    if results.multi_hand_landmarks:
        prev_hand_presence = True
    else:
        if prev_hand_presence:
            button_on = False
        prev_hand_presence = False
    
    button_status = 'ON' if button_on else 'OFF'
    text_size = cv2.getTextSize(button_status, cv2.FONT_HERSHEY_COMPLEX, 1, 3)[0]
    text_x = frame.shape[1] - text_size[0] - 20
    text_y = frame.shape[0] // 2

    cv2.putText(frame, button_status, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0) if button_on else (0, 0, 255), 3)

    ctime = time.time()
    fps = 1 / (ctime - pTime)
    pTime = ctime
    cv2.putText(frame, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
