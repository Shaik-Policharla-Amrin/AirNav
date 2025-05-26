import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

pyautogui.FAILSAFE = False

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

# Webcam
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Timing control
snap_time = scroll_time = pause_time = volume_time = desktop_time = 0

# Cursor control
prev_x, prev_y = 0, 0
smoothening = 3
cursor_boost = 2

def fingers_status(hand_landmarks):
    fingers = []
    tips = [8, 12, 16, 20]
    dips = [6, 10, 14, 18]
    for tip, dip in zip(tips, dips):
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y else 0)
    return fingers

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Cursor control (index finger)
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)
            adj_x = np.interp(x, [0, w], [0, screen_width * cursor_boost])
            adj_y = np.interp(y, [0, h], [0, screen_height * cursor_boost])
            adj_x = np.clip(adj_x, 0, screen_width)
            adj_y = np.clip(adj_y, 0, screen_height)
            curr_x = prev_x + (adj_x - prev_x) / smoothening
            curr_y = prev_y + (adj_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            fingers = fingers_status(hand_landmarks)

            # Peace ✌️ = Screenshot
            if fingers == [1, 1, 0, 0] and time.time() - snap_time > 2:
                pyautogui.screenshot("screenshot.png")
                print("📸 Screenshot taken")
                snap_time = time.time()

            # Three fingers = Scroll
            if fingers[:3] == [1, 1, 1] and fingers[3] == 0 and time.time() - scroll_time > 1:
                pyautogui.scroll(800)
                print("🔼 Scroll up")
                scroll_time = time.time()

            # Fist = Pause/Play
            if fingers == [0, 0, 0, 0] and time.time() - pause_time > 2:
                pyautogui.press('space')
                print("⏯ Pause/Play")
                pause_time = time.time()

            # Thumbs up = Volume up
            if thumb_tip.y < wrist.y and time.time() - volume_time > 1:
                pyautogui.press('volumeup')
                print("🔊 Volume Up")
                volume_time = time.time()

            # Thumbs down = Volume down
            if thumb_tip.y > index_tip.y and time.time() - volume_time > 1:
                pyautogui.press('volumedown')
                print("🔉 Volume Down")
                volume_time = time.time()

            # Palm open = Show Desktop
            if fingers == [1, 1, 1, 1] and time.time() - desktop_time > 2:
                pyautogui.hotkey('win', 'd')
                print("🖥 Show Desktop")
                desktop_time = time.time()

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Mouse One-Handed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
