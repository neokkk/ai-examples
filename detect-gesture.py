import cv2 as cv
import RPi.GPIO as gpio
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

pinRed = 37
pinGreen = 13
pinYellow = 36

first_fingers = [mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_TIP]
second_fingers = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP]
third_fingers = [mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
fourth_fingers = [mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_TIP]
last_fingers = [mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP]

def rock(hand_landmarks):
  return all(hand_landmarks.landmark[dip].y >= hand_landmarks.landmark[tip].y for [pip, dip, tip] in [second_fingers, third_fingers, fourth_fingers, last_fingers])

def scissors(hand_landmarks):
  return all(hand_landmarks.landmark[tip].y >= hand_landmarks.landmark[dip].y for [pip, dip, tip] in [first_fingers, second_fingers]) and all(hand_landmarks.landmark[pip].y >= hand_landmarks.landmark[tip].y for [pip, dip, tip] in [third_fingers, fourth_fingers, last_fingers])

def papers(hand_landmarks):
  return all(hand_landmarks.landmark[tip].y >= hand_landmarks.landmark[dip].y for [pip, dip, tip] in [first_fingers, second_fingers, third_fingers, fourth_fingers, last_fingers])

if __name__ == '__main__':
  hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

  cap = cv.VideoCapture(0, cv.CAP_V4L) 
  cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

  gpio.setmode(gpio.BOARD)
  gpio.setup(pinRed, gpio.OUT)
  gpio.setup(pinGreen, gpio.OUT)
  gpio.setup(pinYellow, gpio.OUT)

  try:
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret: break

      result = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
      if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if rock(hand_landmarks):
          print("바위!")
          gpio.output(pinRed, gpio.HIGH)
          gpio.output(pinGreen, gpio.LOW)
          gpio.output(pinYellow, gpio.LOW)

        if scissors(hand_landmarks):
          print("가위!")
          gpio.output(pinRed, gpio.LOW)
          gpio.output(pinGreen, gpio.HIGH)
          gpio.output(pinYellow, gpio.LOW)

        if papers(hand_landmarks):
          print("보!")
          gpio.output(pinRed, gpio.LOW)
          gpio.output(pinGreen, gpio.LOW)
          gpio.output(pinYellow, gpio.HIGH)

      cv.imshow('frame', frame)
      if cv.waitKey(5) & 0xFF == 27: break

  finally:
    cap.release()
    gpio.cleanup()
    cv.destroyAllWindows()