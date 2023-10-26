import cv2 as cv
import sys
import RPi.GPIO as gpio

pinNum = int(sys.argv[1])

if not pinNum:
  exit(-1)

gpio.setmode(gpio.BOARD)
gpio.setup(pinNum, gpio.OUT)
videoCapture = cv.VideoCapture(0, cv.CAP_V4L)

def clean():
  videoCapture.release()
  gpio.cleanup()

try:
  while True:
    ret, frame = videoCapture.read()
    if not ret: break

    cv.imshow('frame', frame)
    key = cv.waitKey(10)

    # ESC 입력 시 종료
    if key == 27: break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # grayscale
    blur = cv.GaussianBlur(gray, (15, 15), 0)

    circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 2, 100, param1=100, param2=100, minRadius=35, maxRadius=500) # Hough Transform

    if circles is not None:
      gpio.output(pinNum, gpio.HIGH)
    else:
      gpio.output(pinNum, gpio.LOW)
except KeyboardInterrupt:
  clean

clean