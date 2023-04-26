import RPi.GPIO as gpio
import time

def dimming(pinNum, duration=1, frequency=100):
  pwm = gpio.PWM(pinNum, frequency)

  try:
    pwm.start(0)

    for _ in range(int(duration * frequency)):
      for cycle in range(0, 101, 5):
        pwm.ChangeDutyCycle(cycle)
        time.sleep(1 / (frequency * 2))

      for cycle in range(100, -1, -5):
        pwm.ChangeDutyCycle(cycle)
        time.sleep(1 / (frequency * 2))
  finally:
    pwm.stop()

if __name__ == '__main__':
  gpio.setmode(gpio.BOARD)
  gpio.setup(13, gpio.OUT)

  dimming(13)

  gpio.cleanup()
