import argparse
import cv2 as cv
import time

classFile = "coco.names"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

classNames = []

with open(classFile, "rt") as f:
  classNames = f.read().rstrip("\n").split("\n")

net = cv.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, threshold, nms, draw=True, objects=[]):
  classIds, confs, bbox = net.detect(img, confThreshold=threshold)
  print(classIds)
  print(confs)
  print(bbox)

  if len(objects) == 0:
    objects = classNames

  objectInfo = []

  if len(classIds) != 0:
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
      className = classNames[classId - 1]

      if className in objects:
        objectInfo.append([box, className])

        if draw: # 윈도우에 이미지 표시
          cv.rectangle(img, box, color=(0, 255, 0), thickness=2)
          cv.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
          cv.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

  return img, objectInfo

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--thres", type=float, default=0.45, help="Confidence threshold for object detection")
  parser.add_argument("--nms", type=float, default=0.2, help="NMS threshold for object detection")
  args = parser.parse_args()

  cap = cv.VideoCapture(0)
  cap.set(3, 640)
  cap.set(4, 480)

  iterations = 10
  avg = []

  while iterations:
    start = time.time()
    success, img = cap.read()
    result, objectInfo = getObjects(img, args.thres, args.nms)
    end = time.time()
    detectionTime = end - start
    print(f"thres={args.thres}, nms={args.nms}, each={detectionTime}")

    cv.imshow('output', img)
    cv.waitKey(1)
    iterations -= 1
    avg.append(detectionTime)

    print(f"thres={args.thres}, nms={args.nms}, avg={sum(avg) / len(avg)}")