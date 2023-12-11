import cv2
import numpy as np
import dlib
from math import hypot
import time
import pyglet
import cvlib as cv
capture = cv2.VideoCapture(0)

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("/Users/mohammedqalandarshahquazi/Desktop/INFINITEEYES/eye-detectin final/shape_predictor_68_face_landmarks.dat")

def point(a1 ,a2):
    return int((a1.x + a2.x)/2), int((a1.y + a2.y)/2)

while True:
    _, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detect(gray)
    for face in faces:
        x, y = face.left(), face.top()
        a1, b1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (a1, b1), (0, 255, 0), 2)

        landmarks = predict(gray, face)
        left = (landmarks.part(36).x, landmarks.part(36).y)
        right = (landmarks.part(39).x, landmarks.part(39).y)
        top = point(landmarks.part(37), landmarks.part(38))
        bottom = point(landmarks.part(41), landmarks.part(40))

        hor = cv2.line(frame, left, right, (0, 255, 0), 2)
        ver= cv2.line(frame, top, bottom, (0, 255, 0), 2)
        bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov3-tiny')
        print(bbox, label, conf)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27 and 0xFF == ord('q'):
        break
# bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov3-tiny')
# print(bbox, label, conf)
#print(hor,ver)

capture.release()
cv2.destroyAllWindows()