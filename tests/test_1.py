import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/milosz/RiSA_1/SW/train/2023-05-08 (11).jpg')
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #konwersja do skali szarości
th1 = 130
th2 = 100

def th1_callback(value):
    global th1
    th1 = value

def th2_callback(value):
    global th2
    th2 = value

cv2.namedWindow('Image')
cv2.createTrackbar('th1', 'Image', th1, 255, th1_callback)
cv2.createTrackbar('th2', 'Image', th2, 255, th2_callback)
'''
Transformata white top hat
'''

rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 33))
blackhat = cv2.morphologyEx(img_grey, cv2.MORPH_BLACKHAT, rectKern)

img_grey = cv2.blur(img_grey, (3,3))
squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
light = cv2.morphologyEx(img_grey, cv2.MORPH_CLOSE, squareKern)
light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# lines = cv2.HoughLinesP(light, 1, np.pi / 180, 110, minLineLength=200,maxLineGap=10)
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(light, (x1, y1), (x2, y2), (0, 255, 0), 2)

while True:
    cv2.imshow('Image', blackhat)
    cv2.imshow('Image_', light)
    if cv2.waitKey(10) == ord('q'):
        break