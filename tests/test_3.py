import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/milosz/RiSA_1/SW/train/2023-05-08 (1).jpg')
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #konwersja do skali szarości


'''
Adaptacyjne progowanie -> dylatacja -> kontury -> Długość łuku -> aproksymacja -> sortowanie kształtów 
Działa: 
Nie działa: 19, 
'''
kernel = np.ones((2, 2), np.uint8)
thresh1 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
thresh1 = cv2.dilate(thresh1, kernel)
# thresh1 = cv2.blur(thresh1, (3, 3))

contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
screenCnt = []

for c in contours:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    if len(approx) == 4:
        screenCnt.append(approx)

screenCnt = sorted(screenCnt, key=cv2.contourArea, reverse=True)
cv2.drawContours(img, screenCnt, -1, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

while True:
    cv2.imshow('Image', img)
    if cv2.waitKey(10) == ord('q'):
        break
