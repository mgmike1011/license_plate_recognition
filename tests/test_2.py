import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/milosz/RiSA_1/SW/train/2023-05-08 (19).jpg')
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #konwersja do skali szarości


'''
progowanie -> dylacja -> kontury -> Długość łuku -> aproksymacja -> sortowanie kształtów 
Działa: 1, 6, 8, 11, 12, 13, 14, 19
Nie działa: 2, 3, 4, 5, 7, 9, 10, 15, 16, 17, 18
'''
th1 = 130
th2 = 100
kernel = np.ones((3, 3), np.uint8)
ret, thresh1 = cv2.threshold(img_grey, th1, 255, cv2.THRESH_BINARY)
thresh1 = cv2.dilate(thresh1, kernel, iterations=1)

contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
screenCnt = []

for c in contours:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * perimeter, True)
    if len(approx) == 4:
        screenCnt.append(approx)

screenCnt = sorted(screenCnt, key=cv2.contourArea, reverse=True)[:1]
cv2.drawContours(img, screenCnt, -1, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

while True:
    cv2.imshow('Image', img)
    if cv2.waitKey(10) == ord('q'):
        break
