import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/milosz/RiSA_1/SW/train/2023-05-08 (19).jpg')
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #konwersja do skali szarości


'''
Rozmywanie -> Canny -> Kontury ->sortowanie konturów względem area -> Długość łuku -> aproksymacja -> maska
Działa: 2, 3, 5, 7, 8, 11, 14, 18, 19
Nie działa: 1, 4, 6, 9, 10, 12, 13, 15, 16, 17, 
'''
img_grey = cv2.bilateralFilter(img_grey, 11, 17, 17)
edged = cv2.Canny(img_grey, 30, 200)

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]

screenCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

mask = np.zeros(img_grey.shape, np.uint8)
new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
new_image = cv2.bitwise_and(img, img, mask=mask)

while True:
    cv2.imshow('Image', new_image)
    if cv2.waitKey(10) == ord('q'):
        break
