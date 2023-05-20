import time
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/milosz/RiSA_1/SW/train/2023-05-08 (9).jpg')
t_start = time.perf_counter()
img = cv2.resize(img, (600, 450), interpolation=cv2.INTER_CUBIC)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #konwersja do skali szarości

'''
Nie działa: 3, 5, 12, 16
Wątpliwe: 6, 9
'''

#Sobel
Mx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
M_xy_sobel = np.sqrt((cv2.filter2D(img_grey, cv2.CV_32F, Mx) / 4.0)**2 + (cv2.filter2D(img_grey, cv2.CV_32F, My) / 4.0)**2)
img_sobel = M_xy_sobel.astype(np.uint8)

ret, thg = cv2.threshold(img_sobel, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #127

contours, hierarchy = cv2.findContours(thg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
# cv2.drawContours(img, cnts, -1, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
screenCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(img, [screenCnt], -1, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# Prostowanie
rows = 200
cols = 600

screenCnt_x = (sorted(screenCnt, key=lambda x: x[0,0])) #Sortowanie po x

if screenCnt_x[0][0][1] > screenCnt_x[1][0][1]:
    Lg = screenCnt_x[1]
    Ld = screenCnt_x[0]
else:
    Lg = screenCnt_x[0]
    Ld = screenCnt_x[1]
if screenCnt_x[2][0][1] > screenCnt_x[3][0][1]:
    Pg = screenCnt_x[3]
    Pd = screenCnt_x[2]
else:
    Pg = screenCnt_x[2]
    Pd = screenCnt_x[3]
dest = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]]) #moga byc 200 x 900
points = [Lg,Pg,Pd,Ld]
M = cv2.getPerspectiveTransform(np.float32(points), dest)
img = cv2.warpPerspective(img, M, (cols, rows))
t_stop = time.perf_counter()
print(f'Linear: {t_stop - t_start} s')
while True:
    cv2.imshow('Image', img)
    if cv2.waitKey(10) == ord('q'):
        break
