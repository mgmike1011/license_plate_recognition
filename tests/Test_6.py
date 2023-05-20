import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/milosz/RiSA_1/SW/train/2023-05-08 (11).jpg')
img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #konwersja do skali szarości
# ///////////////////
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
# ////////////////////
# Pierwszy blur
blur1 = cv2.GaussianBlur(img_grey, (51, 51), 0)
# Drugi blur
blur2 = cv2.GaussianBlur(img_grey, (3, 3), 0)
# Odjecie
blur = blur2 - blur1
# erode
# kernel = np.ones((3, 3), np.uint8)
# erosion = cv2.erode(blur, kernel)
# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
# Canny
edged = cv2.Canny(blur, 100, 100)
# Find
contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) == 4:
        print(f'approx = {approx}')
cv2.drawContours(img, contours, -1, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
'''
img_blur = cv2.GaussianBlur(image_copy, (gw, gw), gs)
g1 = cv2.GaussianBlur(img_blur, (gw1, gw1), gs1)
g2 = cv2.GaussianBlur(img_blur, (gw2, gw2), gs2)
ret, thg = cv2.threshold(g2-g1, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("image", thg)
cv2.imshow("twoja stara", g2-g1)

contours, hierarchy = cv2.findContours(thg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
'''
while True:

    cv2.imshow('Image', img)
    if cv2.waitKey(10) == ord('q'):
        break
