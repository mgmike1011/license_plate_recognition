import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/milosz/RiSA_1/SW/train/2023-05-08 (6).jpg')
img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #konwersja do skali szarości


# gw = 5
# gs = 0
# gw1 = 11
# gs1 = 2
# gw2 = 5
# gs2 = 3
# img_blur = cv2.GaussianBlur(img_grey, (gw, gw), gs)
# g1 = cv2.GaussianBlur(img_blur, (gw1, gw1), gs1)
# g2 = cv2.GaussianBlur(img_blur, (gw2, gw2), gs2)
# blur = g2-g1
# ret, thg = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# contours, hierarchy = cv2.findContours(thg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
# screenCnt = None
# # loop over our contours
# for c in cnts:
# # approximate the contour
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.018 * peri, True)
#     # if our approximated contour has four points, then
#     # we can assume that we have found our screen
#     if len(approx) == 4:
#         screenCnt = approx
#         break
#
# cv2.drawContours(img, [screenCnt], -1, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
gw = 23
gs = 21
gw1 = 63
gs1 = 6
gw2 = 14
gs2 = 6
th1 = 127
th2 = 255
def th1_callback(value):
    global th1
    th1 = value

def th2_callback(value):
    global th2
    th2 = value
def gw_callback(value):
    global gw
    if value % 2 == 0:
        gw = value + 1
    else:
        gw = value
def gs_callback(value):
    global gs
    gs = value
def gw1_callback(value):
    global gw1
    if value % 2 == 0:
        gw1 = value + 1
    else:
        gw1 = value
def gs1_callback(value):
    global gs1
    gs1 = value
def gw2_callback(value):
    global gw2
    if value % 2 == 0:
        gw2 = value + 1
    else:
        gw2 = value
def gs2_callback(value):
    global gs2
    gs2 = value

cv2.namedWindow('Image')
cv2.createTrackbar('th1', 'Image', th1, 255, th1_callback)
cv2.createTrackbar('th2', 'Image', th2, 255, th2_callback)
cv2.createTrackbar('gw', 'Image', gw, 100, gw_callback)
cv2.createTrackbar('gs', 'Image', gs, 100, gs_callback)
cv2.createTrackbar('gw1', 'Image', gw1, 100, gw1_callback)
cv2.createTrackbar('gs1', 'Image', gs1, 100, gs1_callback)
cv2.createTrackbar('gw2', 'Image', gw2, 100, gw2_callback)
cv2.createTrackbar('gs2', 'Image', gs2, 100, gs2_callback)


img_blur = cv2.GaussianBlur(img_grey, (gw, gw), gs)
g1 = cv2.GaussianBlur(img_blur, (gw1, gw1), gs1)
g2 = cv2.GaussianBlur(img_blur, (gw2, gw2), gs2)
blur = g2 - g1
ret, thg = cv2.threshold(blur, th1, th2, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
new_img = img.copy()
contours, hierarchy = cv2.findContours(thg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(new_img, contours, -1, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
screenCnt = None
# loop over our contours
for c in cnts:
# approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(new_img, [screenCnt], -1, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)


while True:

    cv2.imshow('Image', new_img)
    if cv2.waitKey(10) == ord('q'):
        break
