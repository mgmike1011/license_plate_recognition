import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/milosz/RiSA_1/SW/train/2023-05-08 (16).jpg')
img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #konwersja do skali szarości

'''
Nie działa: 3, 5, 12, 16
Wątpliwe: 6, 9
'''

#Sobel
Mx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_sobelx = np.abs(cv2.filter2D(img_grey, cv2.CV_32F, Mx) / 4.0)
img_sobely = np.abs(cv2.filter2D(img_grey, cv2.CV_32F, My) / 4.0)
img_sobelx_max = np.amax(img_sobelx)
img_sobely_max = np.amax(img_sobely)
img_sobelx = img_sobelx / img_sobelx_max * 255 #Przeskalowanie 0...2555
img_sobely = img_sobely / img_sobely_max * 255
M_xy_sobel = np.sqrt((cv2.filter2D(img_grey, cv2.CV_32F, Mx) / 4.0)**2 + (cv2.filter2D(img_grey, cv2.CV_32F, My) / 4.0)**2)
img_sobel = M_xy_sobel.astype(np.uint8)

ret, thg = cv2.threshold(img_sobel, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #127

contours, hierarchy = cv2.findContours(thg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
peri = cv2.arcLength(cnts[21], True)
approx = cv2.approxPolyDP(cnts[21], 0.018 * peri, True)
print(len(approx))
(x, y, w, h) = cv2.boundingRect(cnts[21])
# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
cv2.drawContours(img, [approx], -1, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# screenCnt = None
# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.018 * peri, True)
#     print(len(approx))
#     if len(approx) == 4:
#         (x, y, w, h) = cv2.boundingRect(c)
#         ar = w / float(h)
#         print(ar)
#         if ar < 1.5:
#             screenCnt = approx
#             # break
#
# cv2.drawContours(img, [screenCnt], -1, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

while True:
    cv2.imshow('Image', img)
    if cv2.waitKey(10) == ord('q'):
        break
