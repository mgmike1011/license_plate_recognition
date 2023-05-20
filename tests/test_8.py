import operator
import time
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/milosz/RiSA_1/SW/train/2023-05-08 (19).jpg')
t_start = time.perf_counter()
img = cv2.resize(img, (600, 450), interpolation=cv2.INTER_CUBIC)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #konwersja do skali szarości
# Gausowski blur
img_blur = cv2.GaussianBlur(img_grey, (3, 3), 0)

'''
Nie działa: 12, 13
Wątpliwe: 
Rozpoznane litery: 
'''
b1 = 6
canny1 = 130 #130
canny2 = 255

img_blur = cv2.GaussianBlur(img_grey, (3, 3), b1)
edged = cv2.Canny(img_blur, canny1, canny2)  # Perform Edge detection
dilation = cv2.dilate(edged, (3, 3), iterations=1)
# ret, thg = cv2.threshold(dilation, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #127
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
# cv2.drawContours(img, cnts, -1, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

screenCnt = None
aspect_ratio = 0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = float(w)/h
    if len(approx) == 4 and aspect_ratio >= 1.5:
        screenCnt = approx
        break
if screenCnt is None:
    Mx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    M_xy_sobel = np.sqrt(
        (cv2.filter2D(img_grey, cv2.CV_32F, Mx) / 4.0) ** 2 + (cv2.filter2D(img_grey, cv2.CV_32F, My) / 4.0) ** 2)
    img_sobel = M_xy_sobel.astype(np.uint8)

    ret, thg = cv2.threshold(img_sobel, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 127

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
# x,y,w,h = cv2.boundingRect(screenCnt) # <-- Get rectangle here
# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# Prostowanie
rows = 200
cols = 600

screenCnt_x = (sorted(screenCnt, key=lambda x: x[0,0])) #Sortowanie po x
# screenCnt_y = sorted(screenCnt, key=lambda x: x[0,1])

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
points = [Lg, Pg-2, Pd, Ld+3]
M = cv2.getPerspectiveTransform(np.float32(points), dest)
tablica = cv2.warpPerspective(img_grey, M, (cols, rows))

th1 = 0
th2 = 255

ret, thg = cv2.threshold(tablica, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #127
opening = cv2.morphologyEx(thg, cv2.MORPH_OPEN, (3, 3))
# dilation = cv2.dilate(opening, (3, 3), iterations=5)

# ret, thresh1 = cv2.threshold(tablica, 110, 255, cv2.THRESH_BINARY)
# dilation = cv2.dilate(thresh1, (3, 3), iterations=1)
#
# img_blur = cv2.GaussianBlur(dilation, (3, 3), b1)
# edged = cv2.Canny(img_blur, canny1, canny2)  # Perform Edge detection
#
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:50]
# cv2.drawContours(tablica, cnts, -1, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
litery = []
for c in cnts:
    area = cv2.contourArea(c)
    if 1670 < area < 12000:
        x, y, w, h = cv2.boundingRect(c) # <-- Get rectangle here
        # cv2.rectangle(tablica, (x, y), (x+w, y+h), (0, 255, 0), 2)
        letter = [c, x, y, w, h]
        litery.append(letter)
litery_ = sorted(litery, key=lambda x: x[1])
liter_to_pop = []
for i in range(len(litery_)-1):
    if (litery_[i+1][1]+litery_[i+1][3]) < (litery_[i][1]+litery_[i][3]):
        liter_to_pop.append(i+1)

# cv2.rectangle(tablica, (litery_[d][1], litery_[d][2]), (litery_[d][1]+litery_[d][3], litery_[d][2]+litery_[d][4]), (0, 255, 0), 2)
for index in sorted(liter_to_pop, reverse=True):
    del litery_[index]

for d in range(len(litery_)):
    cv2.rectangle(tablica, (litery_[d][1], litery_[d][2]), (litery_[d][1]+litery_[d][3], litery_[d][2]+litery_[d][4]), (0, 255, 0), 2)


t_stop = time.perf_counter()
print(f'test_8: {t_stop - t_start} s')
while True:

    cv2.imshow('Image', tablica)
    if cv2.waitKey(10) == ord('q'):
        break
