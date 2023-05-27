import operator
import time
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

# Wczytanie zdjęcia
template = cv2.imread('Template_2.png')
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template_gray = cv2.resize(template_gray, (0, 0), fx=0.3, fy=0.3)
letter = cv2.imread('/home/milosz/Pictures/Screenshots/out_18/2.png')
letter_gray = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)

w, h = letter_gray.shape[::-1]
res = cv2.matchTemplate(template_gray, letter_gray, cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where(res >= threshold)
letter_read = {}
for pt in zip(*loc[::-1]):
    cv2.rectangle(template_gray, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
    if 0 <= pt[0] < 10:
        if '0' in letter_read:
            letter_read['0'] += 1
        else:
            letter_read['0'] = 1
    elif 124 < pt[0] < 133:
        if '1' in letter_read:
            letter_read['1'] += 1
        else:
            letter_read['1'] = 1
    elif 250 < pt[0] < 260:
        if '2' in letter_read:
            letter_read['2'] += 1
        else:
            letter_read['2'] = 1
    elif 380 < pt[0] < 390:
        if '3' in letter_read:
            letter_read['3'] += 1
        else:
            letter_read['3'] = 1
    elif 509 < pt[0] < 516:
        if '4' in letter_read:
            letter_read['4'] += 1
        else:
            letter_read['4'] = 1
    elif 635 < pt[0] < 645:
        if '5' in letter_read:
            letter_read['5'] += 1
        else:
            letter_read['5'] = 1
    elif 764 < pt[0] < 771:
        if '6' in letter_read:
            letter_read['6'] += 1
        else:
            letter_read['6'] = 1
    elif 892 < pt[0] < 899:
        if '7' in letter_read:
            letter_read['7'] += 1
        else:
            letter_read['7'] = 1
    elif 1020 < pt[0] < 1030:
        if '8' in letter_read:
            letter_read['8'] += 1
        else:
            letter_read['8'] = 1
    elif 1148 < pt[0] < 1155:
        if '9' in letter_read:
            letter_read['9'] += 1
        else:
            letter_read['9'] = 1
    elif 1276 < pt[0] < 1283:
        if 'A' in letter_read:
            letter_read['A'] += 1
        else:
            letter_read['A'] = 1
    elif 1404 < pt[0] < 1411:
        if 'B' in letter_read:
            letter_read['B'] += 1
        else:
            letter_read['B'] = 1
    elif 1533 < pt[0] < 1539:
        if 'C' in letter_read:
            letter_read['C'] += 1
        else:
            letter_read['C'] = 1
    elif 1660 < pt[0] < 1668:
        if 'D' in letter_read:
            letter_read['D'] += 1
        else:
            letter_read['D'] = 1
    elif 1788 < pt[0] < 1795:
        if 'E' in letter_read:
            letter_read['E'] += 1
        else:
            letter_read['E'] = 1
    elif 1915 < pt[0] < 1925:
        if 'F' in letter_read:
            letter_read['F'] += 1
        else:
            letter_read['F'] = 1
    elif 2043 < pt[0] < 2053:
        if 'G' in letter_read:
            letter_read['G'] += 1
        else:
            letter_read['G'] = 1
    elif 2173 < pt[0] < 2179:
        if 'H' in letter_read:
            letter_read['H'] += 1
        else:
            letter_read['H'] = 1
    elif 2300 < pt[0] < 2308:
        if 'I' in letter_read:
            letter_read['I'] += 1
        else:
            letter_read['I'] = 1
    elif 2429 < pt[0] < 2435:
        if 'J' in letter_read:
            letter_read['J'] += 1
        else:
            letter_read['J'] = 1
    elif 2550 < pt[0] < 2565:
        if 'K' in letter_read:
            letter_read['K'] += 1
        else:
            letter_read['K'] = 1
    elif 2683 < pt[0] < 2690:
        if 'L' in letter_read:
            letter_read['L'] += 1
        else:
            letter_read['L'] = 1
    elif 2809 < pt[0] < 2820:
        if 'M' in letter_read:
            letter_read['M'] += 1
        else:
            letter_read['M'] = 1
    elif 2940 < pt[0] < 2948:
        if 'N' in letter_read:
            letter_read['N'] += 1
        else:
            letter_read['N'] = 1
    elif 3069 < pt[0] < 3075:
        if 'O' in letter_read:
            letter_read['O'] += 1
        else:
            letter_read['O'] = 1
    elif 3196 < pt[0] < 3205:
        if 'P' in letter_read:
            letter_read['P'] += 1
        else:
            letter_read['P'] = 1
    elif 3324 < pt[0] < 3331:
        if 'R' in letter_read:
            letter_read['R'] += 1
        else:
            letter_read['R'] = 1
    elif 3450 < pt[0] < 3460:
        if 'S' in letter_read:
            letter_read['S'] += 1
        else:
            letter_read['S'] = 1
    elif 3580 < pt[0] < 3587:
        if 'T' in letter_read:
            letter_read['T'] += 1
        else:
            letter_read['T'] = 1
    elif 3709 < pt[0] < 3716:
        if 'U' in letter_read:
            letter_read['U'] += 1
        else:
            letter_read['U'] = 1
    elif 3836 < pt[0] < 3843:
        if 'W' in letter_read:
            letter_read['W'] += 1
        else:
            letter_read['W'] = 1
    elif 3964 < pt[0] < 3971:
        if 'Y' in letter_read:
            letter_read['Y'] += 1
        else:
            letter_read['Y'] = 1
    elif 4062 < pt[0] < 4100:
        if 'Z' in letter_read:
            letter_read['Z'] += 1
        else:
            letter_read['Z'] = 1
    else:
        print(pt)
        if 'X' in letter_read:
            letter_read['X'] += 1
        else:
            letter_read['X'] = 1
print(letter_read)
out_letter = max(letter_read, key=letter_read.get)
a = ''
a += out_letter
print(str(a))
# template_gray = cv2.resize(template_gray, (0, 0), fx=0.3, fy=0.3)
while True:
    cv2.imshow('Image', template_gray)
    if cv2.waitKey(10) == ord('q'):
        break

