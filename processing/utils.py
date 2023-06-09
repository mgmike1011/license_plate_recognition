import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity


def perform_processing(image: np.ndarray) -> str:
    # print(f'image.shape: {image.shape}')
    # Template MIX
    MIX = 'processing/MIX'
    mix_letters = os.listdir(MIX)
    #
    # Plate localization
    #
    img = cv2.resize(image, (600, 450), interpolation=cv2.INTER_CUBIC)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Parametrization
    b1 = 6
    canny1 = 130
    canny2 = 255
    rows = 200
    cols = 500
    # Gaussian blur
    img_blur = cv2.GaussianBlur(img_grey, (3, 3), b1)
    # Canny
    edged = cv2.Canny(img_blur, canny1, canny2)
    # Dilatation
    dilation = cv2.dilate(edged, (3, 3), iterations=1)
    # Finding licence plate contour
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    screenCnt = None
    aspect_ratio = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if len(approx) == 4 and aspect_ratio >= 1.5:
            screenCnt = approx
            break
    # If the contour is not found
    if screenCnt is None:
        Mx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        M_xy_sobel = np.sqrt(
            (cv2.filter2D(img_grey, cv2.CV_32F, Mx) / 4.0) ** 2 + (cv2.filter2D(img_grey, cv2.CV_32F, My) / 4.0) ** 2)
        img_sobel = M_xy_sobel.astype(np.uint8)
        ret, thg = cv2.threshold(img_sobel, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break
    if screenCnt is None:
        return "PO12345"
    # Sorting by te x coordinate
    screenCnt_x = (sorted(screenCnt, key=lambda x: x[0, 0]))
    # Rectification
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
    dest = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    points = [Lg, Pg - 2, Pd, Ld + 3]
    M = cv2.getPerspectiveTransform(np.float32(points), dest)
    tablica = cv2.warpPerspective(img_grey, M, (cols, rows))
    #
    # Finding letters
    #
    ret, thg = cv2.threshold(tablica, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(thg, cv2.MORPH_OPEN, (3, 3))
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:50]
    litery = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 1460 < area < 12000:
            x, y, w, h = cv2.boundingRect(c)
            letter = [c, x, y, w, h]
            litery.append(letter)
    litery_ = sorted(litery, key=lambda x: x[1])
    liter_to_pop = []
    for i in range(len(litery_) - 1):
        if (litery_[i + 1][1] + litery_[i + 1][3]) < (litery_[i][1] + litery_[i][3]):
            liter_to_pop.append(i + 1)
    for index in sorted(liter_to_pop, reverse=True):
        del litery_[index]
    out_letters = ''
    for d in range(len(litery_)):
        p1 = tablica[litery_[d][2]:(litery_[d][2] + litery_[d][4]), litery_[d][1]:(litery_[d][1] + litery_[d][3])]
        _, p1 = cv2.threshold(p1, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        p1 = cv2.resize(p1, (128, 151), interpolation=cv2.INTER_CUBIC)
        biggest_similarity = 0
        biggest_similarity_letter = ''
        for l in mix_letters:
            similarity = structural_similarity(p1, cv2.imread(MIX + '/' + l, cv2.IMREAD_GRAYSCALE))
            if similarity >= biggest_similarity:
                biggest_similarity = similarity
                biggest_similarity_letter = l[0]
        out_letters += biggest_similarity_letter
    # Verification
    if len(out_letters) == 0:
        out_letters += 'P'
    if (len(out_letters) < 7) and out_letters[0] == 'L':
        out_letters = 'P' + out_letters
    if len(out_letters) < 7:
        for _ in range(7-len(out_letters)):
            out_letters += 'N'
    if out_letters[0] == '0':
        out_letters = list(out_letters)
        out_letters[0] = 'O'
        out_letters = ''.join(out_letters)
    elif out_letters[1] == '0':
        out_letters = list(out_letters)
        out_letters[1] = 'O'
        out_letters = ''.join(out_letters)
    elif out_letters[2] == '0':
        out_letters = list(out_letters)
        out_letters[2] = 'O'
        out_letters = ''.join(out_letters)
    elif out_letters[3] == 'O':
        out_letters = list(out_letters)
        out_letters[3] = '0'
        out_letters = ''.join(out_letters)
    elif out_letters[4] == 'O':
        out_letters = list(out_letters)
        out_letters[4] = '0'
        out_letters = ''.join(out_letters)
    elif out_letters[5] == 'O':
        out_letters = list(out_letters)
        out_letters[5] = '0'
        out_letters = ''.join(out_letters)
    elif out_letters[6] == 'O':
        out_letters = list(out_letters)
        out_letters[6] = '0'
        out_letters = ''.join(out_letters)
    return out_letters
