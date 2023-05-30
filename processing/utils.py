import numpy as np
import cv2


def perform_processing(image: np.ndarray) -> str:
    print(f'image.shape: {image.shape}')
    template = cv2.imread('processing/Template_5.png')
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
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
        # p1 = np.zeros((litery_[d][4], litery_[d][3]), dtype=np.uint8)
        p1 = tablica[litery_[d][2]:(litery_[d][2] + litery_[d][4]), litery_[d][1]:(litery_[d][1] + litery_[d][3])]
        _, p1 = cv2.threshold(p1, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        p1 = cv2.resize(p1, (128, 151), interpolation=cv2.INTER_CUBIC)
        # w, h = p1.shape[::-1]
        res = cv2.matchTemplate(template_gray, p1, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        letter_read = {}
        for pt in zip(*loc[::-1]):
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
                if 'V' in letter_read:
                    letter_read['V'] += 1
                else:
                    letter_read['V'] = 1
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
        letter_read['W'] = 1
        out_letter = max(letter_read, key=letter_read.get)
        out_letters += out_letter
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
    return out_letters
