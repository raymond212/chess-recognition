import cv2 as cv
import numpy as np
from main import predict, resize, size

label_short = {
    'wk': 'K',
    'wq': 'Q',
    'wr': 'R',
    'wb': 'B',
    'wn': 'N',
    'wp': 'P',
    'bk': 'k',
    'bq': 'q',
    'br': 'r',
    'bb': 'b',
    'bn': 'n',
    'bp': 'p',
    'e': '_'
}


def main():
    img = cv.imread('images/boards_testing/board5.png', )

    canny = cv.Canny(img, 125, 175)
    # cv.imshow('canny', canny)

    contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    contours.sort(key=lambda c: cv.contourArea(c), reverse=True)

    blank = np.zeros(img.shape, dtype='uint8')

    s = ''

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        print((x, y, w, h))
        if abs(w / h - 1) <= 0.01:
            cv.drawContours(img, [cnt], -1, (0, 255, 0), 1)
            gw, gh = w / 8, h / 8

            for row in range(8):
                for col in range(8):
                    x1, x2 = x + round(gw * col), x + round(gw * (col + 1))
                    y1, y2 = y + round(gh * row), y + round(gh * (row + 1))
                    # cv.circle(img, (x1, y1), 10, (0, 0, 255), -1)
                    # cv.circle(img, (x1, y2), 10, (0, 0, 255), -1)
                    # cv.circle(img, (x2, y1), 10, (0, 0, 255), -1)
                    # cv.circle(img, (x2, y2), 10, (0, 0, 255), -1)

                    cell = img[y1:y2, x1:x2]
                    cell = cv.resize(cell, dsize=(size, size), interpolation=cv.INTER_CUBIC)
                    # cv.imshow('cell', cell)
                    # cv.waitKey(0)
                    predicted_piece = label_short[predict(cell)]
                    # print(predicted_piece)
                    s += '|' + predicted_piece
                s += '|\n'

            print(gw, gh)

            print(s)

            break

    cv.imshow('board', img)

    cv.waitKey(0)


if __name__ == '__main__':
    main()