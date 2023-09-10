import cv2 as cv
import numpy as np
from main import predict, resize

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
    img = cv.imread('images/boards_testing/board8.png', )

    canny = cv.Canny(img, 125, 175)
    # cv.imshow('canny', canny)

    contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    contours.sort(key=lambda c: cv.contourArea(c), reverse=True)

    blank = np.zeros(img.shape, dtype='uint8')

    s = ''

    board_cnt = None

    for cnt in contours:
        _, _, w, h = cv.boundingRect(cnt)
        if h > 0 and abs(w / h - 1) <= 0.05:
            board_cnt = cnt
            break

    if board_cnt is None:
        print("No chessboard detected.")
    else:
        x, y, w, h = cv.boundingRect(board_cnt)
        cv.drawContours(img, [board_cnt], -1, (0, 255, 0), 1)

        gw, gh = w / 8, h / 8

        board = [[""] * 8 for _ in range(8)]

        for row in range(8):
            for col in range(8):
                x1, x2 = x + round(gw * col), x + round(gw * (col + 1))
                y1, y2 = y + round(gh * row), y + round(gh * (row + 1))

                cv.circle(img, (x1, y1), 10, (0, 0, 255), -1)
                cv.circle(img, (x1, y2), 10, (0, 0, 255), -1)
                cv.circle(img, (x2, y1), 10, (0, 0, 255), -1)
                cv.circle(img, (x2, y2), 10, (0, 0, 255), -1)

                cell = resize(img[y1:y2, x1:x2])
                # cv.imshow('cell', cell)
                # cv.waitKey(0)
                predicted_piece = label_short[predict(cell)]
                board[row][col] = predicted_piece

        print('\n'.join([f'|{"|".join(rank)}|' for rank in board]))
        print(board_to_fen(board) + " w - - 0 1")

        cv.imshow('board', img)
        cv.waitKey(0)


def board_to_fen(board):
    res = []
    for rank in board:
        rank_str = ""
        empty_counter = 0
        for square in rank:
            if square == "_":
                empty_counter += 1
                continue
            if empty_counter > 0:
                rank_str += str(empty_counter)
                empty_counter = 0
            rank_str += square
        if empty_counter > 0:
            rank_str += str(empty_counter)
        res.append(rank_str)
    return '/'.join(res)



if __name__ == '__main__':
    main()
