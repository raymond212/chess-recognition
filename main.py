import keras
import requests
import numpy as np
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from keras.models import model_from_json
from pathlib import Path

pieces = ['wk', 'wq', 'wr', 'wb', 'wn', 'wp', 'bk', 'bq', 'br', 'bb', 'bn', 'bp']
idx_to_label = pieces + ['e']
label_to_idx = {idx_to_label[i]: i for i in range(len(idx_to_label))}

piece_styles = ['neo', 'game_room', 'wood', 'glass', 'classic']  # 'metal', 'bases', 'neo_wood', 'icy_sea']
board_styles = ['green', 'dark_wood', 'glass', 'brown', 'icy_sea']

size = 32


def scrape_images():
    for piece_style in piece_styles:
        for piece in pieces:
            url = f'https://images.chesscomfiles.com/chess-themes/pieces/{piece_style}/{size}/{piece}.png'
            img_bytes = requests.get(url).content
            location = f'images/pieces/{piece}_{piece_style}.png'
            save_img(img_bytes, location)

    for board_style in board_styles:
        url = f'https://images.chesscomfiles.com/chess-themes/boards/{board_style}/{size}.png'
        img_bytes = requests.get(url).content
        location = f'images/full_boards/{board_style}.png'
        save_img(img_bytes, location)

        full_board = cv.imdecode(np.frombuffer(img_bytes, np.uint8), cv.IMREAD_COLOR)
        white_square = full_board[0:size, 0:size]
        black_square = full_board[0:size, size:2 * size]

        cv.imwrite(f'images/board_squares/{board_style}_w.png', white_square)
        cv.imwrite(f'images/board_squares/{board_style}_b.png', black_square)


def generate_data():
    x_data = []
    y_data = []

    for board_style in board_styles:
        for board_color in ['w', 'b']:
            board_img = cv.imread(f'images/board_squares/{board_style}_{board_color}.png')
            cv.imwrite(f'images/training/{board_style}_{board_color}.png', board_img)
            x_data.append(board_img)
            y_data.append([label_to_idx['e']])

            for piece_style in piece_styles:
                for piece in pieces:
                    piece_img = cv.imread(f'images/pieces/{piece}_{piece_style}.png', cv.IMREAD_UNCHANGED)
                    training_img = overlay(piece_img, board_img)

                    cv.imwrite(f'images/training/{piece}_{piece_style}_{board_style}_{board_color}.png', training_img)

                    x_data.append(training_img)
                    y_data.append([label_to_idx[piece]])

    x_data = np.array(x_data)
    x_data = x_data.astype('float32') / 255

    y_data = np.array(y_data)
    y_data = keras.utils.to_categorical(y_data, 13)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

    save_np(x_train, 'x_train.npy')
    save_np(x_test, 'x_test.npy')

    save_np(y_train, 'y_train.npy')
    save_np(y_test, 'y_test.npy')


def train():
    x_train = load_np('x_train.npy')
    y_train = load_np('y_train.npy')

    model = keras.models.Sequential()
    # model.add(Conv2D(size, (3, 3), padding='same', input_shape=(size, size, 3), activation='relu'))
    # model.add(Conv2D(size, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    model.add(Flatten(input_shape=(size, size, 3)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(13, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    BATCH_SIZE = 16
    EPOCHS = 20

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2
    )

    model_structure = model.to_json()
    f = Path('model_structure.json')
    f.write_text(model_structure)

    model.save_weights("model_weights.h5")


def test():
    f = Path('model_structure.json')
    model = model_from_json(f.read_text())
    model.load_weights('model_weights.h5')

    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')

    results = model.predict(x_test)

    cnt = 0

    for i, result in enumerate(results):
        predicted_idx = int(np.argmax(result))

        if y_test[i][predicted_idx] == 1:
            cnt += 1

    print(f'Predicted {cnt} out of {len(results)} correctly - {100 * cnt / len(results):.3f}%')


def test_additional():
    f = Path('model_structure.json')
    model = model_from_json(f.read_text())
    model.load_weights('model_weights.h5')

    img_arr_list = []
    f_list = []

    for f in os.listdir('images/testing'):
        img_arr_list.append(resize(path_to_img_array(f'images/testing/{f}')))
        f_list.append(f)

    img_arr_list = np.array(img_arr_list)
    results = model.predict(img_arr_list)

    cnt = 0

    for i, result in enumerate(results):
        predicted_idx = int(np.argmax(result))
        likelihood = result[predicted_idx]
        class_label = idx_to_label[predicted_idx]
        correct = f_list[i].startswith(class_label)

        if correct:
            cnt += 1

        print(f'{f_list[i]:15} - predicted: {class_label:2} - {likelihood * 100:>6.2f}% - {correct}')

    print(f'Predicted {cnt} out of {len(results)} correctly - {100 * cnt / len(results):.3f}%')


def predict(img_arr):
    f = Path('model_structure.json')
    model = model_from_json(f.read_text())
    model.load_weights('model_weights.h5')

    result = model(np.array([img_arr]))[0]

    predicted_idx = int(np.argmax(result))
    return idx_to_label[predicted_idx]


def path_to_img_array(directory):
    img = cv.imread(directory)
    return img


def resize(img_arr):
    return cv.resize(img_arr, dsize=(size, size), interpolation=cv.INTER_CUBIC)


def overlay(foreground, background):
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255
    alpha_mask = alpha_channel[:, :, np.newaxis]
    composite = background * (1 - alpha_mask) + foreground_colors * alpha_mask
    return composite.astype('uint8')


def load_np(location):
    with open(location, 'rb') as f:
        return np.load(f)


def save_np(np_arr, location):
    with open(location, 'wb') as f:
        np.save(f, np_arr)


def save_img(data, location):
    f = open(location, 'wb')
    f.write(data)
    f.close()


if __name__ == '__main__':
    # scrape_images()
    # generate_data()
    # train()
    # test()
    test_additional()
