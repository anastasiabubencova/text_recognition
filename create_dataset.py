import cv2
import numpy as np

names = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
names_dict = {}
for i in range(len(names)):
    names_dict[names[i]] = i
errors = [7, 8, 13, 32]


def letters_extract(image_file: str, out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    cv2.imshow("Output", output)
    cv2.waitKey(0)

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    return letters


def to_png(letters):
    # сохраняем картинки 28х28
    c = 0

    for i in range(len(letters)):
        if i not in errors:
            cv2.imwrite('letters\\' + str(names_dict[names[c]]) + '.png', letters[i][2])
            c += 1


def to_csv(letters, type_of_dataset):
    letters_to_csv = []
    for letter in letters:
        if letters.index(letter) not in errors:
            if type_of_dataset == 'train':
                letters_to_csv.append([names_dict[names[len(letters_to_csv)]]])
            else:
                letters_to_csv.append([])
            for pixel in letter[2]:
                letters_to_csv[-1] += list(pixel)

    labels = []

    if type_of_dataset == 'train':
        for i in range(0) :
            letters_to_csv += letters_to_csv
        labels += ['label']

    for i in range(1, 29):
        for j in range(1, 29):
            labels.append(str(i) + 'x' + str(j))

    import pandas as pd
    pd.DataFrame(data=letters_to_csv, columns=labels).to_csv(type_of_dataset + '.csv', index=False)


def create_dataset(image_file):
    letters = letters_extract(image_file, out_size=28)
    to_png(letters)
    letters = error_letters(letters)
    to_csv(letters, 'train')


def text_to_csv(image_file):
    letters = letters_extract(image_file, out_size=28)
    to_csv(letters, 'predict')

def error_letters(letters):
    e_l = [6, 12, 31]
    for i in e_l:
        img = cv2.imread('error_letters\\' + str(i) + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        '''print('----------------------------' + str(i) + '--------------------------------------------')
        iimmgg = []
        for c in img:
            iimmgg += list(c)
        print(iimmgg)'''
    return letters

def tuple_to_list(let) :
    letters = []
    for i in range(len(let)):
        letters.append([])
        for j in range(len(let[i][2])):
            letters[i].append(let[i][2][j])
    return letters