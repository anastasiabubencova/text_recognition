import cv2
import numpy as np

names = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
names_dict = {}
for i in range(len(names)):
    names_dict[names[i]] = i
errors = [7, 8, 13, 32]

def find_text_areas(image, iter) :
    # Load image, grayscale, Gaussian blur, adaptive threshold

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    dilate = cv2.dilate(thresh, kernel, iterations=iter)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if iter <= 2 :
        text_areas = {}
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
            text_areas[x] = image[y:y+h, x:x+w]
            #cv2.imshow('p', image[y:y + h, x:x + w])
            #cv2.waitKey(0)


        text_areas = sorted(text_areas.items())
        just_images = []
        for area in text_areas:
            just_images.append(area[1])
        return just_images
    else :
        text_areas = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
            text_areas.append(image[y:y + h, x:x + w])

            #cv2.imshow('p', image[y:y + h, x:x + w])
            #cv2.waitKey(0)

        return text_areas[::-1]

def find_all_words(image):
    # строки
    strings = find_text_areas(image, 5)
    num_strings = len(strings)

    # слова в строках
    words = [find_text_areas(string, 2) for string in strings]
    num_words = [len(word) for word in words]

    all_words = []
    for word in words:
        all_words += word

    return all_words, num_strings, num_words

def letters_extract(img, out_size=28):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((7, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    x_w_dict = {}
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:


            x_w_dict[x] = w


            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

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
        if (type_of_dataset == 'train') :#& (letters.index(letter) not in errors):
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

    text = ''
    for i in letters_to_csv:
        for j in i:
            text += str(j) + ','
        text += '\n'

    return text

def create_dataset(image_file):
    letters, spaces = letters_extract(image_file, out_size=28)
    to_png(letters)
    letters = error_letters(letters)
    to_csv(letters, 'train')

'''def find_letters(img):
    letters = find_text_areas(img, 1)
    for i in range(len(letters)):
        #gray = cv2.cvtColor(letters[i], cv2.COLOR_BGR2GRAY)
        #_, letters[i] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        #letters[i] = cv2.resize(letters[i], (28, 28), interpolation=cv2.INTER_AREA)
        #cv2.imshow('o', letters[i])
    #cv2.waitKey(0)
    return letters'''

def text_to_csv(img, filename):
    letters = letters_extract(img, out_size=28)
    return to_csv(letters, filename), len(letters)

def error_letters(letters):
    e_l = [6, 12, 31]
    for i in e_l:
        img = cv2.imread('error_letters\\' + str(i) + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        '''iimmgg = []
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