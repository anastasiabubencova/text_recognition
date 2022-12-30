import cv2
from recognition import letters_extract
names = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя123'
names_dict = {}
for i in range(len(names)):
    names_dict[names[i]] = i
errors = [7, 8, 13, 32]

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

def create_datasets():
    path = "alphabets\\"
    files = ['arial', 'colibri', 'consolas', 'courer_new', 'georgia', 'lucida', 'tahoma', 'times', 'verdana']
    for file in files:
        file = path + file + '.png'
        img = cv2.imread(file)
        letters = letters_extract(img, out_size=28)
        letters = error_letters(letters)
        #to_png(letters)

    path = 'letters\\'
    files = ['arial', 'colibri', 'consolas', 'courer_new', 'georgia', 'lucida', 'tahoma', 'times', 'verdana']
    for n in files:
        n = path + n + '\\'
        names = range(32)
        data = []
        for i in names:
            img = cv2.imread(n + str(i) + '.png')  # , cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            g = [i]
            for j in img:
                g += list(j)
            data.append(g)

        labels = ['label']
        for i in range(1, 29):
            for j in range(1, 29):
                labels.append(str(i) + 'x' + str(j))

        import pandas as pd

        pd.DataFrame(data=data, columns=labels).to_csv(n + 'ds.csv', index=False)

    df = []
    names = ['arial', 'colibri', 'consolas', 'courer_new', 'georgia', 'lucida', 'tahoma', 'times', 'verdana']
    for i in names:
        i = path + i + '\\'
        df.append(pd.read_csv(i + 'ds.csv'))
    pd.concat(df).to_csv('all.csv', index=False)