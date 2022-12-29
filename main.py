from recognition import create_dataset, text_to_csv, find_all_words
from nn import nn
import cv2

if __name__ == '__main__' :

    #create_dataset("alphabet.png")

    filename = 'samples\\vgrug2.png'
    image = cv2.imread(filename)

    all_words, num_strings, num_words = find_all_words(image)

    text = ''
    num_letters = []
    for i in range(len(all_words)):
        tmp_text, tmp_len_let = text_to_csv(all_words[i], filename)
        text += tmp_text
        num_letters.append(tmp_len_let)


    labels = ''
    for i in range(1, 29):
        for j in range(1, 29):
            labels += str(i) + 'x' + str(j) + ','
    labels += '\n'

    with open(filename + '.csv', "w") as f:
        f.write(labels + text)
        f.truncate()



    nn(filename, num_words, num_letters)

