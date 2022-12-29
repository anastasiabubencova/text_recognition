from recognition import create_dataset, text_to_csv, find_all_words
from nn import nn
import cv2
import threading

if __name__ == '__main__' :

    #create_dataset("alphabet.png")

    filename = 'samples\\vgrug2.png'
    image = cv2.imread(filename)

    all_words, num_strings, num_words = find_all_words(image)

    def parallel_part(i, results_text, results_num_letters):
        results_text[i], results_num_letters[i] = text_to_csv(all_words[i], filename)

    results_text = {}
    results_num_letters = {}

    threads = [threading.Thread(target=parallel_part(i, results_text, results_num_letters)) for i in range(len(all_words))]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    text = ''
    num_letters = list(results_num_letters.values())
    for t in list(results_text.values()):
        text += t

    labels = ''
    for i in range(1, 29):
        for j in range(1, 29):
            labels += str(i) + 'x' + str(j) + ','
    labels += '\n'

    with open(filename + '.csv', "w") as f:
        f.write(labels + text)
        f.truncate()

    nn(filename, num_words, num_letters)

