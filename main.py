from recognition import create_datasets, text_to_csv, find_all_words
from nn import nn_parallel, nn_no_parallel
import cv2
import threading
import time

filename = 'samples\\disciplin.png'


def main_with_parallel():
    #create_dataset("alphabet.png")


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

    nn_parallel(filename, num_words, num_letters)

def main_no_parallel():
    # create_dataset("alphabet.png")

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

    nn_no_parallel(filename, num_words, num_letters)



if __name__ == '__main__' :
    #create_datasets()
    start_with_parallel = time.time()
    main_with_parallel()
    end_with_parallel = time.time()
    main_no_parallel()
    end_no_parallel = time.time()
    print('time results')
    print('no parallel ' + str(end_with_parallel-start_with_parallel) + ' seconds')
    print('with parallel ' + str(end_no_parallel - end_with_parallel) + ' seconds')

