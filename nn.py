import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as f_act

def init_net():
    input_nodes = 784
    hidden_nodes = 200
    out_nodes = 33
    learn_speed = 0.2
    return input_nodes, hidden_nodes, out_nodes, learn_speed

# установка случайных весов
def create_net(input_nodes, hidden_nodes, out_nodes):
    w_in2hidden = np.random.uniform(-0.5, 0.5, (hidden_nodes, input_nodes))
    w_hidden2out = np.random.uniform(-0.5, 0.5, (out_nodes, hidden_nodes))
    return w_in2hidden, w_hidden2out


# выход нейронной сети
def net_output(w_in2hidden, w_hidden2out, input_signal, return_hidden):
    inputs = np.array(input_signal, ndmin=2).T
    if inputs.size == 783 :
        '''print('inputs')
        print(inputs.size)
        print('w_in2hidden')
        print(w_in2hidden.size)'''
        inputs = np.append(inputs, np.array(1.))

    hidden_in = np.dot(w_in2hidden, inputs)
    hidden_out = f_act(hidden_in)
    final_in = np.dot(w_hidden2out, hidden_out)
    final_out = f_act(final_in)

    if return_hidden == 0:
        return final_out
    else:
        return final_out, hidden_out


# обучение
def net_train(target_list, input_signal, w_in2hidden, w_hidden2out, learn_speed):
    targets = np.array(target_list, ndmin=2).T
    inputs = np.array(input_signal, ndmin=2).T
    final_out, hidden_out = net_output(w_in2hidden, w_hidden2out, input_signal, 1)

    out_errors = targets - final_out
    hidden_errors = np.dot(w_hidden2out.T, out_errors)

    w_hidden2out += learn_speed * np.dot((out_errors * final_out * (1 - final_out)), hidden_out.T)
    w_in2hidden += learn_speed * np.dot((hidden_errors * hidden_out * (1 - hidden_out)), inputs.T)

    return w_in2hidden, w_hidden2out


# 6
def train_set(w_in2hidden, w_hidden2out, learn_speed):
    data_file = open("train.csv", 'r')
    training_list = data_file.readlines()[1:]
    data_file.close()

    index = int(0.8*len(training_list))
    test_list = training_list[index:]
    training_list = training_list[:index]

    for record in training_list:
        all_values = record.split(',')
        # range of input data is scaled from [0.0,255] to [0.001,1.0]
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.999) + 0.001

        targets = np.zeros(33) + 0.001
        # digits 0-9
        targets[int(all_values[0])] = 1.0
        net_train(targets, inputs, w_in2hidden, w_hidden2out, learn_speed)

    return w_in2hidden, w_hidden2out, test_list


# 7
def test_set(w_in2hidden, w_hidden2out, test_list):
    '''    data_file = open("predict.csv", 'r')
    test_list = data_file.readlines()[1:]
    data_file.close()'''
    test = []

    for record in test_list:
        all_values = record.split(',')
        # range of input data is scaled from [0.0,255] to [0.001,1.0]
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.999) + 0.001
        out_session = net_output(w_in2hidden, w_hidden2out, inputs, 0)

        if int(all_values[0]) == np.argmax(out_session):
            test.append(1)
        else:
            test.append(0)

    test = np.asarray(test)
    print('Net efficiency % =', (test.sum() / test.size) * 100)

#8
def plot_image(pixels: np.array):
    print(len(pixels))
    plt.imshow(pixels.reshape((28, 28)), cmap='gray')
    plt.show()


def nn():

    input_nodes, hidden_nodes, out_nodes, learn_speed = init_net()
    w_in2hidden, w_hidden2out = create_net(input_nodes, hidden_nodes, out_nodes)
    My_Variant = 5

    for i in range(5):
        print('epoch', i+1)
        _, _, test_list = train_set(w_in2hidden, w_hidden2out, learn_speed)
        test_set(w_in2hidden, w_hidden2out, test_list)

    data_file = open("predict.csv", 'r')
    predict_list = data_file.readlines()[1:]
    data_file.close()

    names = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    names_dict = {}
    for i in range(len(names)):
        names_dict[names[i]] = i

    text = ''

    for pred in predict_list:
        all_values = pred.split(',')
        inputs = (np.asfarray(all_values) / 255.0 * 0.999) + 0.001
        out_session = net_output(w_in2hidden, w_hidden2out, inputs, 0)

        text += str(list(names_dict.keys())[np.argmax(out_session)])

    print(text)