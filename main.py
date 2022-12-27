from create_dataset import create_dataset, text_to_csv
from nn import nn

if __name__ == '__main__' :
    #create_dataset("alphabet.png")
    text_to_csv('hello_world.png')
    nn()

    '''import pandas as pd
    ds = pd.read_csv('train.csv')
    data = list(ds.values)
    for i in range(5) :
        data += data
    pd.DataFrame(data=data, columns=ds.columns).to_csv('train1.csv', index=False)'''