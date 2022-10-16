import pickle
from os import path

import numpy as np

from utilities.utils import to_categorical

TOTAL_BATCHES= 5
NUM_CLASSES = 10
BATCH_SIZE = 10000
DIMENSIONS = 32*32*3
FILE_NAMES={
    "training":"data_batch_",
    "testing":"test_batch"
}


def unpickle(file,num_samples=10000):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict[b"data"],to_categorical(dict[b'labels'], NUM_CLASSES)


def get_data(data_path="data",num_samples=50000,dataset="training"):
    data= np.zeros(shape=(num_samples,DIMENSIONS))
    labels=np.zeros(shape=(NUM_CLASSES,num_samples))
    num_batches = num_samples//BATCH_SIZE

    for i in range(num_batches):
        file_name = f"{FILE_NAMES['training']}{i+1}" if dataset=="training" else FILE_NAMES["testing"]
        file_path = path.join(".",data_path,file_name)
        ret_val = unpickle(file_path)
        data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]=ret_val[0]
        labels[:,i*BATCH_SIZE:(i+1)*BATCH_SIZE]=ret_val[1]
    
    return data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32), labels.T



def test():
    for _ in range(10):
        print(_)
        print(type(_))

if __name__ == "__main__":
    test()