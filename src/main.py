import numpy as np

from layers.activations import Relu, Softmax
from layers.conv import ConvLayer
from layers.fc import FCLayer
from layers.flatten import Flatten
from layers.pool import PoolLayer
from losses.categorical_crossentropy import CategoricalCrossEntropy
from model import Model
from utilities.file_reader import get_data

if __name__ == "__main__":
    train_data, train_labels = get_data(num_samples=50000)
    test_data, test_labels = get_data(num_samples=10000, dataset="testing")

    train_data = train_data / 255
    test_data = test_data / 255

    print("Train data shape: {}, {}".format(train_data.shape, train_labels.shape))
    print("Test data shape: {}, {}".format(test_data.shape, test_labels.shape))

    model = Model(
        ConvLayer(filters=5, padding='same'),
        Relu(),
        # Elu(),
        PoolLayer(mode='max', kernel_shape=(2, 2), stride=2),
        Flatten(),
        FCLayer(units=10),
        Softmax(),
        name='cnn5'
    )

    model.read_weights()

    # model.set_loss(CategoricalCrossEntropy)

    # model.train(train_data, train_labels.T, epochs=2)
    # model.load_weights() # uncomment if loading previously trained weights and comment above line to skip training and only load trained weights.

    # print('Testing accuracy = {}'.format(model.evaluate(test_data, test_labels)))
