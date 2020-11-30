from tensorflow.keras.datasets import cifar10
import pickle

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# dump data into files
pickle.dump((x_train, y_train), open("data/train_data", "wb"))
pickle.dump((x_test, y_test), open("data/test_data", "wb"))

