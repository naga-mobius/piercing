from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import numpy as np

def load_and_split_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_train_stream, y_train, y_train_stream = train_test_split(
        x_train, y_train, test_size=0.3, random_state=42, stratify=y_train
    )
    x_pierce_splits = np.array_split(x_train_stream, 10)
    y_pierce_splits = np.array_split(y_train_stream, 10)
    return x_train, x_test, y_train, y_test, x_pierce_splits, y_pierce_splits

