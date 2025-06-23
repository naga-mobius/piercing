import tensorflow as tf

def data_prep(x_train, x_test, y_train, y_test):
    x_train = x_train.astype(float) / 255
    x_test = x_test.astype(float) / 255
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(x_test))
    return train_dataset, test_dataset

