import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_preprocess import data_prep
from keras.optimizers import Adam

# Piercing parameters
pierce_params = {
    "epochs": 5,
    "accuracy": 0.9,
    "optimiser_class": Adam,
    "optimiser_params": {
        "learning_rate": 0.01,
        "beta_1": 0.9,
        "beta_2": 0.99
    }
}

def get_accuracy(model, x_test, y_test):
    preds = model.predict(x_test)
    preds = preds.reshape((preds.shape[0], preds.shape[1]))
    results = accuracy_score(preds, y_test)
    print(f"Test Accuracy score : {results * 100}%")
    return results

def pierce_model(model, new_x, new_y, pierce_params):
    model.update_pierce_params(pierce_params)
    epochs = pierce_params.get("epochs", 5)
    required_accuracy = pierce_params.get("accuracy", 0.9)

    x_train, x_test, y_train, y_test = train_test_split(
        new_x, new_y, test_size=0.3, random_state=42, stratify=new_y
    )
    train_dataset, _ = data_prep(x_train, x_test, y_train, y_test)

    model.fit(train_dataset, epochs=epochs)
    prev_accuracy = 0
    accuracy = get_accuracy(model, x_test, y_test)

    while accuracy < required_accuracy and accuracy > prev_accuracy:
        prev_accuracy = accuracy
        params = pierce_params["optimiser_params"]
        params["learning_rate"] = np.random.uniform(0.001, 0.01)
        params["beta_1"] = np.random.uniform(0.9, 0.99)
        params["beta_2"] = np.random.uniform(0.99, 0.9999)

        pierce_params["optimiser"] = pierce_params["optimiser_class"](**params)
        model.update_pierce_params(pierce_params)

        model.fit(train_dataset, epochs=epochs)
        accuracy = get_accuracy(model, x_test, y_test)

# Loop over pierce splits (to be executed when this file is run/imported)
from load_data import load_and_split_mnist
from model import FFNetwork

x_train, x_test, y_train, y_test, x_pierce_splits, y_pierce_splits = load_and_split_mnist()
model = FFNetwork(dims=[784, 500, 500])

for new_x, new_y in zip(x_pierce_splits, y_pierce_splits):
    pierce_model(model, new_x, new_y, pierce_params)
    # break  # optional: remove break to process all splits
