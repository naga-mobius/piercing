from keras.optimizers import Adam

def compile_and_train_model(model, train_dataset, epochs=20):
    model.compile(
        optimizer=Adam(learning_rate=0.03),
        loss="mse",
        jit_compile=False,
        metrics=[]
    )
    history = model.fit(train_dataset, epochs=epochs)
    return model, history
