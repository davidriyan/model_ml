# train_model.py
def train_and_save_model(model, train_generator, test_generator):
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator
    )

    model.save("model.h5")
