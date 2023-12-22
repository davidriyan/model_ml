# train_model.py

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

model.save("model.h5")
