# data_preprocessing.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def prepare_dataset(train_dir, test_dir, image_size):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='binary'  # Sesuaikan dengan tipe klasifikasi Anda
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='binary'  # Sesuaikan dengan tipe klasifikasi Anda
    )

    return train_generator, test_generator
