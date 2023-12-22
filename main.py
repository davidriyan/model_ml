# data_preprocessing.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'  # Sesuaikan dengan tipe klasifikasi Anda
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'  # Sesuaikan dengan tipe klasifikasi Anda
)


# main.py
train_dir = "model_ml/hijab_dataset_multiclass/train"
test_dir = "model_ml/hijab_dataset_multiclass/test"

# Inisialisasi generator gambar untuk pelatihan
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',  # Sesuaikan dengan tipe kelas Anda
    # Sesuaikan dengan nama kelas yang sesuai dengan nama sub-direktori
    classes=['class_1', 'class_2']
)

# Inisialisasi generator gambar untuk pengujian
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',  # Sesuaikan dengan tipe kelas Anda
    # Sesuaikan dengan nama kelas yang sesuai dengan nama sub-direktori
    classes=['class_1', 'class_2']
)

# model.py
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Multiclass classification
