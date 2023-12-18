# main.py
from model import build_model
from data_preprocessing import prepare_dataset
from train_model import train_and_save_model
from keras.preprocessing.image import ImageDataGenerator

train_dir = "C:/Users/HP/Documents/project_ml/hijab_dataset_multiclass/train"
test_dir = "C:/Users/HP/Documents/project_ml/hijab_dataset_multiclass/test"

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
    class_mode='categorical',  # Sesuaikan dengan tipe kelas Anda
    # Sesuaikan dengan nama kelas yang sesuai dengan nama sub-direktori
    classes=['class_1', 'class_2']
)

# Inisialisasi generator gambar untuk pengujian
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Sesuaikan dengan tipe kelas Anda
    # Sesuaikan dengan nama kelas yang sesuai dengan nama sub-direktori
    classes=['class_1', 'class_2']
)
