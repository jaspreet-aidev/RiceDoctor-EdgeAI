# 1. Connect your Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install the Kaggle downloader
!pip install opendatasets
import opendatasets as od
import os

# 3. Create a clean, separate folder for Version 2.0 in your Drive
v2_path = '/content/drive/MyDrive/Agritech_V2_Dataset'
os.makedirs(v2_path, exist_ok=True)

print("\nDrive connected and folder created! Connecting to Kaggle...")

# 4. Download the real-world field dataset directly into your Drive
dataset_url = "https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset"
od.download(dataset_url, data_dir=v2_path)

print("\nDownload complete! Your real-world data is permanently saved.")


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Point to your newly downloaded Drive folder
base_dir = '/content/drive/MyDrive/Agritech_V2_Dataset/rice-diseases-image-dataset/RiceDiseaseDataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# 2. Load the messy, real-world images into memory
print("Loading the new field data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

# 3. The Data Audit: Let's look at the new images!
class_names = train_ds.class_names
print(f"\nDisease Classes Found: {class_names}")

for images, labels in train_ds.take(1):
    plt.figure(figsize=(12, 12))

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        img = images[i].numpy().astype("uint8")
        plt.imshow(img)

        # Decode the label name
        label_index = np.argmax(labels[i])
        plt.title(f"Label: {class_names[label_index]}", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

import shutil

# The path to the bad lab dataset we just downloaded
bad_data_path = '/content/drive/MyDrive/Agritech_V2_Dataset'

print("Deleting the sterile lab data...")
shutil.rmtree(bad_data_path, ignore_errors=True)
print("Deletion complete. Drive space cleared!")


import opendatasets as od
import os

# Create a brand new folder for the True Field Data
field_v2_path = '/content/drive/MyDrive/Agritech_True_Field_Dataset'
os.makedirs(field_v2_path, exist_ok=True)

print("Connecting to Kaggle for the Zambali Field Dataset...")

# The real-world field dataset collected by farmers
true_dataset_url = "https://www.kaggle.com/datasets/gettingintoml/zambali-rice-dataset-v3-1"
od.download(true_dataset_url, data_dir=field_v2_path)

print("\nDownload complete! The messy, real-world data is locked in.")




import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1. Point to the raw Zambali Field Dataset
base_dir = '/content/drive/MyDrive/Agritech_True_Field_Dataset/zambali-rice-dataset-v3-1/ZAMBALI_RICE_DATASET_V3'

print("Loading the Zambali Field Data and splitting it 80/20...")

# 2. The AI takes 80% for studying (Training)
train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2, # Slices off 20% for testing
    subset="training",
    seed=123, # Keeps the random shuffle consistent
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

# 3. The AI saves the remaining 20% for testing (Validation)
val_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

# 4. The Data Audit: Let's see the REAL farm data!
class_names = train_ds.class_names
print(f"\nDisease Classes Found: {class_names}")

for images, labels in train_ds.take(1):
    plt.figure(figsize=(12, 12))

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        img = images[i].numpy().astype("uint8")
        plt.imshow(img)

        label_index = np.argmax(labels[i])
        plt.title(f"Label: {class_names[label_index]}", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()





import shutil
import opendatasets as od
import os

# 1. Nuke the Hamster Dataset
print("Deleting the hamster and the Christmas tree...")
shutil.rmtree('/content/drive/MyDrive/Agritech_True_Field_Dataset', ignore_errors=True)

# 2. Download the OFFICIAL Paddy Doctor Data
paddy_doc_path = '/content/drive/MyDrive/Paddy_Doctor_Dataset'
os.makedirs(paddy_doc_path, exist_ok=True)

print("Connecting to Kaggle for the true Paddy Doctor dataset...")
# This is a clean upload of the verified agricultural competition data
od.download("https://www.kaggle.com/datasets/dasa7753912/new-paddy-doctor-paddy-disease-classification", data_dir=paddy_doc_path)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Point directly to the competition's training images
base_dir = '/content/drive/MyDrive/Paddy_Doctor_Dataset/new-paddy-doctor-paddy-disease-classification/train_images'

print("Loading the official Paddy Doctor field data...")












