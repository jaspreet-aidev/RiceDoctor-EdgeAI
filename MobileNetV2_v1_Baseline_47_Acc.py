from google.colab import drive
drive.mount('/content/drive')

!ls /content/drive/MyDrive/

!mkdir -p ~/.kaggle
!echo "KGAT_defc9d3f9fd06b1ac542a91eaf746b3a" > ~/.kaggle/access_token
!chmod 600 ~/.kaggle/access_token

# 1. Create a fresh folder in your Drive
!mkdir -p /content/drive/MyDrive/New_Rice_Dataset

# 2. Download the dataset directly to that folder
!kaggle datasets download -d shayanriyaz/riceleafs -p /content/drive/MyDrive/New_Rice_Dataset

# 3. Unzip the downloaded file quietly (-q) so it doesn't crash the screen
!unzip -q /content/drive/MyDrive/New_Rice_Dataset/riceleafs.zip -d /content/drive/MyDrive/New_Rice_Dataset/

print("Dataset successfully downloaded and unzipped!")

!ls /content/drive/MyDrive/New_Rice_Dataset/RiceLeafs/

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import image_dataset_from_directory

# 1. Define paths to your unzipped dataset folders
train_dir = '/content/drive/MyDrive/New_Rice_Dataset/RiceLeafs/train'
val_dir = '/content/drive/MyDrive/New_Rice_Dataset/RiceLeafs/validation'

# 2. Load datasets into memory
print("Loading training data...")
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

print("Loading validation data...")
val_ds = image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

num_classes = len(train_ds.class_names)
print(f"Classes your model will learn: {train_ds.class_names}")

# 3. Load pre-trained MobileNetV2 (Transfer Learning)
print("Building model architecture...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False # Freeze base weights so they don't get ruined

# 4. Build custom classification head for your specific diseases
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x) # Helps prevent the model from memorizing the data
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# 5. Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Start the actual training!
print("Starting training phase (5 Epochs)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)


# Save the model to your Drive so you don't lose your 32 minutes of training!
model.save('/content/drive/MyDrive/New_Rice_Dataset/rice_disease_model.keras')
print("Model saved safely to Google Drive!")


import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()


import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.utils import image_dataset_from_directory

# 1. Define paths to your dataset folders
train_dir = '/content/drive/MyDrive/New_Rice_Dataset/RiceLeafs/train'
val_dir = '/content/drive/MyDrive/New_Rice_Dataset/RiceLeafs/validation'

# 2. Load datasets into memory
print("Loading training data...")
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

print("Loading validation data...")
val_ds = image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

num_classes = len(train_ds.class_names)

# 3. Load pre-trained MobileNetV2 base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze pre-trained weights

# 4. Build the model architecture with the FIXED Rescaling layer
inputs = tf.keras.Input(shape=(224, 224, 3))

# THE FIX: This scales the pixels from [0, 255] down to [-1, 1] as MobileNet expects
x = Rescaling(1./127.5, offset=-1)(inputs)

# Pass the scaled images into the base model
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# 5. Compile the corrected model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Starting training phase with proper image scaling...")
# 6. Start training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# Get the last values from the training history
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {final_train_acc * 100:.2f}%")
print(f"Final Validation Accuracy: {final_val_acc * 100:.2f}%")



import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Corrected Model Accuracy (With Rescaling)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

model.save('/content/drive/MyDrive/New_Rice_Dataset/rice_disease_model_fixed.keras')
print("Fixed model saved successfully to Google Drive!")


import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Model

# 1. Build a robust architecture with Data Augmentation
inputs = tf.keras.Input(shape=(224, 224, 3))

# STEP 1: Randomly alter images during training to fight overfitting
x = RandomFlip("horizontal_and_vertical")(inputs)
x = RandomRotation(0.2)(x)
x = RandomZoom(0.2)(x)

# STEP 2: Scale the pixels properly for MobileNetV2
x = Rescaling(1./127.5, offset=-1)(x)

# STEP 3: Pass to the pre-trained base
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)  # Increased capacity slightly
x = Dropout(0.4)(x)                   # Stronger dropout to penalize memorization
outputs = Dense(num_classes, activation='softmax')(x)

# 2. Re-compile the anti-overfitting model
model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Anti-overfitting model successfully built with Data Augmentation!")

# 3. Restart the 5-epoch training session
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)


# Save the new model with data augmentation
model.save('/content/drive/MyDrive/New_Rice_Dataset/rice_disease_model_augmented.keras')
print("Augmented model saved successfully to Google Drive!")


import matplotlib.pyplot as plt

# Plot the new accuracy curves
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy with Data Augmentation')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()


# 1. Unfreeze the base model
base_model.trainable = True

# Optional but recommended: Keep the very first/bottom layers frozen (they detect basic edges/lines)
# and only unfreeze the top layers (which detect complex shapes)
for layer in base_model.layers[:100]:
    layer.trainable = False

# 2. Re-compile the model with a VERY TINY learning rate (1e-5)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model successfully unfrozen and ready for Fine-Tuning!")

# 3. Start Fine-Tuning for 10 Epochs
# We use more epochs because the tiny learning rate means the model takes smaller steps
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Save the fine-tuned model so you don't lose your progress
model.save('/content/drive/MyDrive/New_Rice_Dataset/rice_disease_model_finetuned.keras')
print("Fine-tuned model safely saved to Google Drive!")



import matplotlib.pyplot as plt

# Plot the fine-tuning accuracy curves
plt.plot(history_fine.history['accuracy'], label='Training Accuracy')
plt.plot(history_fine.history['val_accuracy'], label='Validation Accuracy')
plt.title('Fine-Tuned Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()


from google.colab import files
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# 1. Trigger the file uploader
print("Please select a rice leaf image from your device...")
uploaded = files.upload()

for filename in uploaded.keys():
    # 2. Load and resize the image to 224x224 (exactly what the model expects)
    img = load_img(filename, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert single image into a batch

    # 3. Predict the disease
    predictions = model.predict(img_array)

    # 4. Extract the class names and confidence score
    class_names = train_ds.class_names
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]

    # Calculate confidence percentage
    confidence = predictions[0][predicted_class_idx] * 100

    # 5. Output the final result
    print("\n" + "="*40)
    print(f"DIAGNOSIS FOR: {filename}")
    print(f"Predicted Condition: {predicted_class}")
    print(f"Confidence Score: {confidence:.2f}%")
    print("="*40)
  


