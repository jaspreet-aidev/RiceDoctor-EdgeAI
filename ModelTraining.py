import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# 1. Define the Data Path
# Ensure this path matches exactly where your data is in Google Drive
BASE_DIR = '/content/drive/MyDrive/kisan_mitra_data/Rice_Leaf_AUG'

# 2. Extreme Data Augmentation (Crushing Memorization)
# We are creating a hostile environment so the AI is forced to learn actual disease features
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,         # Rotate heavily to simulate weird camera angles
    width_shift_range=0.2,     # Shift off-center
    height_shift_range=0.2,
    shear_range=0.2,           # Distort perspective
    zoom_range=0.3,            # Simulate taking the photo too close/far
    horizontal_flip=True,      # Farmers will hold the phone in any direction
    vertical_flip=True,
    brightness_range=[0.7, 1.3], # Simulate bright Haryana sun vs cloudy days
    validation_split=0.2       # Reserve 20% of data for the validation test
)

# Validation generator ONLY rescales, it does NOT augment (we want a pure test)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# 3. Load the Data
print("Loading Training Data...")
train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(224, 224), # MobileNetV2 standard input size
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

print("Loading Validation Data...")
validation_generator = val_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 4. Load Base Model (Frozen)
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)
base_model.trainable = False # Lock the foundational knowledge

# 5. Build the Hardened Architecture
x = base_model.output
x = GlobalAveragePooling2D()(x) # Condenses the map, drastically reducing parameters

# L2 Regularization and Dropout to break the overfitting loop
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.5)(x) 

# Final Output Layer (Matches the number of disease classes)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 6. Compile with a low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Safety Callbacks (Auto-Stop if it starts memorizing again)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

# 8. ENGAGE TRAINING
print("Initiating Heavy Compute Sprint...")
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=callbacks
)

# 9. Save the Final Asset
model.save('/content/drive/MyDrive/KisanMitra_Hardened_Model.h5')
print("Model Secured and Saved to Drive.")
