import os
import tensorflow as tf
from casia_loader import CASIABSequenceLoader
from model import build_gait_model
from config import *

# Load data
train_loader = CASIABSequenceLoader(
    root_dir=TRAIN_DIR,
    batch_size=BATCH_SIZE,
    sequence_len=SEQUENCE_LEN,
    image_size=(64, 64),
    shuffle=True
)

val_loader = CASIABSequenceLoader(
    root_dir=TEST_DIR,
    batch_size=BATCH_SIZE,
    sequence_len=SEQUENCE_LEN,
    image_size=(64, 64),
    shuffle=False
)

# Build model
model = build_gait_model(
    input_shape=(SEQUENCE_LEN, 64, 64, 1),
    num_classes=NUM_CLASSES
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        SAVE_MODEL_PATH, monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1
    ),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

# Train
model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=NUM_EPOCHS,
    callbacks=callbacks
)
