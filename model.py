import tensorflow as tf
from tensorflow.keras import layers, models
from tkan import TKAN

def build_gait_model(input_shape=(None, 64, 64, 1), num_classes=124):
    inputs = tf.keras.Input(shape=input_shape)  # [B, T, H, W, C]

    # TimeDistributed CNNFeatureExtractor
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))(inputs)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)

    # Final CNN layers
    x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Dropout(0.3))(x)  # Output: [B, T, 128]

    # TKAN layers (using various spline configs, as per README)
    x = TKAN(100, sub_kan_configs=[
        {'spline_order': 3, 'grid_size': 10},
        {'spline_order': 1, 'grid_size': 5},
        {'spline_order': 4, 'grid_size': 6}
    ], return_sequences=True, use_bias=True)(x)

    x = TKAN(100, sub_kan_configs=[1, 2, 3, 3, 4], return_sequences=True, use_bias=True)(x)

    x = TKAN(100, sub_kan_configs=['relu'] * 5, return_sequences=True, use_bias=True)(x)

    x = TKAN(512, sub_kan_configs=[None for _ in range(3)], return_sequences=False, use_bias=True)(x)

    # Classification output
    outputs = layers.Dense(num_classes)(x)  # [B, 124]

    return models.Model(inputs, outputs)
