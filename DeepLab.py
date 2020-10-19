# Importing Libraries
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
import pandas as pd
import keras

# DL class is a neural network like DeepLab. Input data: dataset
# Define parameters: img_size, num_class
class DL():
    # Class initialization
    def __init__(self):
        self. callbacks = [keras.callbacks.Callback()]

    # Create model
    def get_model(self, img_size, num_classes):
        inputs = keras.Input(shape=img_size + (3,))

        # Entry block
        x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

        # Define the model
        model = keras.Model(inputs, outputs)
        
        # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()
        # Complite model
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        # Print model summary
        model.summary()
        return model

    def training(self, train, valid, model, ep = 10000, st_ep = 24, val_steps = 4):
        
        # Пересчет эпох
        for i in range(ep):
            print('Epoch ', i)
            history = model.fit(train, validation_data = valid, 
                            validation_steps = val_steps, steps_per_epoch = st_ep, 
                            verbose = 2, callbacks = self.callbacks)
            print(pd.DataFrame(history.history))
            if i % 5 == 0 and i != 0:
                model.save(r"D:\NeuroNet\DeepLab\Model")
                            
