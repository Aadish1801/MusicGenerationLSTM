import tensorflow.keras as keras  # type: ignore
import numpy as np
from tensorflow.keras.utils import Sequence  # type: ignore
from preprocessing import SEQUENCE_LENGTH

# Model Parameters
OUTPUT_UNITS = 96
LOSS = 'sparse_categorical_crossentropy'
LR = 0.001
epochs = 25
batch = 64  # Reduced batch size to save memory
model_path = 'model.h5'

# Data Generator Class
class DataGenerator(Sequence):
    def __init__(self, inputs_path, targets_path, batch_size):
        self.inputs = np.load(inputs_path, mmap_mode='r')  # Load in read-only mode
        self.targets = np.load(targets_path, mmap_mode='r')
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.inputs) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        return self.inputs[start:end].astype(np.float16), self.targets[start:end].astype(np.int32)  # Optimized memory usage

# LSTM Model Function
def lstm_model(output_units, loss, lr):
    inp = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(256)(inp)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(output_units, activation='softmax')(x)

    model = keras.Model(inp, out)
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
    model.summary()
    return model

# Training Function
def train(output_units=OUTPUT_UNITS, loss=LOSS, lr=LR):
    # Use the generator for training
    train_generator = DataGenerator('inputs.npy', 'targets.npy', batch_size=batch)

    # Build the model
    model = lstm_model(output_units, loss, lr)

    # Train the model using the generator
    model.fit(train_generator, epochs=epochs)

    # Save the model
    model.save(model_path)

# Run the training process
if __name__ == '__main__':
    train()
