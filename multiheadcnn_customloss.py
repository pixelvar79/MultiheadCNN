##NEED TO SETUP YOUR OWN Tensorflow and Keras venv for execution

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate dummy data with separate lists for X and y
X_full = tf.random.normal(shape=(100, 32, 32, 3))
y_full = [tf.random.uniform(shape=(100,)) for _ in range(3)]  # List of target variables

from sklearn.model_selection import train_test_split
import numpy as np

# Convert to numpy
X_full_np = X_full.numpy()

# Stack the arrays vertically to create a single array of shape (3, 100)
y_full_stacked = np.vstack(y_full).T  # Transpose to get shape (100, 3)

# Convert y_full_stacked to integers
y_full_stacked = y_full_stacked.astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_full_np, y_full_stacked, test_size=0.2, random_state=42
)

# Split y_train and y_test back into three separate arrays
y1_train, y2_train, y3_train = y_train[:, 0], y_train[:, 1], y_train[:, 2]
y1_test, y2_test, y3_test = y_test[:, 0], y_test[:, 1], y_test[:, 2]

# Print the shapes to verify
print(y1_train.shape, y1_test.shape)
print(y2_train.shape, y2_test.shape)
print(y3_train.shape, y3_test.shape)
print(X_train.shape, X_test.shape)


def create_model():
  inputs = keras.Input(shape=(32, 32, 3))
  x = layers.Conv2D(32, kernel_size=3, activation="relu")(inputs)
  x = layers.MaxPooling2D(pool_size=2)(x)
  x = layers.Flatten()(x)
  outputs = []
  # Define separate heads for each target variable
  for _ in range(3):
    branch = layers.Dense(64, activation="relu")(x)
    branch = layers.Dense(1)(branch)
    outputs.append(branch)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model


# scenario 1 full dataset and loss function accessing to full y1, y2 and y3 rows
#
# Create the model
model = create_model()

print(model.summary())

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model on the full dataset
model.fit(X_train, [y1_train, y2_train, y3_train], epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
#model.evaluate(X_test, [y1_test, y2_test, y3_test])

# Evaluate the model on the test data
loss_values, *metrics_values = model.evaluate(X_test, [y1_test, y2_test, y3_test])

# Print the loss value
print("Loss:", loss_values)

# Print the metric values (e.g., mean absolute error)
for i, metric_value in enumerate(metrics_values):
    print(f"Metric {i+1}:", metric_value)


# Scenario 2: our target situation: dummy case reducing access to y3 values by with custom loss function in 'head3'

def create_model1():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    outputs = []
    # Define separate heads for each target variable
    for i in range(3):
        branch = layers.Dense(64, activation="relu", name=f'dense_{i}_out')(x)
        branch = layers.Dense(1, name=f'dense_{i}')(branch)
        outputs.append(branch)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create the model
model1 = create_model1()

# Define custom loss functions for each output
def custom_loss_0(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true, y_pred)

def custom_loss_1(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true, y_pred)

def custom_loss_2(y_true, y_pred):
    threshold = 0.8 ##example here gor 80%, but this should be iterated as 20, 40, 60, 80% 
    mask = tf.random.uniform(shape=tf.shape(y_true)) < threshold
    masked_true = tf.boolean_mask(y_true, mask)
    masked_pred = tf.boolean_mask(y_pred, mask)
    return keras.losses.mean_squared_error(masked_true, masked_pred)

custom_losses = {
    'dense_0': custom_loss_0,
    'dense_1': custom_loss_1,
    'dense_2': custom_loss_2
}

# Compile the model with custom losses for each output
model1.compile(optimizer='adam', loss=custom_losses, metrics=['mae'])

# Train the model with full y1, y2 and masked y3_train
model1.fit(X_train, {'dense_0': y1_train, 'dense_1': y2_train, 'dense_2': y3_train}, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
loss_values, *metrics_values = model1.evaluate(X_test, [y1_test, y2_test, y3_test])

# Print the loss value
print("Loss:", loss_values)

# Print the metric values (e.g., mean absolute error)
for i, metric_value in enumerate(metrics_values):
    print(f"Metric {i+1}:", metric_value)


