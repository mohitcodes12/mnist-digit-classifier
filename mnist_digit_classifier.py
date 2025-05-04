import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Visualize one training image
plt.matshow(X_train[9])
plt.title(f"Label: {y_train[9]}")
plt.show()

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten the 28x28 images into 784-length vectors
X_train_flattened = X_train.reshape(-1, 28 * 28)
X_test_flattened = X_test.reshape(-1, 28 * 28)

# -------------------------------
# Model 1: Simple Neural Network
# -------------------------------
model = keras.Sequential([
    layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_flattened, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# -------------------------------
# Model 2: Improved with Hidden Layer
# -------------------------------
model = keras.Sequential([
    layers.Dense(100, input_shape=(784,), activation='relu'),
    layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test_flattened, y_test)
print(f"Improved Model Accuracy: {test_accuracy:.4f}")

# -------------------------------
# Model 3: With Flatten Layer (for 2D input)
# -------------------------------
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

# Predictions
y_pred = model.predict(X_test)
y_pred_labels = [np.argmax(i) for i in y_pred]

# Confusion Matrix
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
