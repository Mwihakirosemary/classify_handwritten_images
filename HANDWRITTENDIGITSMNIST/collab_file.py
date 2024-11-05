# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tqplA50Yacs2kNu26xAODXtdxAPhAl_a
"""

# !pip install tensorflow==2.10.0

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

from google.colab import drive
drive.mount('/content/drive')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Check the shape of the training and testing datasets
print("Training data shape:", x_train.shape)  # (60000, 28, 28)
print("Testing data shape:", x_test.shape)    # (10000, 28, 28)
print("Number of unique labels:", len(np.unique(y_train)))

# Visualize random images from the training set
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_train[np.random.randint(0, len(x_train))], cmap='gray')
    plt.axis('off')
plt.show()

# Check pixel value range and other properties
print("Image dimensions:", x_train.shape[1:])  # Should print (28, 28)
print("Pixel values range:", x_train.min(), "to", x_train.max())  # Should print 0 to 255

plt.hist(y_train, bins=10, alpha=0.75)
plt.xlabel('Digits')
plt.ylabel('Frequency')
plt.title('Distribution of Digits in Training Set')
plt.xticks(range(10))
plt.show()

mean_intensity = np.mean(x_train)
std_intensity = np.std(x_train)
print("Mean pixel intensity:", mean_intensity)
print("Standard deviation of pixel intensities:", std_intensity)

"""###Noise and Quality Check: Manually inspect a few random images."""

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[np.random.randint(0, len(x_train))], cmap='gray')
    plt.axis('off')
plt.show()

assert x_train.shape[1:] == (28, 28), "Image shape mismatch!"

"""#Step 2: Preprocessing
Objective:

Prepare the images for model training.

###Tasks
1.  Normalization Scale pixel values to a range of 0-1.




"""

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train_flat = x_train.reshape(-1, 28 * 28)
x_test_flat = x_test.reshape(-1, 28 * 28)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
# Reshape x_train to include the channel dimension
x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# Now x_train_reshaped has shape (60000, 28, 28, 1)
datagen.fit(x_train_reshaped)

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Keep 95% variance
x_train_pca = pca.fit_transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)

x_train_binary = (x_train > 0.5).astype(np.float32)
x_test_binary = (x_test > 0.5).astype(np.float32)

import cv2

x_train_denoised = np.array([cv2.GaussianBlur(img, (5, 5), 0) for img in x_train])
x_test_denoised = np.array([cv2.GaussianBlur(img, (5, 5), 0) for img in x_test])

"""###One-Hot Encoding of Labels Transform labels into one-hot encoded vectors.


"""

from tensorflow.keras.utils import to_categorical

y_train_encoded = to_categorical(y_train, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

"""#1. Traditional Models

1.1 Logistic Regression

"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

from sklearn.linear_model import LogisticRegression

# Train a Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(x_train_flat, y_train)

# Evaluate the model
y_pred_logistic = logistic_model.predict(x_test_flat)
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_logistic))

"""K-Nearest Neighbors (KNN)

"""

# Train a KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train_flat, y_train)

# Evaluate the model
y_pred_knn = knn_model.predict(x_test_flat)
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

"""Support Vector Machine (SVM)"""

# Train an SVM model
svm_model = SVC()
svm_model.fit(x_train_flat, y_train)

# Evaluate the model
y_pred_svm = svm_model.predict(x_test_flat)
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

"""#Step 3: Model Selection and Training
Objective:

Train a simple model to classify the images.

Tasks
Choose a Model You can implement different models. Here's how to implement a simple neural network and a CNN.

Simple Neural Network:
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Build the MLP model
mlp_model = Sequential()
mlp_model.add(Flatten(input_shape=(28, 28)))
mlp_model.add(Dense(128, activation='relu'))
mlp_model.add(Dense(10, activation='softmax'))

# Compile the model
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
mlp_history = mlp_model.fit(x_train, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
mlp_loss, mlp_accuracy = mlp_model.evaluate(x_test, y_test_encoded)
print("MLP Test Accuracy:", mlp_accuracy)

# Plot training and validation accuracy for MLP
plt.plot(mlp_history.history['accuracy'], label='Train Accuracy')
plt.plot(mlp_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('MLP Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Build the CNN model
cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # Convolution Layer 1
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))  # Pooling Layer 1
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))  # Convolution Layer 2
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))  # Pooling Layer 2
cnn_model.add(Flatten())  # Flatten the output
cnn_model.add(Dense(128, activation='relu'))  # Fully Connected Layer
cnn_model.add(Dense(10, activation='softmax'))  # Output layer

# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Assuming x_train is your training data and it needs to be reshaped for the CNN
# Reshape x_train to have a single channel
x_train_cnn = x_train.reshape(x_train.shape[0], 28, 28, 1)

# Reshape x_test to have a single channel
x_test_cnn = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Train the model
# Use x_train_cnn instead of x_train_cnn
cnn_history = cnn_model.fit(x_train_cnn, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
# Use x_test_cnn instead of x_test_cnn
cnn_loss, cnn_accuracy = cnn_model.evaluate(x_test_cnn, y_test_encoded)
print("CNN Test Accuracy:", cnn_accuracy)

# Plot training and validation accuracy for CNN
plt.plot(cnn_history.history['accuracy'], label='Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

"""#Step 4: Prediction and Evaluation
Objective:
Test the models on unseen data and measure their performance using various evaluation metrics:

*   Accuracy: The percentage of correctly classified instances.
*   Confusion Matrix: A matrix showing correct and incorrect classifications for each class.
*   Precision and Recall: Metrics that provide additional insights into model
*   performance, especially useful for identifying specific class-related issues.

"""

# Import necessary libraries for evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

"""##4.3. Defining a Function to Plot Confusion Matrix"""

def plot_confusion_matrix(y_true, y_pred, title):
    """
    Plots a confusion matrix using Seaborn heatmap.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - title: Title for the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix: {title}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

"""##4.4. Evaluate Traditional Models
Traditional models like Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM) do not use epochs since they are not trained using iterative gradient-based methods. Instead, they are trained using optimization algorithms that converge to a solution.
###4.4.1. Logistic Regression
"""

# logistic_model = LogisticRegression(max_iter=1000)
# logistic_model.fit(x_train_flat, y_train)

# Predictions with Logistic Regression
y_pred_logistic = logistic_model.predict(x_test_flat)

# Calculate Accuracy
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f"Logistic Regression Accuracy: {accuracy_logistic:.4f}")

# Plot Confusion Matrix
plot_confusion_matrix(y_test, y_pred_logistic, 'Logistic Regression')

# Classification Report
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_logistic))

"""###4.4.2. K-Nearest Neighbors (KNN)"""

# Predictions with KNN
y_pred_knn = knn_model.predict(x_test_flat)

# Calculate Accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn:.4f}")

# Plot Confusion Matrix
plot_confusion_matrix(y_test, y_pred_knn, 'K-Nearest Neighbors')

# Classification Report
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

"""###4.4.3. Support Vector Machine (SVM)"""

# Predictions with SVM
y_pred_svm = svm_model.predict(x_test_flat)

# Calculate Accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.4f}")

# Plot Confusion Matrix
plot_confusion_matrix(y_test, y_pred_svm, 'Support Vector Machine')

# Classification Report
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

"""##4.5. Evaluate Deep Learning Models
Deep learning models like the Multilayer Perceptron (MLP) and Convolutional Neural Network (CNN) are trained using epochs, which are complete passes through the training dataset.

###4.5.1. Multilayer Perceptron (MLP)
"""

# Predictions with MLP
y_pred_mlp = np.argmax(mlp_model.predict(x_test), axis=1)

# Calculate Accuracy
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"MLP Accuracy: {accuracy_mlp:.4f}")

# Plot Confusion Matrix
plot_confusion_matrix(y_test, y_pred_mlp, 'Multilayer Perceptron (MLP)')

# Classification Report
print("MLP Classification Report:\n", classification_report(y_test, y_pred_mlp))

"""###4.5.2. Convolutional Neural Network (CNN)"""

# Predictions with CNN
y_pred_cnn = np.argmax(cnn_model.predict(x_test_cnn), axis=1)

# Calculate Accuracy
accuracy_cnn = accuracy_score(y_test, y_pred_cnn)
print(f"CNN Accuracy: {accuracy_cnn:.4f}")

# Plot Confusion Matrix
plot_confusion_matrix(y_test, y_pred_cnn, 'Convolutional Neural Network (CNN)')

# Classification Report
print("CNN Classification Report:\n", classification_report(y_test, y_pred_cnn))

"""#4.6. Comparing Models
Interpretation:

###Traditional Models:

Logistic Regression: Generally provides good baseline performance but may lag behind more complex models.
K-Nearest Neighbors (KNN): Often performs well but can be computationally intensive with large datasets.
Support Vector Machine (SVM): Typically offers robust performance, especially with appropriate kernel choices.

###Deep Learning Models:

MLP: Shows improved performance over traditional models due to its ability to learn complex patterns.
CNN: Usually achieves the highest accuracy on image classification tasks like MNIST by effectively capturing spatial hierarchies.
"""

# Create a DataFrame to summarize accuracies
import pandas as pd

model_names = ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine', 'MLP', 'CNN']
accuracies = [accuracy_logistic, accuracy_knn, accuracy_svm, accuracy_mlp, accuracy_cnn]

results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies
})

print(results_df)

"""#4.7. Observations and Insights
###Performance Hierarchy:

CNNs tend to outperform both traditional models and simpler neural networks like MLPs on image data because they can capture spatial features and patterns effectively.

SVMs and KNNs also perform well but might require more computational resources compared to CNNs for large datasets.

Logistic Regression serves as a good baseline but may not capture complex patterns as effectively as other models.

###Confusion Matrix Analysis:

Identify which digits are commonly misclassified.
For instance, if '5' is often confused with '3', it might indicate that certain features need to be better captured or that data augmentation might help.
###Precision and Recall:

High precision and recall across all classes indicate a well-performing model.
Any low scores in specific classes should be investigated to understand underlying issues, such as insufficient training data or ambiguous handwriting styles.
##Training vs. Validation Accuracy:

Monitor for overfitting (high training accuracy but low validation accuracy) or underfitting (low accuracy on both training and validation sets).
Techniques like regularization, dropout (for neural networks), or increasing model complexity can help address these issues.

"""

import joblib
import tensorflow as tf
cnn_model.save('mnist_cnn_model.h5')
joblib.dump(mlp_model, 'mnist_mlp_model.joblib')
joblib.dump(logistic_model, 'mnist_logistic_model.joblib')
joblib.dump(knn_model, 'mnist_knn_model.joblib')
joblib.dump(knn_model, 'mnist_knn_model.joblib')
joblib.dump(svm_model, 'mnist_svm_model.joblib')

from google.colab import files
files.download('mnist_cnn_model.h5')
files.download('mnist_mlp_model.joblib')
files.download('mnist_logistic_model.joblib')
files.download('mnist_knn_model.joblib')
files.download('mnist_svm_model.joblib')