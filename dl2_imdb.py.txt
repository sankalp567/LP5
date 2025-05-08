import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import imdb
from keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Load IMDB dataset (top 10,000 most frequent words)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Merge train and test for shuffling and re-splitting
data = np.concatenate((X_train, X_test), axis=0)
labels = np.concatenate((y_train, y_test), axis=0)

# One-hot encoding of the reviews
def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

data = vectorize(data)
labels = labels.astype("float32")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

# Basic EDA (optional but insightful)
print("Number of categories:", np.unique(labels))
print("Number of unique words:", len(np.unique(np.hstack([x for x in imdb.load_data(num_words=10000)[0][0]]))))
review_lengths = [len(x) for x in imdb.load_data(num_words=10000)[0][0]]
print("Avg. review length:", np.mean(review_lengths))

# Visualizing class distribution
sns.set(color_codes=True)
sns.countplot(x=pd.Series(labels).map(int))
plt.title("Distribution of Labels")
plt.show()

# Model definition
model = models.Sequential([
    layers.Dense(50, activation="relu", input_shape=(10000,)),
    layers.Dropout(0.3),
    layers.Dense(50, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(50, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.summary()

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,  # Increased for better learning
    batch_size=500,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Evaluate the model
predictions = (model.predict(X_test) > 0.5).astype("int32")
print("Test Accuracy:", accuracy_score(y_test, predictions))

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()