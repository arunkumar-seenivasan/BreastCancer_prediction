import pandas as pd
import tensorflow.keras as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv('breast_cancer.csv')

# Check initial data and info
print(data.head())
print(data.info())

# Drop unwanted columns
data = data.drop(['Unnamed: 32'], axis=1)

# Encode the 'diagnosis' column
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])
print(data.head())

# Divide the data into features and targets
x = data.iloc[:, 2:]  # Features
y = data.iloc[:, 1]   # Target (Diagnosis)

# Print feature and target info
print(x.head())
print(y.head())
print(y.value_counts())

# Split the data into training and testing samples
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.80, random_state=1)

# Reshape the data for LSTM (samples, timesteps, features)
# LSTM expects 3D input: [samples, timesteps, features]
xtrain = xtrain.values.reshape(xtrain.shape[0], 1, xtrain.shape[1])  # 1 timestep (for this problem)
xtest = xtest.values.reshape(xtest.shape[0], 1, xtest.shape[1])      # 1 timestep

# Create the LSTM model
model = tf.models.Sequential()

# Add LSTM layer with 64 units and 'relu' activation
model.add(tf.layers.LSTM(64, input_shape=(1, xtrain.shape[2]), activation='relu'))

# Add Dense layer for hidden layer
model.add(tf.layers.Dense(32, activation='relu'))

# Add output layer with 'sigmoid' activation for binary classification
model.add(tf.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

# Train the model and store the training history
history = model.fit(xtrain, ytrain, epochs=550, batch_size=32, validation_data=(xtest, ytest))

# Predict on the test data
ypred = model.predict(xtest)
ypred = ypred.round()  # Round to nearest integer (0 or 1)

# Print predicted values for xtest
print("Predicted values for xtest:")
print(ypred)

# Check the prediction for the first instance
if ypred[0, 0] == 0:
    print("Breast Cancer Detected for (0,0)")
else:
    print("Breast Cancer Not Detected for (0,0)")

# Confusion matrix and accuracy score
cm = confusion_matrix(ytest, ypred)
print("The confusion matrix is:")
print(cm)

accu = accuracy_score(ytest, ypred)
print("The overall accuracy is:")
print(accu)

# Plotting loss curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting accuracy curve
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Save the loss and accuracy plot
plt.tight_layout()
plt.savefig('loss_accuracy_curve.png')  # Save the plot as image
plt.show()

# Plot confusion matrix with improved color contrast and adding values inside the matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Cancer', 'Cancer'])
plt.yticks(tick_marks, ['No Cancer', 'Cancer'])

# Add values inside the confusion matrix cells
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

# Plot labels
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()

# Save confusion matrix plot
plt.savefig('confusion_matrix.png')  # Save confusion matrix as image
plt.show()
