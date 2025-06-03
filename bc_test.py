import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow.keras as tf
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model (H5 format)
model = tf.models.load_model('breast_cancer_model.h5')

# Or load the model using pickle (if you saved as .pkl)
# with open('breast_cancer_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# Create a Tkinter window for input and classification
def classify():
    try:
        # Get the input values from the user
        features = [
            float(entry1.get()),
            float(entry2.get()),
            float(entry3.get()),
            float(entry4.get()),
            float(entry5.get()),
            float(entry6.get()),
            float(entry7.get()),
            float(entry8.get()),
            float(entry9.get()),
            float(entry10.get()),
            float(entry11.get()),
            float(entry12.get()),
            float(entry13.get()),
            float(entry14.get()),
            float(entry15.get()),
            float(entry16.get()),
            float(entry17.get()),
            float(entry18.get()),
            float(entry19.get()),
            float(entry20.get()),
            float(entry21.get()),
            float(entry22.get()),
            float(entry23.get()),
            float(entry24.get()),
            float(entry25.get()),
            float(entry26.get()),
            float(entry27.get()),
            float(entry28.get()),
            float(entry29.get()),
            float(entry30.get())
        ]

        # Reshape to match the model's expected input shape (1 timestep)
        features = np.array(features).reshape(1, 1, -1)

        # Predict using the loaded model
        prediction = model.predict(features)
        prediction = prediction.round()

        # Show the result in a messagebox
        if prediction[0, 0] == 0:
            messagebox.showinfo("Prediction", "Breast Cancer Detected")
        else:
            messagebox.showinfo("Prediction", "Breast Cancer Not Detected")
    except Exception as e:
        messagebox.showerror("Error", "Invalid input. Please enter valid values.\n" + str(e))

# Create the main window
window = tk.Tk()
window.title("Breast Cancer Classification")

# Add labels and entry widgets for each feature (assuming 30 features)
label = tk.Label(window, text="Enter the features for classification:")
label.grid(row=0, column=0, columnspan=2)

# Generate labels and entry fields for all features
entries = []
for i in range(1, 31):
    label = tk.Label(window, text=f"Feature {i}:")
    label.grid(row=i, column=0)
    entry = tk.Entry(window)
    entry.grid(row=i, column=1)
    entries.append(entry)

# Store references to each entry widget
(entry1, entry2, entry3, entry4, entry5, entry6, entry7, entry8, entry9, entry10,
 entry11, entry12, entry13, entry14, entry15, entry16, entry17, entry18, entry19,
 entry20, entry21, entry22, entry23, entry24, entry25, entry26, entry27, entry28,
 entry29, entry30) = entries

# Button to classify
classify_button = tk.Button(window, text="Classify", command=classify)
classify_button.grid(row=32, column=0, columnspan=2)

# Run the Tkinter event loop
window.mainloop()
