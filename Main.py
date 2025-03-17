import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import skfuzzy as fuzz
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.linear_model import LogisticRegression
import seaborn as sns

# Initialize tkinter
main = tk.Tk()
main.title("Credit Card Fraud Detection Using Fuzzy Logic and Neural Network")
main.geometry("1200x1200")

# Global variables
global X_train, X_test, y_train, y_test
global model
global filename, dataset
global X, Y
mse = []

# Logistic Regression classifier
classifier = LogisticRegression(max_iter=1000)


# Upload dataset
def uploadDataset():
    global filename, dataset, X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', tk.END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(tk.END, str(filename) + " Dataset Loaded\n\n")
    pathlabel.config(text=str(filename) + " Dataset Loaded")

    # Load the dataset
    dataset = pd.read_csv(filename, nrows=100000)

    # Print out column names to verify
    text.insert(tk.END, "Columns in dataset:\n")
    text.insert(tk.END, str(list(dataset.columns)) + "\n\n")

    # Select features (exclude the label column)
    feature_columns = [col for col in dataset.columns if col != 'label']

    # Select features and target
    X = dataset[feature_columns]
    Y = dataset['label']

    # Preprocessing
    # Convert categorical columns if needed
    categorical_columns = ['Location']
    for col in categorical_columns:
        if col in X.columns:
            X[col] = pd.Categorical(X[col]).codes

    # Normalize numerical features
    numerical_columns = ['Time_Difference', 'Amount_Difference', 'Interval', 'Frequency']
    scaler = MinMaxScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Encode target variable
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    text.insert(tk.END, f"Dataset Shape: {X.shape}\n")
    text.insert(tk.END, f"Number of Transactions in Label: {np.sum(Y)}\n")
    text.insert(tk.END, f"Label Distribution:\n{pd.Series(Y).value_counts()}\n")
    text.insert(tk.END, str(dataset.head()))


# Run fuzzy clustering
def runFuzzy():
    global X, Y, dataset
    global X_train, X_test, y_train, y_test, mse
    text.delete('1.0', tk.END)
    mse.clear()

    if X is None or Y is None:
        messagebox.showerror("Error", "Please upload and preprocess the dataset first")
        return

    # Fuzzy clustering with scikit-fuzzy
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X.T,  # Transpose required for skfuzzy
        c=2,  # Number of clusters (fraud/non-fraud)
        m=2,  # Fuzziness parameter
        error=0.005,
        maxiter=1000,
        init=None
    )

    # Get the predicted cluster
    predict = np.argmax(u, axis=0)  # Convert fuzzy memberships to hard labels

    # Calculate performance metrics
    accuracy = accuracy_score(Y, predict)
    fuzzy_mse = 1.0 - accuracy
    mse.append(fuzzy_mse)

    text.insert(tk.END, "Fuzzy Logic Results:\n")
    text.insert(tk.END, f"MSE: {fuzzy_mse}\n")
    text.insert(tk.END, f"Accuracy: {accuracy}\n")
    text.insert(tk.END, "Confusion Matrix:\n")
    text.insert(tk.END, str(confusion_matrix(Y, predict)) + "\n")
    text.insert(tk.END, "Classification Report:\n")
    text.insert(tk.END, str(classification_report(Y, predict)))


# Run LSTM model
def runLSTM():
    global X, Y, dataset
    global X_train, X_test, y_train, y_test, mse

    if X_train is None or y_train is None:
        messagebox.showerror("Error", "Please upload and preprocess the dataset first")
        return

    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    if os.path.exists("model/lstm_model.json"):
        with open('model/lstm_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            lstm = model_from_json(loaded_model_json)
        json_file.close()
        lstm.load_weights("model/lstm_weights.weights.h5")
    else:
        lstm = Sequential()
        lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        lstm.add(Dropout(0.2))
        lstm.add(LSTM(units=50, return_sequences=True))
        lstm.add(Dropout(0.2))
        lstm.add(LSTM(units=50))
        lstm.add(Dropout(0.2))
        lstm.add(Dense(units=y_train1.shape[1], activation='softmax'))
        lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = lstm.fit(X_train1, y_train1, batch_size=16, epochs=20, shuffle=True, verbose=2,
                        validation_data=(X_test1, y_test1))
        lstm.save_weights('model/lstm_weights.weights.h5')
        model_json = lstm.to_json()
        with open("model/lstm_model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()

    # Predict with the trained LSTM model
    predict = lstm.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test1, predict)
    lstm_mse = 1.0 - accuracy
    mse.append(lstm_mse)

    text.insert(tk.END, "LSTM Results:\n")
    text.insert(tk.END, f"MSE: {lstm_mse}\n")
    text.insert(tk.END, f"Accuracy: {accuracy}\n")
    text.insert(tk.END, "Confusion Matrix:\n")
    text.insert(tk.END, str(confusion_matrix(y_test1, predict)) + "\n")
    text.insert(tk.END, "Classification Report:\n")
    text.insert(tk.END, str(classification_report(y_test1, predict)))


# Display MSE comparison graph
def mseGraph():
    global mse
    if len(mse) < 2:
        messagebox.showerror("Error", "Run both algorithms first")
        return

    height = mse
    bars = ('Fuzzy Logic MSE', 'LSTM MSE')
    y_pos = np.arange(len(bars))
    plt.figure(figsize=(8, 6))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Fuzzy Logic VS LSTM MSE Comparison Graph")
    plt.ylabel("Mean Squared Error")
    plt.show()


# Visualize Confusion Matrix
def visualizeConfusionMatrix():
    global X_test, y_test
    if X_test is None or y_test is None:
        messagebox.showerror("Error", "Please upload and preprocess the dataset first")
        return

    # Predict using Logistic Regression
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


# Close application
def close():
    main.destroy()


# GUI Layout
font = ('times', 14, 'bold')
title = tk.Label(main, text='Credit Card Fraud Detection Using Fuzzy Logic and Neural Network')
title.config(bg='DarkGoldenrod1', fg='black', font=font, height=3, width=120)
title.place(x=5, y=5)

font1 = ('times', 13, 'bold')
uploadButton = tk.Button(main, text="Upload Credit Card Fraud Dataset", command=uploadDataset, font=font1)
uploadButton.place(x=50, y=100)

pathlabel = tk.Label(main, bg='brown', fg='white', font=font1)
pathlabel.place(x=560, y=100)

fuzzyButton = tk.Button(main, text="Run Fuzzy Logic Algorithm", command=runFuzzy, font=font1)
fuzzyButton.place(x=50, y=200)

lstmButton = tk.Button(main, text="Run LSTM Algorithm", command=runLSTM, font=font1)
lstmButton.place(x=50, y=250)

msegraphButton = tk.Button(main, text="MSE Comparison Graph", command=mseGraph, font=font1)
msegraphButton.place(x=50, y=300)

confusionMatrixButton = tk.Button(main, text="Visualize Confusion Matrix", command=visualizeConfusionMatrix, font=font1)
confusionMatrixButton.place(x=50, y=350)

exitButton = tk.Button(main, text="Exit", command=close, font=font1)
exitButton.place(x=50, y=450)

text = tk.Text(main, height=25, width=100)
text.place(x=400, y=150)
text.config(font=('times', 12, 'bold'))

# Initialize global variables
X = None
Y = None
X_train = None
X_test = None
y_train = None
y_test = None

main.config(bg='LightSteelBlue1')
main.mainloop()