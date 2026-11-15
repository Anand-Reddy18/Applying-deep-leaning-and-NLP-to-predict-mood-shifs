import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical


def train_model():
    print("Loading dataset...")
    df = pd.read_csv("mood_dataset.csv")

    encoder = LabelEncoder()
    df["emotion_encoded"] = encoder.fit_transform(df["emotion"])

    # Save the encoder for inference
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    print("LabelEncoder saved to label_encoder.pkl")

    sequence_length = 5
    X = []
    Y = []
    data = df["emotion_encoded"].values

    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        Y.append(data[i+sequence_length])

    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    Y = to_categorical(Y)

    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence_length, 1)))
    model.add(Dense(Y.shape[1], activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Training model...")
    model.fit(X, Y, epochs=20, batch_size=16)

    model.save("mood_predictor.h5")
    print("Training complete")
    print("Model saved successfully")

if __name__ == "__main__":
    train_model()