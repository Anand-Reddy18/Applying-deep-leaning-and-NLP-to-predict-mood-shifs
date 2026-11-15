import numpy as np
import pandas as pd
import os
import pickle

from tensorflow.keras.models import load_model

if not os.path.exists("mood_dataset.csv"):
    print("Dataset not found. Please collect data first.")
    exit()

if not os.path.exists("label_encoder.pkl") or not os.path.exists("mood_predictor.h5"):
    print("Model or encoder not found. Please run train_model.py first.")
    exit()

df = pd.read_csv("mood_dataset.csv")

if df.empty or "timestamp" not in df.columns:
    print("Dataset is empty or incorrectly formatted.")
    exit()

# Parse timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

def get_dominant_emotion(series):
    if len(series) == 0:
        return np.nan
    modes = series.mode()
    if len(modes) > 0:
        return modes.iloc[0]
    return np.nan

# Aggregate to hourly
hourly_data = df.resample('1h').agg({'emotion': get_dominant_emotion})
hourly_data['emotion'] = hourly_data['emotion'].fillna('away')

sequence_length = 24

if len(hourly_data) < sequence_length:
    print(f"Not enough data to predict.")
    print(f"You have {len(hourly_data)} unique hours recorded, but {sequence_length} are required.")
    exit()

# Load encoder
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Filter unknown emotions
known_classes = set(encoder.classes_)
hourly_data = hourly_data[hourly_data["emotion"].isin(known_classes)]

if len(hourly_data) < sequence_length:
    print("Not enough valid data after filtering unknown emotions.")
    exit()

hourly_data["emotion_encoded"] = encoder.transform(hourly_data["emotion"])

model = load_model("mood_predictor.h5")

# Get the last 24 hours
recent_sequence = hourly_data["emotion_encoded"].values[-sequence_length:]
recent_moods_text = hourly_data["emotion"].values[-sequence_length:]

X = np.array(recent_sequence).reshape(1, sequence_length, 1)

prediction = model.predict(X)
predicted_class = np.argmax(prediction)
future_mood = encoder.inverse_transform([predicted_class])

print(f"\nYour last {sequence_length} hourly moods were:")
print(" -> ".join(recent_moods_text))

print("\n==================================")
print(f"Predicted mood for the NEXT HOUR: {future_mood[0].upper()}")
print("==================================\n")