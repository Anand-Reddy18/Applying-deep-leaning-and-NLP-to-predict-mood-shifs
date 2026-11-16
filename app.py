from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from mood_engine import (
    VALID_MOODS, add_entry, next_mood_forecast, read_history,
)

st.set_page_config(page_title="Mood Journal & Signals", page_icon="🧠", layout="wide")
st.title("Mood Journal & Emotion Signals")
st.caption("A personal reflection tool — not a mental-health diagnosis or a detector of your true feelings.")

with st.sidebar:
    st.header("About this app")
    st.write(
        "Text and facial-expression models estimate emotion-related signals. "
        "Only moods you explicitly log are used for the experimental next-mood estimate."
    )
    st.warning("Camera and text predictions can be wrong. Do not use them for medical or high-stakes decisions.")

@st.cache_resource(show_spinner="Loading the pretrained text emotion model…")
def load_text_classifier():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        truncation=True,
    )

@st.cache_resource(show_spinner="Loading the facial-expression model…")
def load_face_analyzer():
    from deepface import DeepFace
    return DeepFace

tabs = st.tabs(["Log your mood", "Text signal (NLP)", "Face signal (camera/photo)", "History & forecast"])

with tabs[0]:
    st.subheader("How are you actually feeling?")
    st.write("This self-report is the only input used for your future-mood estimate.")
    with st.form("self_report_form"):
        selected_mood = st.selectbox("Choose your mood", VALID_MOODS)
        note = st.text_area("Optional note (what influenced it?)", max_chars=500)
        submitted = st.form_submit_button("Save mood")
    if submitted:
        add_entry(selected_mood, "self_report", note)
        st.success("Your self-reported mood was saved.")

with tabs[1]:
    st.subheader("Text emotion signal")
    text_input = st.text_area("Enter a sentence or short journal entry", placeholder="Today I felt...")
    if st.button("Analyze text", key="analyze_text"):
        if not text_input.strip():
            st.error("Enter some text first.")
        else:
            try:
                result = load_text_classifier()(text_input.strip())[0]
                ranked = sorted(result, key=lambda item: item["score"], reverse=True)
                st.metric("Top text signal", ranked[0]["label"].title(), f"{ranked[0]['score']:.1%} model score")
                st.dataframe(pd.DataFrame(ranked).rename(columns={"label": "Emotion", "score": "Score"}), use_container_width=True)
                st.caption("This model is trained for English text. A model score is not a probability that you truly feel that emotion.")
            except Exception as exc:
                st.error(f"Text model could not run: {exc}")
                st.info("Check your internet connection and install the packages in requirements.txt.")

with tabs[2]:
    st.subheader("Facial-expression signal")
    st.write("Take a snapshot or upload a face image. The app does not continuously stream video.")
    image_file = st.camera_input("Take a camera snapshot")
    if image_file is None:
        image_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])
    if image_file is not None and st.button("Analyze facial expression"):
        try:
            import numpy as np
            image_bytes = image_file.getvalue()
            image_array = np.array(__import__("PIL").Image.open(io.BytesIO(image_bytes)).convert("RGB"))
            analysis = load_face_analyzer().analyze(
                img_path=image_array,
                actions=["emotion"],
                enforce_detection=True,
                detector_backend="opencv",
                silent=True,
            )
            face = analysis[0] if isinstance(analysis, list) else analysis
            emotions = face.get("emotion", {})
            ranked = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
            if ranked:
                st.metric("Top facial-expression signal", ranked[0][0].title(), f"{ranked[0][1]:.1f} model score")
                st.dataframe(pd.DataFrame(ranked, columns=["Expression", "Score"]), use_container_width=True)
                st.caption("A facial expression is not proof of someone's internal mood. This result is not automatically saved.")
        except Exception as exc:
            st.error(f"Could not analyze this image: {exc}")
            st.info("Try a clear, front-facing image with one visible face.")

with tabs[3]:
    st.subheader("Your mood history")
    history = read_history()
    if history.empty:
        st.info("No entries yet. Save a self-reported mood to start your journal.")
    else:
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
        st.dataframe(history.sort_values("timestamp", ascending=False), use_container_width=True)
        self_history = history[history["source"] == "self_report"]
        if not self_history.empty:
            counts = self_history["mood"].value_counts().rename_axis("Mood").reset_index(name="Entries")
            st.bar_chart(counts.set_index("Mood"))
    st.divider()
    st.subheader("Experimental next-mood estimate")
    forecast = next_mood_forecast(history)
    if not forecast["available"]:
        st.info(forecast["reason"])
    else:
        st.metric("Most likely next logged mood pattern", forecast["predicted"].title(), f"{forecast['probability']:.1%} of observed transitions")
        st.write(f"Based on {forecast['entries_used']} self-reported entries. Last entry: **{forecast['last_mood']}**.")
        st.caption(forecast["caveat"])
        st.write("Transition distribution")
        st.bar_chart(pd.Series(forecast["probabilities"], name="Observed share"))
