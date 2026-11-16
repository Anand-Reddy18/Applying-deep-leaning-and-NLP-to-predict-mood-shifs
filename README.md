# Mood Shift Prediction using Deep Learning and NLP

A prototype personal mood journal with two **emotion-signal** inputs:
- Text emotion classification using a pretrained English transformer.
- Facial-expression classification using DeepFace's pretrained models.
- A mood journal with SQLite persistence.
- An experimental next-entry pattern estimate based only on self-reported mood history.

## Important limitations

A face or a sentence cannot reliably reveal a person's internal mood. The app reports model-estimated signals, not facts about a person's mental state. It is not a medical or psychological diagnostic tool.

The next-mood estimate is a simple transition-frequency baseline, not a validated forecasting model. It requires at least 10 self-reported entries and may still be unreliable. It does not claim to predict long-term mental health.

The camera feature takes a snapshot; it is not continuous live video. No face/text inference is automatically stored in the mood journal.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The pretrained models download on first use, so the first inference needs an internet connection. DeepFace may install additional ML dependencies. If your machine cannot install its backend, the text and self-report parts still work; use the app's error message to diagnose the optional face feature.

## Evaluate the existing dataset

```bash
python train_model.py
```

This creates `evaluation.json` using a chronological holdout. It aggregates valid emotion observations into hourly bins and excludes `away` and `sleep` because they are not emotion classes. Missing hours are left missing rather than fabricated.

The old `mood_predictor.h5` and `label_encoder.pkl` are not used by the new app. The old LSTM was trained on 5-step sequences but its inference script supplied 24 steps; it also encoded emotion categories as arbitrary integers and trained on synthetic `sleep`/`away` labels. Those artifacts are retained in the repository for reference, not trusted by the new pipeline.

## Dataset format

The legacy CSV should contain:
- `timestamp`: date/time parseable by pandas
- `emotion`: one of angry, disgust, fear, happy, neutral, sad, surprise

Optional columns such as `hour` and `confidence` are ignored by the evaluator. Non-emotion states such as `away` and `sleep` are excluded.

## Privacy

Mood entries are stored locally in `mood_history.db`. Do not commit that database or private journal content to Git. Camera snapshots are processed for the request and are not saved by this app.
