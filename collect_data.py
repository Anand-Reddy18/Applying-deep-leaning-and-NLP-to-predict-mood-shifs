"""Deprecated legacy webcam collector.

The previous collector guessed sleep from missing eye detections and wrote
synthetic sleep/away labels into the training CSV. That is not a valid mood
measurement. Use the snapshot-based camera feature in app.py instead.
"""
def main():
    print("This legacy collector has been retired.")
    print("It inferred sleep/away states from unreliable webcam signals.")
    print("Run: streamlit run app.py")
    print("Use the self-report tab for mood history; camera results are expression signals only.")

if __name__ == "__main__":
    main()
