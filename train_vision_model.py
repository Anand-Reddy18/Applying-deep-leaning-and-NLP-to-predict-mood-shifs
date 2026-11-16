"""Deprecated: do not train a new facial emotion CNN from scratch.

The rebuilt app uses a pretrained DeepFace model. Facial-expression labels
are uncertain signals and should not be interpreted as a person's true mood.
"""
def main():
    print("Custom CNN training has been retired.")
    print("The app now uses pretrained facial-expression analysis.")
    print("Run: streamlit run app.py")

if __name__ == "__main__":
    main()
