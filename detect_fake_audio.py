import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
np.complex = complex
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)
def load_dataset(data_path):
    X, y = [], []
    for label, category in enumerate(["real", "fake"]):
        category_path = os.path.join(data_path, category)
        for file in os.listdir(category_path):
            if file.endswith(".wav"):
                file_path = os.path.join(category_path, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)
def predict_one(file_path, model):
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    return "FAKE" if prediction == 1 else "REAL"
def train_and_save_model(data_dir, model_path="model.pkl"):
    print("[INFO] Loading dataset...")
    X, y = load_dataset(data_dir)

    print("[INFO] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Training model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print("\n=== Evaluation on Test Set ===")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print(f"[INFO] Saving model to: {model_path}")
    joblib.dump(clf, model_path)

def load_model_and_predict(model_path):
    if not os.path.exists(model_path):
        print("[ERROR] Model not found. Train the model first.")
        return

    model = joblib.load(model_path)
    new_file = input("\nEnter the path to a .wav file to test: ").strip()
    if new_file:
        result = predict_one(new_file, model)
        print(f"\n🎧 Prediction: {result}")
    else:
        print("[INFO] No file path provided.")
if __name__ == "__main__":
    mode = input("Enter mode (train / predict): ").strip().lower()
    data_dir = "audio_data"
    model_file = "model.pkl"

    if mode == "train":
        train_and_save_model(data_dir, model_file)
    elif mode == "predict":
        load_model_and_predict(model_file)
    else:
        print("[ERROR] Invalid mode. Use 'train' or 'predict'.")
