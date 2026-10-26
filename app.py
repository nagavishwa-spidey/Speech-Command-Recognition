import tensorflow as tf
import numpy as np
import librosa
import json
import sys

def load_model(model_path='speech_command_model.keras'):
    return tf.keras.models.load_model(model_path)

def preprocess_wav(file_path, sr=16000, n_mels=40, duration=1.0):
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    if len(y) < sr * duration:
        y = np.pad(y, (0, int(sr * duration) - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel)
    return log_mel.T  # shape (time, n_mels)

def predict(file_path, model, labels_path='labels.json'):
    x = preprocess_wav(file_path)
    x = np.expand_dims(x, axis=0)  # batch dim
    preds = model.predict(x)
    pred_idx = np.argmax(preds, axis=1)[0]
    with open(labels_path) as f:
        labels = json.load(f)
    return labels[str(pred_idx)], float(preds[0, pred_idx])

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python app.py <model_path> <audio_wav>')
        sys.exit(1)
    model = load_model(sys.argv[1])
    label, confidence = predict(sys.argv[2], model)
    print(f'Predicted: {label} ({confidence*100:.2f}% confidence)')
