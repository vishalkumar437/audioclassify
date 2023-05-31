import librosa
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("saved_models/audio_classification.hdf5")

# Load the label encoder
labelencoder = LabelEncoder()
labelencoder.classes_ = np.load("saved_models/labelencoder_classes.npy")



def classify_audio(filename):
    audio, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    predicted_label = np.argmax(model.predict(mfccs_scaled_features), axis=-1)
    prediction_class = labelencoder.inverse_transform(predicted_label)
    return prediction_class[0]

if __name__ == "__main__":
    filename = "4918-3-4-0.wav"
    predicted_class = classify_audio(filename)
    print("Predicted class:", predicted_class)
