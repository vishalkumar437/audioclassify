import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
from scipy.io import wavfile as wav
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime


def features_extractor(filename):
    audio, smaple_rate = librosa.load(filename, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=smaple_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


if __name__ == "__main__":
    filename = "4918-3-4-0.wav"

    # display wave form
    plt.figure(figsize=(14, 5))
    librosa_audio_data, librosa_sample_rate = librosa.load(filename)
    librosa.display.waveshow(librosa_audio_data, sr=librosa_sample_rate)  # mono-channel

    # load the audio file
    ipd.Audio(filename)

    print(librosa_audio_data, librosa_sample_rate)

    wav.read(filename)
    wave_sample_rate, wav_audio = wav.read(filename)  # stereo (dual-channel)
    # plt.plot(wav_audio)

    print(wav_audio, wave_sample_rate)

    # Mel Frequency Cepstral Coefficients (indentify features for classfication)
    mfccs = librosa.feature.mfcc(
        y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40
    )
    print(mfccs.shape)

    # metadata
    audio_dataset_path = "UrbanSound8K/audio/"
    metadata = pd.read_csv("UrbanSound8K/metadata/UrbanSound8K.csv")
    metadata.head(10)

    extracted_features = []
    for index_num, row in tqdm(metadata.iterrows()):
        file_name = os.path.join(
            os.path.abspath(audio_dataset_path),
            "fold" + str(row["fold"]) + "/",
            str(row["slice_file_name"]),
        )
        final_class_labels = row["class"]
        data = features_extractor(file_name)
        extracted_features.append([data, final_class_labels])

    ### converting extracted_features to Pandas dataframe
    extracted_features_df = pd.DataFrame(
        extracted_features, columns=["feature", "class"]
    )
    extracted_features_df.head()

    ### Split the dataset into independent and dependent dataset
    X = np.array(extracted_features_df["feature"].tolist())
    y = np.array(extracted_features_df["class"].tolist())

    ### Label Encoding
    ###y=np.array(pd.get_dummies(y))
    ### Label Encoder
    labelencoder = LabelEncoder()
    y = to_categorical(labelencoder.fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Model
    ### No of classes
    num_labels = y.shape[1]

    model = Sequential()
    ###first layer
    model.add(Dense(100, input_shape=(40,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    ###second layer
    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    ###third layer
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    ###final layer
    model.add(Dense(num_labels))
    model.add(Activation("softmax"))

    print(model.summary())
    model.compile(
        loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
    )

    # Model Training
    num_epochs = 100
    num_batch_size = 32

    checkpointer = ModelCheckpoint(
        filepath="saved_models/audio_classification.hdf5",
        verbose=1,
        save_best_only=True,
    )
    start = datetime.now()

    model.fit(
        X_train,
        y_train,
        batch_size=num_batch_size,
        epochs=num_epochs,
        validation_data=(X_test, y_test),
        callbacks=[checkpointer],
        verbose=1,
    )

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    # test
    filename="4918-3-4-0.wav"
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    print(mfccs_scaled_features)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    print(mfccs_scaled_features)
    print(mfccs_scaled_features.shape)
    predicted_label=np.argmax(model.predict(mfccs_scaled_features), axis=-1)
    # predicted_label=model.predict_classes(mfccs_scaled_features)
    print(predicted_label)
    prediction_class = labelencoder.inverse_transform(predicted_label) 
    print(prediction_class)