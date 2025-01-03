from flask import Flask, request, jsonify
import librosa
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.image import resize
import traceback
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = tf.keras.models.load_model("Trained_model (1).h5")

# Define the class labels
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load and preprocess audio data
def load_and_preprocess_data(file, target_shape=(210, 210)):
    data = []
    audio_data, sample_rate = librosa.load(file, sr=None)

    # Define chunk and overlap durations
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds

    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    # Iterate over each chunk
    for i in range(num_chunks):
        # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples

        # Extract the chunk of audio
        chunk = audio_data[start:end]

        # Pad the chunk if it is too short
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

        # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)

        # Resize the Mel spectrogram to the target shape
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram.numpy())

    return np.array(data)

# Model prediction function
def model_prediction(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count_index = np.argmax(counts)
    max_count_element = unique_elements[max_count_index]
    return max_count_element

@app.route('/predict', methods=['POST'])
def predict_genre():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        audio_bytes = BytesIO(file.read())


        # Preprocess the audio data
        X_test = load_and_preprocess_data(file=audio_bytes)

        # Perform prediction
        predicted_index = model_prediction(X_test)
        predicted_genre = classes[predicted_index]

        return jsonify({"genre": predicted_genre})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)
