from flask import Flask, request, jsonify
import joblib
import numpy as np
import soundfile as sf
import librosa

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('emotion_recognition_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature extraction function
def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_scaled = np.mean(chroma.T, axis=0)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spectral_contrast_scaled = np.mean(spectral_contrast.T, axis=0)
    zero_crossings = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossings_scaled = np.mean(zero_crossings.T, axis=0)
    feature_vector = np.hstack([mfccs_scaled, chroma_scaled, spectral_contrast_scaled, zero_crossings_scaled])
    return feature_vector

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the audio file temporarily
    temp_file_path = 'temp_audio.wav'
    file.save(temp_file_path)

    # Extract features from the audio file
    features = extract_features(temp_file_path)
    features = scaler.transform(features.reshape(1, -1))  # Normalize features

    # Make prediction and get confidence
    predicted_emotion = model.predict(features)[0]
    prediction_probabilities = model.predict_proba(features)
    confidence = np.max(prediction_probabilities) * 100  # Confidence level for the predicted emotion

    # Return the predicted emotion and confidence level
    return jsonify({
        'predicted_emotion': predicted_emotion,
        'confidence': f"{confidence:.2f}%"
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Change port as needed
