# Human Emotion Detection

A real-time facial emotion detection web application built with React, MediaPipe, and LiteRT.js.

The application uses the device camera to detect faces and classify facial expressions into eight emotion categories directly in the browser.

## Features

- Real-time facial emotion detection
- 8 emotion classes:
  - Anger
  - Contempt
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprised
- Live face detection with bounding box
- Real-time emotion confidence percentage
- Front and rear camera switching
- Responsive desktop and mobile interface
- Camera permission handling
- In-browser AI inference
- No continuous camera-frame upload to a backend server
- Lightweight client-side inference using LiteRT.js

## Tech Stack

### Frontend

- React
- Vite
- JavaScript
- CSS

### AI & Computer Vision

- LiteRT.js
- MediaPipe Tasks Vision
- MediaPipe Face Detector
- TensorFlow / Keras for model development

### Model

The emotion classification model accepts grayscale facial images of size `48 × 48` pixels and predicts one of the eight supported emotion classes.

The web application uses a fixed-batch TensorFlow Lite model optimized for browser inference.

## Project Structure

```text
human-emotion-detection/
│
├── public/
│   ├── models/
│   │   ├── emotion_model_v1_web.tflite
│   │   └── blaze_face_short_range.tflite
│   │
│   ├── litert-wasm/
│   └── mediapipe/
│
├── src/
│   ├── App.jsx
│   ├── App.css
│   └── ...
│
├── .gitignore
├── eslint.config.js
├── index.html
├── package.json
├── package-lock.json
├── vite.config.js
└── README.md
How It Works

The application follows this pipeline:

Device Camera
      ↓
MediaPipe Face Detection
      ↓
Face Region Extraction
      ↓
48 × 48 Grayscale Preprocessing
      ↓
LiteRT Emotion Model
      ↓
Emotion + Confidence
      ↓
Live UI Overlay

Face detection and emotion inference are performed locally in the browser.

Supported Emotions
Emotion	Description
Anger	Angry facial expression
Contempt	Contemptuous facial expression
Disgust	Disgusted facial expression
Fear	Fearful facial expression
Happy	Happy facial expression
Neutral	Neutral facial expression
Sad	Sad facial expression
Surprised	Surprised facial expression
Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/human-emotion-detection.git

Move into the project directory:

cd human-emotion-detection

Install dependencies:

npm install

Start the development server:

npm run dev

Open the local URL provided by Vite in your browser.

Camera Permissions

The application requires camera access for real-time emotion detection.

When prompted by the browser, allow camera access.

For deployed versions, camera access should be used through a secure HTTPS connection.

Browser Inference

The application performs AI inference directly in the browser using LiteRT.js and WebAssembly.

This design allows the application to process camera frames locally without sending every video frame to a remote server.

Model Development

The emotion model was developed using FER2013 and FER+ annotations and trained as an 8-class facial emotion classifier.

The final web version uses a fixed input shape:

[1, 48, 48, 1]

Input preprocessing:

RGB → Grayscale → Normalize to 0–1 → 48 × 48
Development

Run the project in development mode:

npm run dev

Create a production build:

npm run build

Preview the production build locally:

npm run preview
Deployment

The frontend can be deployed using platforms such as Vercel.

The required AI model files are stored inside:

public/models/

so they can be loaded by the browser at runtime.

License

This project is licensed under the MIT License.

Author

Utsadeep Kundu

Developed and created with 