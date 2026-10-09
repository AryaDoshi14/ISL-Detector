# ISL Detector

**Real-time Indian Sign Language (ISL) recognition using hand landmarks and classical machine learning.**

🌐 **Live demo:** [isl-detector-2wum.vercel.app](https://isl-detector-2wum.vercel.app/)

## Overview

ISL Detector recognizes Indian Sign Language hand signs (digits and alphabet letters) from images or a live camera feed. Instead of feeding raw pixels into a deep network, it uses **MediaPipe HandLandmarker** to extract 21 3D landmarks per hand, normalizes them, and classifies the resulting feature vector with one of three lightweight scikit-learn models.

Because ISL signs often use both hands, the pipeline tracks **two hands** at once.

## How It Works

```
Image / camera frame
        │
        ▼
MediaPipe HandLandmarker  →  21 landmarks (x, y, z) per hand
        │
        ▼
Normalization (scale-invariant)  →  126-feature vector
        │
        ▼
Classifier (MLP / Random Forest / KNN)  →  predicted sign + confidence
```

### Feature extraction

- Each hand contributes **63 features** (21 landmarks × x, y, z).
- The final vector has **126 features**: 63 for the left hand followed by 63 for the right hand. A hand that is not visible is zero-padded.
- Landmarks are shifted by the hand's minimum x/y and divided by the bounding-box diagonal, making features independent of hand size and position in the frame.

### Models

| Model             | Configuration                                                    | Saved as       |
| ----------------- | ---------------------------------------------------------------- | -------------- |
| **MLP**           | Hidden layers (256, 128, 64), ReLU, early stopping, max 700 iters | `model_mlp.p`  |
| **Random Forest** | 200 trees                                                        | `model_rf.p`   |
| **KNN**           | k = 5, Euclidean distance                                        | `model_knn.p`  |

`train.py` trains all three on an 80/20 stratified split, prints a comparison report with per-class metrics for the best model, and saves each model together with its label encoder.

## Project Structure

```
├── app.py               # Flask API serving predictions
├── preprocess.py        # Extracts landmarks from images -> data.pickle
├── train.py             # Trains and compares MLP, Random Forest and KNN
├── test_accuracy.py     # Accuracy evaluation script
├── camera_test.py       # Local webcam testing
├── utils.py             # Shared helpers
├── hand_landmarker.task # MediaPipe hand landmark model
├── data.pickle          # Extracted landmark dataset
├── model_mlp.p          # Trained MLP
├── model_rf.p           # Trained Random Forest
├── model_knn.p          # Trained KNN
├── web/                 # Web frontend
└── requirements.txt     # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.10 or newer (recommended)
- A webcam (for live testing)

### Installation

```bash
git clone https://github.com/AryaDoshi14/ISL-Detector.git
cd ISL-Detector

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run the API

The repository includes pre-trained models, so you can start the server right away:

```bash
python app.py
```

The server listens on port `5000` by default, or on the port given by the `PORT` environment variable. For production, use Gunicorn:

```bash
gunicorn app:app
```

### Test with your webcam

```bash
python camera_test.py
```

## API Reference

### `POST /predict`

Classifies a set of hand landmarks.

**Request body**

```json
{
  "landmarks": {
    "Left":  [/* 63 floats */],
    "Right": [/* 63 floats */]
  },
  "model_type": "RF"
}
```

- `landmarks`: normalized landmarks for each hand (63 values each). A missing or invalid hand is treated as absent (zero-filled).
- `model_type` *(optional)*: `MLP`, `RF` (default) or `KNN`. If the requested model is not loaded, the server falls back to a loaded one.

**Response**

```json
{
  "prediction": "A",
  "confidence": 97.5,
  "model": "RF"
}
```

### `GET /health`

Returns the server status, the list of loaded models and the feature sizes.

```json
{
  "status": "ok",
  "models_loaded": ["MLP", "RF", "KNN"],
  "features_per_hand": 63,
  "total_features": 126
}
```

## Training Your Own Model

1. Place your images in `data/new_data/`, with one folder per class, named like `A-samples`, `B-samples`, `10-samples`. The text before the first `-` becomes the label. Supported formats are `.jpg`, `.jpeg` and `.png`.
2. Extract landmarks. This writes `data.pickle` and prints the hand-detection rate per class:

   ```bash
   python preprocess.py
   ```

3. Train and compare the models. This writes `model_mlp.p`, `model_rf.p` and `model_knn.p`:

   ```bash
   python train.py
   ```

4. Evaluate:

   ```bash
   python test_accuracy.py
   ```

> **Note:** Features at inference time must be computed exactly as in `preprocess.py`, or predictions will be unreliable.

## Tech Stack

- **Hand tracking:** MediaPipe
- **Image processing:** OpenCV
- **Machine learning:** scikit-learn, NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Backend:** Flask, Flask-CORS, Gunicorn
- **Frontend and hosting:** `web/` directory, deployed on Vercel

## Limitations

- Recognition works on single static signs, not continuous signing or full sentences.
- Accuracy depends on lighting, background and how clearly the hands are detected.
- The classifiers only know the signs present in the training data.

## Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements

- [MediaPipe](https://ai.google.dev/edge/mediapipe) for hand landmark detection
- [scikit-learn](https://scikit-learn.org/) for the classifiers
