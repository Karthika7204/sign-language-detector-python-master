# Sign Language Detection Project

This project uses MediaPipe and a RandomForestClassifier to detect 19 sign language alphabets (A–S, including 'G') from hand landmarks captured via webcam.

## Prerequisites

- Python 3.8+
- Webcam
- Git (optional, for cloning the repository)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd sign-language-detector-python
```

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install opencv-python mediapipe numpy scikit-learn
```

### 3. Project Structure
Ensure the following scripts are in your project directory:
- `create_dataset.py`: Collects hand landmark data for 19 alphabets.
- `clean_dataset.py`: Cleans the dataset to remove invalid samples.
- `train_classifier.py`: Trains the RandomForestClassifier model.
- `inference_classifier.py`: Runs real-time sign language detection.

### 4. Collect Data
1. Run the data collection script to capture 500 samples per alphabet (A–S):
   ```bash
   python create_dataset.py
   ```
2. Follow on-screen instructions: Perform each alphabet sign in front of the webcam. Press 'q' to move to the next alphabet.
3. Output: `data.pickle` (~9500 samples).

### 5. Clean the Dataset
Run the cleaning script to process `data.pickle` and generate a cleaned dataset:
```bash
python clean_dataset.py
```
Output: `data_cleaned.pickle`.

### 6. Train the Model
Train the RandomForestClassifier on the cleaned dataset:
```bash
python train_classifier.py
```
Output: `model.p` (trained model file).

### 7. Run Inference
Run the inference script for real-time alphabet detection:
```bash
python inference_classifier.py
```
- Point your webcam at your hand to detect signs.
- The predicted alphabet (A–S) will be displayed on the video feed.

## Usage Notes
- Ensure good lighting and a clear view of your hand during data collection and inference.
- The model recognizes 19 alphabets: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S.
- Press 'q' to exit scripts that display a webcam feed.