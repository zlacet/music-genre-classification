# 🎧 Music Genre Classification (CNN + Audio Processing)
    This project builds a deep learning model to classify music genres from raw audio using a Convolutional Neural Network     (CNN). The model is trained on the GTZAN dataset and compared against a Logistic Regression baseline.

## 📌 Project Overview
    Music genre classification is a classic problem in audio signal processing and machine learning. This project demonstrates how to:

    - Convert raw audio into Mel spectrograms
    - Train a CNN on audio features
    - Evaluate performance using classification metrics
    - Compare deep learning vs traditional machine learning

## 🧠 Models Used
### 🔹 Convolutional Neural Network (Primary Model)
    - 3 convolutional layers with batch normalization
    - Max pooling for feature extraction
    - Global average pooling
    - Fully connected classifier with dropout

### 🔹 Logistic Regression (Baseline)
    - Trained on flattened spectrogram features
    - Provides a performance benchmark

## 📊 Results
| Model                 | Test Accuracy |
|-----------------------|--------------|
| CNN (Primary Model)   | **51.66%**   |
| Logistic Regression   | 41.06%       |

### Key Observations:
    - CNN outperforms baseline by ~10%
    - Strong performance on structured genres (e.g., classical, jazz)
    - Lower accuracy on overlapping genres (e.g., pop, rock)
    
## 📈 Visualizations
### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)
### Training Loss
![Loss Curve](outputs/loss_curve.png)
### Accuracy Curve
![Accuracy Curve](outputs/accuracy_curve.png)

## ⚙️ Tech Stack
- Python
- PyTorch
- Torchaudio
- Librosa
- Scikit-learn
- Matplotlib / Seaborn

## ▶️ How to Run
```bash
python train.py
python evaluate.py
python baseline.py
