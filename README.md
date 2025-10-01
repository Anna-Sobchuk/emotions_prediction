# Emotion Recognition System

This project implements an emotion recognition system using classical machine learning methods (LBP + KNN, HOG + SVM) and a deep learning approach (Mini-Xception CNN). It processes webcam images, detects faces, and predicts one of seven emotion classes.

---

## Theory

### Local Binary Patterns (LBP)
LBP is a texture descriptor that encodes the relationship between a pixel and its neighbors. For a pixel with intensity \(I_c\) and \(P\) surrounding neighbors \(I_p\) on a circle of radius \(R\):
$$
LBP_{P,R}(x_c, y_c) = \sum_{p=0}^{P-1} s(I_p - I_c) \cdot 2^p
$$
where  
$$
s(x) =
\begin{cases} 
1 & x \geq 0 \\
0 & x < 0
\end{cases}
$$
The resulting LBP histogram is normalized and used as a feature vector for the KNN classifier.

### Histogram of Oriented Gradients (HOG)
HOG captures gradient orientation distributions in localized image regions. The image is divided into cells; for each cell, a histogram of gradient directions is computed. Histograms are then normalized over blocks of cells to increase invariance to illumination.

### Mini-Xception CNN
A compact convolutional neural network designed for facial expression recognition. It uses depthwise separable convolutions to reduce computation while maintaining performance. Input images are grayscale faces of size 48x48. Output is a probability distribution over 7 emotion classes:  
`['angry','disgust','fear','happy','sad','surprise','neutral']`.

---

## Code Overview

**Dependencies:**  
`numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `tensorflow`, `gradio`, `joblib`

**Main Modules:**  
- `tp1.py` – main script for running the emotion recognition system.  
- **Face detection:** Haar cascades from OpenCV.  
- **Feature extraction:**
  - LBP: computes a 243-dimensional histogram for the face region → KNN classifier.
  - HOG: computes gradient histograms → linear SVM.  
- **Deep learning:** Mini-Xception CNN on 48x48 grayscale face.  

**Key Functions:**  
- `detect_face(img_rgb)` – returns bounding box of the largest detected face.  
- `lbp_descriptor(gray)` – computes LBP histogram.  
- `hog_descriptor(gray)` – computes HOG feature vector.  
- `predict_lbp(face_rgb)`, `predict_hog(face_rgb)`, `predict_cnn(face_rgb)` – run predictions and measure latency.  
- `process_video(frame)` – processes a single frame: detects face, extracts features, predicts emotions, and returns annotated frame + summary.

---

## Results

| Method      | Mean Latency (ms) |
|------------|-----------------|
| LBP + KNN  | 35.41           |
| HOG + SVM  | 11.47           |
| Mini-Xception | 208.67       |

The interface runs with Gradio and displays emotion predictions with confidence and latency in real-time.

---

## Usage

1. Install dependencies:  
```
pip install numpy opencv-python scikit-image scikit-learn tensorflow gradio joblib
```
2. Run the script:
```
python tp1.py
```
