import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ---------------- CONFIG ----------------
TRAIN_DIR = "train"
MODEL_PATH = "models/lbp_knn.joblib"
K = 7

# ---------------- LOAD DATA ----------------
X_images, y_labels = [], []
for label in os.listdir(TRAIN_DIR):
    label_path = os.path.join(TRAIN_DIR, label)
    if not os.path.isdir(label_path):
        continue
    for file in os.listdir(label_path):
        if file.lower().endswith((".png",".jpg",".jpeg")):
            img = cv2.imread(os.path.join(label_path,file), cv2.IMREAD_GRAYSCALE)
            X_images.append(img)
            y_labels.append(label)

X_images = np.array(X_images)
y_labels = np.array(y_labels)
print("Loaded", len(X_images), "images, labels:", np.unique(y_labels))

# ---------------- FEATURE EXTRACTION ----------------
def lbp_hist(img, P=16, R=2):
    lbp = local_binary_pattern(img, P, R, method="uniform")
    n_bins = int(P*(P-1)+3)  # always 59 for P=8
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


X = np.array([lbp_hist(img) for img in X_images])
print("Feature matrix shape:", X.shape)  # should be (num_images, 59)

# ---------------- TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("LBP+KNN test accuracy:", accuracy_score(y_test, y_pred))

# ---------------- SAVE MODEL ----------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(knn, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
