import cv2
import numpy as np
import time
from skimage import color
from skimage.feature import local_binary_pattern, hog
import joblib
from tensorflow.keras.models import load_model
import gradio as gr

# ------------------- CONFIG -------------------
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MODEL_DIR = "models"
LBP_KNN_PATH = f"{MODEL_DIR}/lbp_knn.joblib"
HOG_SVM_PATH = f"{MODEL_DIR}/hog_svm.joblib"
MINI_X_PATH = f"{MODEL_DIR}/mini_xception.h5"
EMO_LABELS = ['angry','disgust','fear','happy','sad','surprise','neutral']

face_cascade = cv2.CascadeClassifier(HAAR_PATH)

# ------------------- HELPER FUNCTIONS -------------------
def detect_face(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    return faces[0]

def crop_face(img_rgb, bbox, size=(48,48)):
    x,y,w,h = bbox
    face = img_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, size, interpolation=cv2.INTER_AREA)
    return face

def lbp_descriptor(gray):
    lbp = local_binary_pattern(gray, P=16, R=2, method='uniform')
    n_bins = int(16*(16-1)+3)  # 243
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist



def hog_descriptor(gray):
    return hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

# ------------------- LOAD MODELS -------------------
lbp_knn = joblib.load(LBP_KNN_PATH)
hog_svm = joblib.load(HOG_SVM_PATH)
mini_x = load_model(MINI_X_PATH)

# ------------------- PREDICTION FUNCTIONS -------------------
def predict_lbp(face_rgb):
    t0 = time.perf_counter()
    gray = color.rgb2gray(face_rgb)
    gray8 = (gray*255).astype('uint8')
    desc = lbp_descriptor(gray8)
    probs = lbp_knn.predict_proba([desc])[0]
    idx = np.argmax(probs)
    label = lbp_knn.classes_[idx]
    conf = float(probs[idx])
    latency = int((time.perf_counter()-t0)*1000)
    return label, conf, latency

def predict_hog(face_rgb):
    t0 = time.perf_counter()
    gray = color.rgb2gray(face_rgb)
    gray8 = (gray*255).astype('uint8')
    desc = hog_descriptor(gray8)
    if hasattr(hog_svm, "predict_proba"):
        probs = hog_svm.predict_proba([desc])[0]
    else:
        probs = softmax(hog_svm.decision_function([desc])[0])
    idx = np.argmax(probs)
    label = hog_svm.classes_[idx]
    conf = float(probs[idx])
    latency = int((time.perf_counter()-t0)*1000)
    return label, conf, latency

def predict_cnn(face_rgb):
    t0 = time.perf_counter()
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    x = gray.astype("float32") / 255.0
    x = x.reshape((1,48,48,1))
    preds = mini_x.predict(x, verbose=0)
    probs = preds[0] if preds.ndim==2 else preds.ravel()
    idx = int(np.argmax(probs))
    label = EMO_LABELS[idx]
    conf = float(probs[idx])
    latency = int((time.perf_counter()-t0)*1000)
    return label, conf, latency

# ------------------- FULL FRAME PROCESSING -------------------
def process_video_frame(frame: np.ndarray):
    """
    frame: a single frame (H, W, 3) from Gradio Video component
    Returns: annotated frame + summary
    """
    if frame is None:
        return np.ones((140,600,3), dtype=np.uint8)*255, "No frame captured"

    if frame.shape[2]==4:  # RGBA -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    bbox = detect_face(frame)
    if bbox is None:
        return frame, "No face detected"
    
    face = crop_face(frame, bbox, (48,48))
    
    # Run predictions
    lbp_label, lbp_conf, lbp_ms = predict_lbp(face)
    hog_label, hog_conf, hog_ms = predict_hog(face)
    cnn_label, cnn_conf, cnn_ms = predict_cnn(face)
    
    # Build visual canvas
    h, w = 140, 600
    canvas = np.ones((h,w,3), dtype=np.uint8)*255
    disp_face = cv2.resize(face, (120,120))
    canvas[10:10+120, 10:10+120] = cv2.cvtColor(disp_face, cv2.COLOR_RGB2BGR)
    
    start_x = 150
    col_w = (w - start_x - 20)//3
    def draw_box(cx, label, conf, ms, title):
        x0, y0 = cx, 20
        cv2.rectangle(canvas, (x0, y0), (x0+col_w-10, y0+100), (230,230,230), -1)
        cv2.putText(canvas, title, (x0+10, y0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1)
        cv2.putText(canvas, f"Label: {label}", (x0+10, y0+45), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
        cv2.putText(canvas, f"Conf: {conf:.2f}", (x0+10, y0+70), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
        cv2.putText(canvas, f"Latency: {ms} ms", (x0+10, y0+90), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    
    draw_box(start_x, lbp_label, lbp_conf, lbp_ms, "LBP + KNN")
    draw_box(start_x + col_w, hog_label, hog_conf, hog_ms, "HOG + SVM")
    draw_box(start_x + 2*col_w, cnn_label, cnn_conf, cnn_ms, "mini-Xception")
    
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    summary = (f"LBP: {lbp_label} ({lbp_conf:.2f}), {lbp_ms}ms | "
               f"HOG: {hog_label} ({hog_conf:.2f}), {hog_ms}ms | "
               f"CNN: {cnn_label} ({cnn_conf:.2f}), {cnn_ms}ms")
    
    return canvas_rgb, summary

def process_video(video):
    """
    video: NumPy array of shape (H, W, 3) for a single frame from the webcam
    Returns: annotated frame + summary
    """
    frame = video
    if frame.shape[2] == 4:  # RGBA -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    
    # Detect face
    bbox = detect_face(frame)
    if bbox is None:
        return frame, "No face detected"
    
    # Crop face for model input
    face = crop_face(frame, bbox, (48,48))
    
    # Run predictions
    lbp_label, lbp_conf, lbp_ms = predict_lbp(face)
    hog_label, hog_conf, hog_ms = predict_hog(face)
    cnn_label, cnn_conf, cnn_ms = predict_cnn(face)
    
    # Build visual canvas
    h, w = 140, 600
    canvas = np.ones((h,w,3), dtype=np.uint8)*255
    disp_face = cv2.resize(face, (120,120))
    canvas[10:10+120, 10:10+120] = cv2.cvtColor(disp_face, cv2.COLOR_RGB2BGR)
    
    start_x = 150
    col_w = (w - start_x - 20)//3
    def draw_box(cx, label, conf, ms, title):
        x0, y0 = cx, 20
        cv2.rectangle(canvas, (x0, y0), (x0+col_w-10, y0+100), (230,230,230), -1)
        cv2.putText(canvas, title, (x0+10, y0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1)
        cv2.putText(canvas, f"Label: {label}", (x0+10, y0+45), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
        cv2.putText(canvas, f"Conf: {conf:.2f}", (x0+10, y0+70), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
        cv2.putText(canvas, f"Latency: {ms} ms", (x0+10, y0+90), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    
    draw_box(start_x, lbp_label, lbp_conf, lbp_ms, "LBP + KNN")
    draw_box(start_x + col_w, hog_label, hog_conf, hog_ms, "HOG + SVM")
    draw_box(start_x + 2*col_w, cnn_label, cnn_conf, cnn_ms, "mini-Xception")
    
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    summary = (f"LBP: {lbp_label} ({lbp_conf:.2f}), {lbp_ms}ms | "
               f"HOG: {hog_label} ({hog_conf:.2f}), {hog_ms}ms | "
               f"CNN: {cnn_label} ({cnn_conf:.2f}), {cnn_ms}ms")
    
    return canvas_rgb, summary


# ------------------- GRADIO INTERFACE -------------------
iface = gr.Interface(
    fn=process_video,
    inputs=gr.Image(source="webcam", label="Webcam Frame"),
    outputs=[gr.Image(), gr.Textbox()],
    title="Compare LBP+KNN | HOG+SVM | mini-Xception",
    description="Detects emotions from your webcam feed."
)


if __name__ == "__main__":
    iface.launch(server_port=7860)
