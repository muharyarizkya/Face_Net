# predict_one.py
import joblib
from utils_facenet import embed_from_path
import numpy as np

clf = joblib.load("facenet_svm.joblib")

def predict_image(path, unknown_threshold=0.55):
    emb = embed_from_path(path)
    if emb is None:
        return "NO_FACE", 0.0
    proba = clf.predict_proba([emb])[0]    # probabilities per kelas
    idx = int(np.argmax(proba))
    label = clf.classes_[idx]
    conf = float(proba[idx])
    # optional rejection: tandai "unknown" bila confidence rendah
    if conf < unknown_threshold:
        return "UNKNOWN", conf
    return label, conf

if __name__ == "__main__":
    # SESUAIKAN DENGAN STRUKTUR ANDA
    test_img = "data/train/arya/a1.jpg"  # GANTI: gunakan gambar yang ada
    label, conf = predict_image(test_img)
    print(f"Prediksi: {label} (conf={conf:.3f})")