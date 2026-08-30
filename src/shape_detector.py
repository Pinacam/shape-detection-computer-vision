# =======================================================
# FINAL CODE – USB WEBCAM + SKLEARN (RASPBERRY PI)
# =======================================================

import cv2
import numpy as np
import joblib
import pandas as pd
import sys

MODEL_PATH = "decision_tree_model.joblib"  # train/export a scikit-learn bundle from data/shape_database.csv (see model/ for the parallel Orange workflow)
OUTPUT_PATH = "FINAL_CORRECT.png"

MIN_AREA = 2000


# ----------------------------
# FEATURE EXTRACTION
# ----------------------------
def extract_features(contour):
    area = cv2.contourArea(contour)
    if area < MIN_AREA:
        return None

    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)

    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    compactness = area / (perimeter ** 2) if perimeter > 0 else 0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0

    features = [
        area,
        perimeter,
        x,
        y,
        w,
        h,
        circularity,
        compactness,
        convexity
    ]

    return features, (x, y, w, h)


def main():
    # ----------------------------
    # LOAD MODEL
    # ----------------------------
    print("Loading model...")
    bundle = joblib.load(MODEL_PATH)

    clf = bundle['decision_tree']
    label_encoder = bundle['label_encoder']
    feature_names = bundle['feature_names']

    print("Features:", feature_names)

    # ----------------------------
    # START USB WEBCAM
    # ----------------------------
    print("Starting USB webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: USB camera not detected.")
        sys.exit()

    # Optional resolution (safe for Raspberry Pi)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera ON — press Q to quit")

    # ----------------------------
    # LIVE LOOP
    # ----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break

        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            data = extract_features(cnt)
            if data is None:
                continue

            features, (x, y, w, h) = data

            # FIX WARNING: use DataFrame with feature names
            X_pred = pd.DataFrame([features], columns=feature_names)
            pred = clf.predict(X_pred)
            label = label_encoder.inverse_transform(pred)[0]

            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

            lines = [
                label,
                f"A:{int(features[0])}",
                f"P:{int(features[1])}",
                f"C:{features[6]:.2f}",
                f"M:{features[7]:.2f}",
                f"V:{features[8]:.2f}",
            ]

            ty = y + 15
            for line in lines:
                cv2.putText(result, line, (x + 5, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                ty += 14

        cv2.imshow("USB Webcam Detection", result)

        # PRESS Q TO EXIT
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(OUTPUT_PATH, result)
            print("Saved final image:", OUTPUT_PATH)
            break

    # ----------------------------
    # SHUTDOWN CAMERA (IMPORTANT)
    # ----------------------------
    cap.release()
    cv2.destroyAllWindows()
    print("Camera OFF — program ended cleanly")


# ----------------------------
# ENTRY POINT
# ----------------------------
# Guarded so the module can be imported (e.g. by tests, which exercise
# extract_features() directly) without opening a camera or requiring a
# trained model file to be present.
if __name__ == "__main__":
    main()
