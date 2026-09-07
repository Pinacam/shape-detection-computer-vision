"""
Trains the decision-tree classifier that src/shape_detector.py expects to find
at model/decision_tree_model.joblib.

This script exists because the live webcam pipeline (src/shape_detector.py)
needs a scikit-learn model bundle that wasn't checked into the repo -- only
the separately-trained Orange Data Mining export (model/decision_tree_model.pkcls)
was. Run this once to generate the missing file from the same training data
(data/shape_database.csv):

    pip install -r requirements.txt
    python model/train_model.py

It prints a held-out test accuracy for a sanity check, then re-fits the tree
on the full labeled dataset (599 rows) before saving, so the deployed model
uses every labeled example rather than just the training split.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/shape_database.csv"
OUTPUT_PATH = "model/decision_tree_model.joblib"

# Must match the feature order extract_features() in src/shape_detector.py
# builds at inference time: [area, perimeter, x, y, w, h, circularity,
# compactness, convexity].
FEATURE_NAMES = [
    "area",
    "perimeter",
    "bounding_x",
    "bounding_y",
    "bounding_width",
    "bounding_height",
    "circularity",
    "compactness",
    "convexity",
]


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    X = df[FEATURE_NAMES]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])

    # Held-out split purely to report a sanity-check accuracy -- the model
    # that actually gets saved is re-fit on the full dataset below.
    # Note: the dataset is imbalanced (Bolt=146 rows vs. Resistor=1 row), so
    # a stratified split isn't possible -- a plain random split is used
    # instead. With only one Resistor example total, the model has no way to
    # learn that class reliably; that's a data-collection gap, not a bug
    # here. Worth collecting more Resistor samples if this matters later.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    eval_clf = DecisionTreeClassifier(random_state=42)
    eval_clf.fit(X_train, y_train)
    y_pred = eval_clf.predict(X_test)
    print(f"Held-out test accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print(classification_report(
        y_test, y_pred,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
    ))

    # Final model: same hyperparameters, fit on all 599 labeled rows.
    final_clf = DecisionTreeClassifier(random_state=42)
    final_clf.fit(X, y)

    bundle = {
        "decision_tree": final_clf,
        "label_encoder": label_encoder,
        "feature_names": FEATURE_NAMES,
    }
    joblib.dump(bundle, OUTPUT_PATH)
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
