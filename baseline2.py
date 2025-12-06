"""
Baseline 2: TF-IDF + Logistic Regression for 4-class risk classification.
"""

import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# ---- config ----
DATA_DIR = "prepared_data"
TRAIN = f"{DATA_DIR}/train.csv"
DEV   = f"{DATA_DIR}/dev.csv"
TEST  = f"{DATA_DIR}/test.csv"

TEXT_COL = "text"
LABEL_COL = "risk_label"

MODEL_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH      = os.path.join(MODEL_DIR, "logreg_tfidf.joblib")


def load_data():
    train_df = pd.read_csv(TRAIN)
    dev_df   = pd.read_csv(DEV)
    test_df  = pd.read_csv(TEST)

    print("Train size:", len(train_df))
    print("Dev size:", len(dev_df))
    print("Test size:", len(test_df))

    return train_df, dev_df, test_df


def train_tfidf_logreg(train_df: pd.DataFrame, dev_df: pd.DataFrame):
    X_train = train_df[TEXT_COL].astype(str).tolist()
    y_train = train_df[LABEL_COL].tolist()

    X_dev = dev_df[TEXT_COL].astype(str).tolist()
    y_dev = dev_df[LABEL_COL].tolist()

    # TF-IDF representation: unigrams + bigrams
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,             # ignore ultra-rare tokens
        max_features=20000,   # cap feature size for stability
    )

    print("\nFitting TF-IDF vectorizer on train...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_dev_vec   = vectorizer.transform(X_dev)

    print("Train TF-IDF shape:", X_train_vec.shape)
    print("Dev TF-IDF shape:", X_dev_vec.shape)

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        n_jobs=-1,
    )

    print("\nTraining Logistic Regression...")
    clf.fit(X_train_vec, y_train)

    # Evaluate on dev
    print("\n===== TF-IDF + Logistic Regression on dev =====")
    y_dev_pred = clf.predict(X_dev_vec)
    print(classification_report(y_dev, y_dev_pred, digits=3))

    macro_f1 = f1_score(y_dev, y_dev_pred, average="macro")
    print("Dev Macro F1:", macro_f1)

    labels = ["low", "moderate", "high", "crisis"]
    cm = confusion_matrix(y_dev, y_dev_pred, labels=labels)
    print("Dev confusion matrix (rows=true, cols=pred):")
    print("labels:", labels)
    print(cm)

    # Save the model + vectorizer for later use
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved model to {MODEL_PATH}")

    return vectorizer, clf


def eval_on_test(vectorizer, clf, test_df: pd.DataFrame):
    X_test = test_df[TEXT_COL].astype(str).tolist()
    y_test = test_df[LABEL_COL].tolist()

    X_test_vec = vectorizer.transform(X_test)

    print("\n===== TF-IDF + Logistic Regression on test =====")
    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_pred, digits=3))

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print("Test Macro F1:", macro_f1)

    labels = ["low", "moderate", "high", "crisis"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Test confusion matrix (rows=true, cols=pred):")
    print("labels:", labels)
    print(cm)


def main():
    train_df, dev_df, test_df = load_data()
    vectorizer, clf = train_tfidf_logreg(train_df, dev_df)
    eval_on_test(vectorizer, clf, test_df)


if __name__ == "__main__":
    main()
