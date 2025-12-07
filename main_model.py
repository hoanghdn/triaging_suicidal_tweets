"""
Custom feature extraction and main model training/evaluation.
"""

import os
import re
from typing import List, Tuple

import pandas as pd
import numpy as np
from scipy import sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---- config ----
DATA_DIR = "prepared_data"
TRAIN = os.path.join(DATA_DIR, "train.csv")
DEV   = os.path.join(DATA_DIR, "dev.csv")
TEST  = os.path.join(DATA_DIR, "test.csv")

TEXT_COL = "text"
LABEL_COL = "risk_label"

MODEL_DIR = "models"

# ========== FEATURE UTILITIES ========== #

# Reuse / extend keyword lexicons from baseline1

# Unified keyword lists for each risk category
CRISIS_KEYWORDS = [
    "kill myself", "killing myself", "kms", "kys",
    "end my life", "end it all", "take my own life",
    "commit suicide", "committing suicide",
    "suicide note", "goodbye world", "say goodbye",
    "i want to die now", "i want to die tonight",
    "i'm going to kill myself", "gonna kill myself",
    "i'm done with this life", "i can't live anymore",
    "won't be here tomorrow", "everyone will be better without me",
    "punish", "only option", "i just want to die",
    "unalive", "overdose", "od", "noose"
]

HIGH_KEYWORDS = [
    "i want to die", "wanna die", "rather die", "prefer to die",
    "rather be dead", "wish i was dead", "wish i were dead",
    "don't want to live", "don't wanna live",
    "life is pointless", "life is not worth", "no reason to live",
    "i hate my life", "sick of living", "tired of living",
    "die in my sleep", "hope i don't wake up",
    "suicidal", "suicide", "suicidality"
]

MODERATE_KEYWORDS = [
    "so depressed", "feel depressed", "i am depressed",
    "feel empty", "feel numb", "i feel nothing",
    "i'm broken", "i'm not okay", "i am not okay",
    "hate myself", "i hate myself", "disgusted with myself",
    "i'm worthless", "feel worthless", "i'm a burden",
    "self harm", "self-harm", "selfharm",
    "cutting again", "cut myself", "hurt myself",
    "can't handle this", "can't take this", "can't do this anymore",
    "mental breakdown", "having a breakdown",
    "anxious all the time", "panic attacks every day",
    "depressed", "depression", "hopeless", "worthless",
    "numb", "empty", "cutting", "selfharm"
]

HOPELESSNESS_TERMS = [
    "hopeless", "no future", "nothing matters",
    "meaningless", "pointless", "empty inside",
]

CATASTROPHIZING_TERMS = [
    "never", "always", "ruined", "forever",
    "nothing helps", "no one cares",
]

FIRST_PERSON_PRONOUNS = [
    "i", "me", "my", "mine", "myself",
]

sentiment_analyzer = SentimentIntensityAnalyzer()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def contains_any_phrase(text: str, phrases: List[str]) -> bool:
    t = text.lower()
    return any(p in t for p in phrases)


def compute_sentiment_features(text: str) -> Tuple[float, float, float]:
    vs = sentiment_analyzer.polarity_scores(text or "")
    # compound ∈ [-1, 1], pos/neg ∈ [0,1]
    return vs["compound"], vs["pos"], vs["neg"]


def compute_intensity_features(text: str) -> Tuple[int, int, int]:
    t = text or ""
    num_excl = t.count("!")
    # max repeated character run length
    max_run = 1
    curr_run = 1
    for i in range(1, len(t)):
        if t[i] == t[i - 1]:
            curr_run += 1
            if curr_run > max_run:
                max_run = curr_run
        else:
            curr_run = 1
    # uppercase words
    tokens = t.split()
    num_upper = sum(1 for tok in tokens if tok.isupper() and len(tok) > 1)
    return num_excl, max_run, num_upper


def compute_self_focus(text: str) -> Tuple[int, float]:
    toks = tokenize(text or "")
    if not toks:
        return 0, 0.0
    fp_count = sum(1 for tok in toks if tok in FIRST_PERSON_PRONOUNS)
    ratio = fp_count / len(toks)
    return fp_count, ratio


def count_lexicon_hits(text: str, lexicon: List[str]) -> int:
    t = text.lower()
    return sum(1 for term in lexicon if term in t)


def compute_custom_features_for_text(text: str) -> np.ndarray:
    """
    Build the custom feature vector for a single tweet.
    Returns a 1D numpy array of shape (D_custom,)
    """
    t = text or ""

    # 1) Sentiment
    compound, pos, neg = compute_sentiment_features(t)

    # 2) Keyword flags (unified)
    has_crisis = int(contains_any_phrase(t, CRISIS_KEYWORDS))
    has_high = int(contains_any_phrase(t, HIGH_KEYWORDS))
    has_moderate = int(contains_any_phrase(t, MODERATE_KEYWORDS))

    # 3) Intensity
    num_excl, max_run, num_upper = compute_intensity_features(t)

    # 4) Self-focus
    fp_count, fp_ratio = compute_self_focus(t)

    # 5) Hopelessness / catastrophizing
    num_hope = count_lexicon_hits(t, HOPELESSNESS_TERMS)
    num_cata = count_lexicon_hits(t, CATASTROPHIZING_TERMS)

    return np.array([
        compound, pos, neg,
        has_crisis, has_high, has_moderate,
        num_excl, max_run, num_upper,
        fp_count, fp_ratio,
        num_hope, num_cata,
    ], dtype=float)


def build_custom_feature_matrix(texts: List[str]) -> np.ndarray:
    feats = [compute_custom_features_for_text(t) for t in texts]
    return np.vstack(feats)  # shape (N, D_custom)


# ========== MAIN MODEL PIPELINE ========== #

def load_data():
    train_df = pd.read_csv(TRAIN)
    dev_df   = pd.read_csv(DEV)
    test_df  = pd.read_csv(TEST)

    print("Train size:", len(train_df))
    print("Dev size:", len(dev_df))
    print("Test size:", len(test_df))

    return train_df, dev_df, test_df


def train_main_model(train_df: pd.DataFrame, dev_df: pd.DataFrame):
    X_train_text = train_df[TEXT_COL].astype(str).tolist()
    y_train = train_df[LABEL_COL].tolist()

    X_dev_text = dev_df[TEXT_COL].astype(str).tolist()
    y_dev = dev_df[LABEL_COL].tolist()

    # TF-IDF on text (same idea as Baseline 2)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )

    print("\nFitting TF-IDF vectorizer on train...")
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_dev_tfidf   = vectorizer.transform(X_dev_text)
    print("Train TF-IDF shape:", X_train_tfidf.shape)
    print("Dev TF-IDF shape:", X_dev_tfidf.shape)

    # Custom dense features
    print("Building custom feature matrices...")
    X_train_custom = build_custom_feature_matrix(X_train_text)  # (N_train, D_custom)
    X_dev_custom   = build_custom_feature_matrix(X_dev_text)    # (N_dev, D_custom)

    # Convert custom dense → sparse and hstack with TF-IDF
    X_train_combined = sp.hstack([X_train_tfidf, sp.csr_matrix(X_train_custom)], format="csr")
    X_dev_combined   = sp.hstack([X_dev_tfidf, sp.csr_matrix(X_dev_custom)], format="csr")

    print("Combined train feature shape:", X_train_combined.shape)
    print("Combined dev feature shape:", X_dev_combined.shape)

    # Logistic Regression with class_weight to handle imbalance
    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        n_jobs=-1,
        class_weight="balanced",
    )

    print("\nTraining Logistic Regression with custom features...")
    clf.fit(X_train_combined, y_train)

    # Evaluate on dev
    print("\n===== Custom Feature Model on dev =====")
    y_dev_pred = clf.predict(X_dev_combined)
    print(classification_report(y_dev, y_dev_pred, digits=3, zero_division=0))
    macro_f1 = f1_score(y_dev, y_dev_pred, average="macro")
    print("Dev Macro F1:", macro_f1)

    labels = ["low", "moderate", "high", "crisis"]
    cm = confusion_matrix(y_dev, y_dev_pred, labels=labels)
    print("Dev confusion matrix (rows=true, cols=pred):")
    print("labels:", labels)
    print(cm)

    os.makedirs(MODEL_DIR, exist_ok=True)
    # Optionally save vectorizer & model with joblib if you want

    return vectorizer, clf


def eval_on_test(vectorizer, clf, test_df: pd.DataFrame):
    X_test_text = test_df[TEXT_COL].astype(str).tolist()
    y_test = test_df[LABEL_COL].tolist()

    X_test_tfidf = vectorizer.transform(X_test_text)
    X_test_custom = build_custom_feature_matrix(X_test_text)
    X_test_combined = sp.hstack([X_test_tfidf, sp.csr_matrix(X_test_custom)], format="csr")

    print("\n===== Custom Feature Model on test =====")
    y_pred = clf.predict(X_test_combined)
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print("Test Macro F1:", macro_f1)

    labels = ["low", "moderate", "high", "crisis"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Test confusion matrix (rows=true, cols=pred):")
    print("labels:", labels)
    print(cm)


def main():
    train_df, dev_df, test_df = load_data()
    vectorizer, clf = train_main_model(train_df, dev_df)
    eval_on_test(vectorizer, clf, test_df)


if __name__ == "__main__":
    main()
