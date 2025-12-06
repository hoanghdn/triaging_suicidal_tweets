
import os
import time
import random
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

RiskLabel = Literal["low", "moderate", "high", "crisis"]

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env file.")
client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = (
    "You are helping label tweets for suicide risk detection. "
    "Given a tweet and its known binary label (suicidal vs non-suicidal), "
    "classify it into one of four categories: low, moderate, high, or crisis. "
    "Output ONLY one word: low, moderate, high, or crisis."
)

VALID_LABELS = {"low", "moderate", "high", "crisis"}

# Aligned with the actual CSV columns and label values
INPUT_CSV = "Suicide_Ideation_Dataset(Twitter-based).csv"   # your original dataset path
TEXT_COL = "Tweet"                  # name of the column with tweet text
LABEL_COL = "Suicide"                # name of the column with label

# How the labels map to suicidal/non-suicidal (all lower, stripped)
SUICIDAL_VALUES = {"potential suicide post"}
NON_SUICIDAL_VALUES = {"not suicide post"}

# Output files
OUTPUT_DIR = "prepared_data"
TRAIN_FILE = "train.csv"
DEV_FILE = "dev.csv"
TEST_FILE = "test.csv"

# Train/dev/test proportions
TEST_SIZE = 0.15
DEV_SIZE_WITHIN_TRAIN = 0.15  # dev fraction of train+dev

# =========================
# LLM LABELING LOGIC
# =========================

RiskLabel = Literal["low", "moderate", "high", "crisis"]


def normalize_binary_label(raw_label):
    """
    Convert raw dataset label into 'suicidal' or 'non-suicidal'.
    Strips whitespace and lowercases for robust matching.
    """
    if isinstance(raw_label, str):
        label_norm = raw_label.strip().lower()
    else:
        label_norm = str(raw_label).strip().lower()

    if label_norm in SUICIDAL_VALUES:
        return "suicidal"
    elif label_norm in NON_SUICIDAL_VALUES:
        return "non-suicidal"
    else:
        raise ValueError(f"Unrecognized label value: {raw_label!r}")




def call_llm_for_risk_label(text: str) -> str:
    """
    Calls OpenAI's API to classify tweet risk into one of: 'low', 'moderate', 'high', 'crisis', based only on the tweet text.
    """
    user_prompt = f"""
Tweet: {text}

Instructions:
Classify the above tweet into one of four suicide risk categories:
- low: no indication of suicidal ideation or distress
- moderate: vague or indirect expressions of distress/sadness
- high: more direct expressions of suicidal ideation without clear imminent intent
- crisis: explicit desire or plan to harm oneself in the immediate future

Respond with exactly one of: low, moderate, high, crisis. Don't include anything else in your answer.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content
        if content is not None:
            label = content.strip().lower()
        else:
            label = ""
        if label not in VALID_LABELS:
            print(f"Warning: LLM returned invalid label '{label}'. Falling back to 'low'.")
            return "low"
        return label
    except Exception as e:
        print(f"OpenAI API error: {e}. Falling back to 'low'.")
        return "low"


def label_dataframe_with_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'binary_label_norm' column and a 'risk_label' column using an LLM.
    """
    # Normalize binary labels first
    df = df.copy()
    df["binary_label_norm"] = df[LABEL_COL].apply(normalize_binary_label)

    risk_labels = []
    print("Generating LLM-predicted risk labels (manual annotation)...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row[TEXT_COL])
        risk = call_llm_for_risk_label(text)
        risk_labels.append(risk)


    df["risk_label"] = risk_labels
    return df


def split_and_save(df: pd.DataFrame):
    """
    Stratified split into train/dev/test on risk_label.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Make sure we only keep what we need
    df_small = df[[TEXT_COL, "binary_label_norm", "risk_label"]].rename(
        columns={TEXT_COL: "text"}
    )

    # First split off test
    train_dev, test = train_test_split(
        df_small,
        test_size=TEST_SIZE,
        stratify=df_small["risk_label"],
        random_state=42,
    )

    # Then split train/dev
    dev_size = DEV_SIZE_WITHIN_TRAIN / (1.0 - TEST_SIZE)
    train, dev = train_test_split(
        train_dev,
        test_size=dev_size,
        stratify=train_dev["risk_label"],
        random_state=42,
    )

    print("📊 Split sizes:")
    print("  train:", len(train))
    print("  dev:  ", len(dev))
    print("  test: ", len(test))

    train.to_csv(os.path.join(OUTPUT_DIR, TRAIN_FILE), index=False)
    dev.to_csv(os.path.join(OUTPUT_DIR, DEV_FILE), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, TEST_FILE), index=False)

    print(f"✅ Saved train/dev/test to '{OUTPUT_DIR}/' directory.")


def main():
    print(f"📥 Loading dataset from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows.")
    print(df.head())

    # Sanity checks
    if TEXT_COL not in df.columns:
        raise KeyError(f"TEXT_COL='{TEXT_COL}' not found in columns: {df.columns}")
    if LABEL_COL not in df.columns:
        raise KeyError(f"LABEL_COL='{LABEL_COL}' not found in columns: {df.columns}")

    # Label with 4-class risk
    df_labeled = label_dataframe_with_risk(df)

    # Show distribution
    print("\n🔍 Risk label distribution:")
    print(df_labeled["risk_label"].value_counts())

    # Split + save
    split_and_save(df_labeled)

    print("Done. You now have a 4-class dataset ready for baselines.")


if __name__ == "__main__":
    main()
