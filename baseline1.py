"""
Baseline 1: Rule-based keyword matching for 4-class risk classification.
"""


import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# ---- config ----
DATA_DIR = "prepared_data"
TRAIN = f"{DATA_DIR}/train.csv"
DEV   = f"{DATA_DIR}/dev.csv"
TEST  = f"{DATA_DIR}/test.csv"

TEXT_COL = "text"
LABEL_COL = "risk_label"

# ---- keyword lexicons ----
CRISIS_KEYWORDS = [
    "kill myself", "killing myself", "kms", "kys",
    "end my life", "end it all", "take my own life",
    "commit suicide", "committing suicide",
    "suicide note", "goodbye world", "say goodbye",
    "i want to die now", "i want to die tonight",
    "i'm going to kill myself", "gonna kill myself",
    "i'm done with this life", "i can't live anymore",
    "won't be here tomorrow", "everyone will be better without me", 
    "unalive", "overdose", "od", "noose", "hanging myself",
]

HIGH_KEYWORDS = [
    "i want to die", "wanna die", "rather die", "prefer to die",
    "rather be dead", "wish i was dead", "wish i were dead",
    "don't want to live", "don't wanna live",
    "life is pointless", "life is not worth", "no reason to live",
    "i hate my life", "sick of living", "tired of living",
    "die in my sleep", "hope i don't wake up", "suicidal", "suicide", 
    "suicidality",
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
    "numb", "empty", "cutting",
]

# order is important: strongest evidence checked first
def rule_based_label(text: str) -> str:
    t = (text or "").lower()

    if any(phrase in t for phrase in CRISIS_KEYWORDS):
        return "crisis"
    if any(phrase in t for phrase in HIGH_KEYWORDS):
        return "high"
    if any(phrase in t for phrase in MODERATE_KEYWORDS):
        return "moderate"
    return "low"


def eval_rule_baseline(df: pd.DataFrame, split_name: str = "dev") -> None:
    y_true = df[LABEL_COL].tolist()
    y_pred = [rule_based_label(t) for t in df[TEXT_COL].tolist()]

    print(f"\n===== Rule-based baseline on {split_name} =====")
    print(classification_report(y_true, y_pred, digits=3))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print("Macro F1:", macro_f1)

    labels = ["low", "moderate", "high", "crisis"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print("labels:", labels)
    print(cm)


def main():
    train_df = pd.read_csv(TRAIN)
    dev_df   = pd.read_csv(DEV)
    test_df  = pd.read_csv(TEST)

    # we only need dev + test for evaluation; train might be useful later
    print("Train size:", len(train_df))
    print("Dev size:", len(dev_df))
    print("Test size:", len(test_df))

    eval_rule_baseline(dev_df, "dev")
    eval_rule_baseline(test_df, "test")


if __name__ == "__main__":
    main()
