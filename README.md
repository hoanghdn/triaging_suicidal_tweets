# Triaging Suicidal Tweets

## Setup Instructions

### 1. Create and Activate the Virtual Environment

On macOS/Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```sh
python -m venv venv
venv\Scripts\activate
```

### 2. Install Required Packages

After activating the virtual environment, install dependencies:
```sh
pip install -r requirements.txt
```

### 3. Run the Data Preparation Script

```sh
python prepare_dataset.py
```

---

## Required Python Packages
- pandas
- scikit-learn
- tqdm

These are listed in `requirements.txt`.
