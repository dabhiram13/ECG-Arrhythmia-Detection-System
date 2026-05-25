"""
ECG Arrhythmia Detection — ML Pipeline
Classifies cardiac arrhythmias into 5 AAMI-standard categories:
  N (Normal), S (Supraventricular), V (Ventricular), F (Fusion), Q (Unknown)

Stack: Python · PyTorch · scikit-learn · NumPy · Pandas · SciPy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import ECG1DCNN, ECGLSTM
from features import extract_features

# AAMI standard class labels
AAMI_LABELS = ["N (Normal)", "S (Supraventricular)", "V (Ventricular)", "F (Fusion)", "Q (Unknown)"]
CLASS_NAMES  = ["N", "S", "V", "F", "Q"]

# ==========================================
# 1. Synthetic Dataset (1,000 samples)
# ==========================================
def generate_synthetic_dataset(num_samples: int = 1000, seq_length: int = 187):
    """
    Generates a 1,000-sample synthetic ECG dataset for pipeline validation.

    Each sample is a 1-D signal of length 187 (MIT-BIH standard beat window).
    Labels are drawn uniformly across the 5 AAMI categories.

    Follow-up task: Replace with MIT-BIH PhysioNet loader using `wfdb`.
    """
    print(f"Generating synthetic dataset  (n={num_samples}, seq_len={seq_length}) ...")
    rng = np.random.default_rng(seed=42)

    # Simulate ECG-like signals: low-freq sinusoid + Gaussian noise
    t = np.linspace(0, 1, seq_length)
    X = np.array([
        0.5 * np.sin(2 * np.pi * (2 + rng.integers(0, 4)) * t)   # base rhythm
        + 0.1 * rng.standard_normal(seq_length)                    # noise
        for _ in range(num_samples)
    ])

    # Class distribution roughly matches real MIT-BIH imbalance (N dominates)
    weights = [0.59, 0.10, 0.22, 0.07, 0.02]
    y = rng.choice(5, size=num_samples, p=weights)

    counts = {CLASS_NAMES[i]: int(np.sum(y == i)) for i in range(5)}
    print(f"Class distribution: {counts}")
    return X, y


# ==========================================
# 2. Traditional ML Pipeline
# ==========================================
def train_traditional_models(X: np.ndarray, y: np.ndarray) -> dict:
    """Train Random Forest and SVM on 13 engineered features."""
    print("\n" + "="*50)
    print("  Traditional ML Models (Random Forest · SVM)")
    print("="*50)

    print("Extracting 13 hand-engineered features per sample ...")
    X_features = np.array([list(extract_features(sig).values()) for sig in X])

    # Feature names for the DataFrame summary
    feature_names = list(extract_features(X[0]).keys())
    print(f"Features: {feature_names}\n")

    # Scale features (important for SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # ---- Random Forest ----
    print("Training Random Forest (n_estimators=100) ...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc   = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
    print(classification_report(y_test, rf_preds, target_names=CLASS_NAMES, zero_division=0))
    results["Random Forest"] = rf_acc

    # ---- SVM (RBF Kernel) ----
    print("Training SVM (RBF kernel, C=1.0) ...")
    svm = SVC(kernel="rbf", C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_acc   = accuracy_score(y_test, svm_preds)
    print(f"SVM Accuracy: {svm_acc*100:.2f}%")
    print(classification_report(y_test, svm_preds, target_names=CLASS_NAMES, zero_division=0))
    results["SVM"] = svm_acc

    return results


# ==========================================
# 3. Deep Learning Pipeline (PyTorch)
# ==========================================
def _train_epoch(model, X_t, y_t, criterion, optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    outputs = model(X_t)
    loss = criterion(outputs, y_t)
    loss.backward()
    optimizer.step()
    return loss.item()


def _evaluate(model, X_t, y_t) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        outputs = model(X_t)
        _, predicted = torch.max(outputs, 1)
    acc   = (predicted == y_t).float().mean().item()
    preds = predicted.cpu().numpy()
    return acc, preds


def train_deep_learning_models(X: np.ndarray, y: np.ndarray, epochs: int = 15) -> dict:
    """Train 1D CNN and LSTM on raw ECG time-series (PyTorch)."""
    print("\n" + "="*50)
    print("  Deep Learning Architectures (1D CNN · LSTM)")
    print("="*50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    criterion = nn.CrossEntropyLoss()
    results    = {}

    # ---- 1D CNN ----
    # Input shape: (Batch, 1, SeqLen)
    X_train_cnn = torch.FloatTensor(X_train).unsqueeze(1).to(device)
    X_test_cnn  = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    y_train_t   = torch.LongTensor(y_train).to(device)
    y_test_t    = torch.LongTensor(y_test).to(device)

    cnn_model = ECG1DCNN(num_classes=5).to(device)
    cnn_opt   = optim.Adam(cnn_model.parameters(), lr=0.001)

    print(f"Training 1D CNN ({epochs} epochs) ...")
    for epoch in range(epochs):
        loss = _train_epoch(cnn_model, X_train_cnn, y_train_t, criterion, cnn_opt)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:>2}/{epochs}]  Loss: {loss:.4f}")

    cnn_acc, cnn_preds = _evaluate(cnn_model, X_test_cnn, y_test_t)
    print(f"\n1D CNN Accuracy: {cnn_acc*100:.2f}%")
    print(classification_report(y_test, cnn_preds, target_names=CLASS_NAMES, zero_division=0))
    results["1D CNN"] = cnn_acc

    # ---- LSTM ----
    # Input shape: (Batch, SeqLen, 1)
    X_train_lstm = torch.FloatTensor(X_train).unsqueeze(2).to(device)
    X_test_lstm  = torch.FloatTensor(X_test).unsqueeze(2).to(device)

    lstm_model = ECGLSTM(input_size=1, hidden_size=64, num_layers=2, num_classes=5).to(device)
    lstm_opt   = optim.Adam(lstm_model.parameters(), lr=0.001)

    print(f"Training LSTM ({epochs} epochs) ...")
    for epoch in range(epochs):
        loss = _train_epoch(lstm_model, X_train_lstm, y_train_t, criterion, lstm_opt)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:>2}/{epochs}]  Loss: {loss:.4f}")

    lstm_acc, lstm_preds = _evaluate(lstm_model, X_test_lstm, y_test_t)
    print(f"\nLSTM Accuracy: {lstm_acc*100:.2f}%")
    print(classification_report(y_test, lstm_preds, target_names=CLASS_NAMES, zero_division=0))
    results["LSTM"] = lstm_acc

    return results


# ==========================================
# 4. Results Summary
# ==========================================
def print_summary(trad_results: dict, dl_results: dict):
    all_results = {**trad_results, **dl_results}
    df = pd.DataFrame(
        {"Model": list(all_results.keys()),
         "Accuracy (%)": [f"{v*100:.2f}" for v in all_results.values()]}
    ).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

    print("\n" + "="*50)
    print("  RESULTS SUMMARY — 1,000-sample synthetic dataset")
    print("="*50)
    print(df.to_string(index=False))
    print("\nDataset: 1,000-sample synthetic ECG (5 AAMI classes)")
    print("Features: 13 (4 time-domain · 5 morphological · 4 frequency-domain)")
    print("Next step: Evaluate on MIT-BIH Arrhythmia Database (PhysioNet/wfdb)")
    print("="*50)


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    print("=============================================")
    print("  ECG Arrhythmia Detection — ML Pipeline")
    print("  Python · PyTorch · scikit-learn · NumPy")
    print("=============================================\n")

    X, y = generate_synthetic_dataset(num_samples=1000, seq_length=187)

    trad_results = train_traditional_models(X, y)
    dl_results   = train_deep_learning_models(X, y, epochs=15)

    print_summary(trad_results, dl_results)
    print("\nPipeline validation complete.")
    print("Ready for follow-up evaluation on MIT-BIH Arrhythmia Database.")
