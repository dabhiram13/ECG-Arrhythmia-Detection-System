"""
ECG Arrhythmia Detection — ML Pipeline
Classifies cardiac arrhythmias into 5 AAMI-standard categories:
  N (Normal), S (Supraventricular), V (Ventricular), F (Fusion), Q (Unknown)

Stack: Python · PyTorch · scikit-learn · NumPy · Pandas · SciPy · Matplotlib
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import ECG1DCNN, ECGLSTM
from features import extract_features

# AAMI standard class labels
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
RESULTS_DIR = "results"

# ==========================================
# 1. Synthetic Dataset (1,000 samples)
# ==========================================
def generate_synthetic_dataset(num_samples: int = 1000, seq_length: int = 187):
    """
    Generates a 1,000-sample synthetic ECG dataset for pipeline validation.
    Each sample: 1D signal of length 187 (MIT-BIH standard beat window).
    Labels drawn with class weights matching MIT-BIH imbalance.

    Follow-up task: Replace with MIT-BIH PhysioNet loader using `wfdb`.
    """
    print(f"Generating synthetic dataset  (n={num_samples}, seq_len={seq_length}) ...")
    rng = np.random.default_rng(seed=42)

    t = np.linspace(0, 1, seq_length)
    X = np.array([
        0.5 * np.sin(2 * np.pi * (2 + rng.integers(0, 4)) * t)
        + 0.1 * rng.standard_normal(seq_length)
        for _ in range(num_samples)
    ])

    # Class distribution mirrors real MIT-BIH imbalance (N dominates ~59%)
    weights = [0.59, 0.10, 0.22, 0.07, 0.02]
    y = rng.choice(5, size=num_samples, p=weights)

    counts = {CLASS_NAMES[i]: int(np.sum(y == i)) for i in range(5)}
    print(f"Class distribution: {counts}")
    return X, y


# ==========================================
# 2. Traditional ML Pipeline
# ==========================================
def train_traditional_models(X: np.ndarray, y: np.ndarray) -> dict:
    print("\n" + "=" * 50)
    print("  Traditional ML Models (Random Forest · SVM)")
    print("=" * 50)

    print("Extracting 13 hand-engineered features per sample ...")
    X_features = np.array([list(extract_features(sig).values()) for sig in X])

    feature_names = list(extract_features(X[0]).keys())
    print(f"Features: {feature_names}\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # Random Forest
    print("Training Random Forest (n_estimators=100) ...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
    print(classification_report(y_test, rf_preds, target_names=CLASS_NAMES, labels=list(range(5)), zero_division=0))
    results["Random Forest"] = {"acc": rf_acc, "fi": rf.feature_importances_, "fi_names": feature_names}

    # SVM
    print("Training SVM (RBF kernel, C=1.0) ...")
    svm = SVC(kernel="rbf", C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)
    print(f"SVM Accuracy: {svm_acc * 100:.2f}%")
    print(classification_report(y_test, svm_preds, target_names=CLASS_NAMES, labels=list(range(5)), zero_division=0))
    results["SVM"] = {"acc": svm_acc}

    return results


# ==========================================
# 3. Deep Learning Pipeline (PyTorch)
# ==========================================
def _train_epoch(model, X_t, y_t, criterion, optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_t), y_t)
    loss.backward()
    optimizer.step()
    return loss.item()


def _evaluate(model, X_t, y_t) -> tuple:
    model.eval()
    with torch.no_grad():
        _, predicted = torch.max(model(X_t), 1)
    acc = (predicted == y_t).float().mean().item()
    return acc, predicted.cpu().numpy()


def train_deep_learning_models(X: np.ndarray, y: np.ndarray, epochs: int = 15) -> dict:
    print("\n" + "=" * 50)
    print("  Deep Learning Architectures (1D CNN · LSTM)")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    criterion = nn.CrossEntropyLoss()
    results = {}

    # ---- 1D CNN ----
    X_train_cnn = torch.FloatTensor(X_train).unsqueeze(1).to(device)
    X_test_cnn  = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    y_train_t   = torch.LongTensor(y_train).to(device)
    y_test_t    = torch.LongTensor(y_test).to(device)

    cnn = ECG1DCNN(num_classes=5).to(device)
    cnn_opt = optim.Adam(cnn.parameters(), lr=0.001)
    cnn_losses = []

    print(f"Training 1D CNN ({epochs} epochs) ...")
    for epoch in range(epochs):
        loss = _train_epoch(cnn, X_train_cnn, y_train_t, criterion, cnn_opt)
        cnn_losses.append(loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch + 1:>2}/{epochs}]  Loss: {loss:.4f}")

    cnn_acc, cnn_preds = _evaluate(cnn, X_test_cnn, y_test_t)
    print(f"\n1D CNN Accuracy: {cnn_acc * 100:.2f}%")
    print(classification_report(y_test, cnn_preds, target_names=CLASS_NAMES, labels=list(range(5)), zero_division=0))
    results["1D CNN"] = {"acc": cnn_acc, "losses": cnn_losses}

    # ---- LSTM ----
    X_train_lstm = torch.FloatTensor(X_train).unsqueeze(2).to(device)
    X_test_lstm  = torch.FloatTensor(X_test).unsqueeze(2).to(device)

    lstm = ECGLSTM(input_size=1, hidden_size=64, num_layers=2, num_classes=5).to(device)
    lstm_opt = optim.Adam(lstm.parameters(), lr=0.001)
    lstm_losses = []

    print(f"Training LSTM ({epochs} epochs) ...")
    for epoch in range(epochs):
        loss = _train_epoch(lstm, X_train_lstm, y_train_t, criterion, lstm_opt)
        lstm_losses.append(loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch + 1:>2}/{epochs}]  Loss: {loss:.4f}")

    lstm_acc, lstm_preds = _evaluate(lstm, X_test_lstm, y_test_t)
    print(f"\nLSTM Accuracy: {lstm_acc * 100:.2f}%")
    print(classification_report(y_test, lstm_preds, target_names=CLASS_NAMES, labels=list(range(5)), zero_division=0))
    results["LSTM"] = {"acc": lstm_acc, "losses": lstm_losses}

    return results


# ==========================================
# 4. Results Summary (terminal)
# ==========================================
def print_summary(trad: dict, dl: dict):
    all_acc = {k: v["acc"] for k, v in {**trad, **dl}.items()}
    df = (
        pd.DataFrame({"Model": list(all_acc.keys()),
                      "Accuracy (%)": [f"{v * 100:.2f}" for v in all_acc.values()]})
        .sort_values("Accuracy (%)", ascending=False)
        .reset_index(drop=True)
    )
    print("\n" + "=" * 50)
    print("  RESULTS SUMMARY — 1,000-sample synthetic dataset")
    print("=" * 50)
    print(df.to_string(index=False))
    print("\nDataset : 1,000-sample synthetic ECG (5 AAMI classes)")
    print("Features: 13 (4 time-domain · 5 morphological · 4 frequency-domain)")
    print("Next step: MIT-BIH Arrhythmia Database (PhysioNet / wfdb)")
    print("=" * 50)


# ==========================================
# 5. Visualization — save results.png
# ==========================================
def save_results_figure(X: np.ndarray, trad: dict, dl: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    DARK   = "#0d1117"
    PANEL  = "#161b22"
    BORDER = "#30363d"
    GREEN  = "#3fb950"
    BLUE   = "#58a6ff"
    ORANGE = "#f0883e"
    PURPLE = "#bc8cff"
    MUTED  = "#8b949e"
    WHITE  = "#e6edf3"

    MODEL_COLORS = {
        "SVM":           BLUE,
        "Random Forest": GREEN,
        "1D CNN":        ORANGE,
        "LSTM":          PURPLE,
    }

    fig = plt.figure(figsize=(16, 10), facecolor=DARK)
    fig.suptitle(
        "ECG Arrhythmia Detection — ML Pipeline Results",
        fontsize=16, fontweight="bold", color=WHITE, y=0.98
    )

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.45, wspace=0.38,
        top=0.91, bottom=0.08, left=0.07, right=0.97
    )

    # ── Panel style helper ──────────────────────────────────────
    def _panel(ax, title):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=10)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        return ax

    # ── 1. ECG Signal strip (top, spans all 3 cols) ─────────────
    ax_ecg = fig.add_subplot(gs[0, :])
    _panel(ax_ecg, "Synthetic ECG Signal — 1 Sample (187 timesteps, seq_len=187, fs=360 Hz)")
    sample = X[0]
    t = np.arange(len(sample)) / 360
    ax_ecg.plot(t, sample, color=GREEN, linewidth=1.2, alpha=0.9)
    ax_ecg.set_xlabel("Time (s)")
    ax_ecg.set_ylabel("Amplitude")
    ax_ecg.axhline(0, color=BORDER, linewidth=0.5, linestyle="--")
    # annotate class distribution
    all_acc = {k: v["acc"] for k, v in {**trad, **dl}.items()}
    label_text = "  ·  ".join([f"{k}: {v*100:.1f}%" for k, v in all_acc.items()])
    ax_ecg.text(
        0.01, 0.96, label_text,
        transform=ax_ecg.transAxes,
        fontsize=7.5, color=MUTED, va="top",
        bbox=dict(facecolor=DARK, edgecolor=BORDER, boxstyle="round,pad=0.3", alpha=0.8)
    )
    ax_ecg.grid(axis="y", color=BORDER, linewidth=0.4, alpha=0.5)

    # ── 2. Model Accuracy — horizontal bar chart (bottom-left) ──
    ax_acc = fig.add_subplot(gs[1, 0])
    _panel(ax_acc, "Model Accuracy Comparison")
    models = list(all_acc.keys())
    accs   = [all_acc[m] * 100 for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    bars = ax_acc.barh(models, accs, color=colors, edgecolor=BORDER, height=0.55)
    ax_acc.set_xlim(0, 100)
    ax_acc.set_xlabel("Accuracy (%)")
    for bar, val in zip(bars, accs):
        ax_acc.text(
            val + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left",
            fontsize=8, color=WHITE, fontweight="bold"
        )
    ax_acc.tick_params(axis="y", labelsize=9)
    ax_acc.tick_params(axis="x", labelcolor=MUTED)
    ax_acc.grid(axis="x", color=BORDER, linewidth=0.4, alpha=0.5)

    # ── 3. CNN Training Loss (bottom-middle) ───────────────────
    ax_cnn = fig.add_subplot(gs[1, 1])
    _panel(ax_cnn, "1D CNN — Training Loss")
    cnn_losses = dl["1D CNN"]["losses"]
    ax_cnn.plot(range(1, len(cnn_losses) + 1), cnn_losses,
                color=ORANGE, linewidth=2, marker="o", markersize=4)
    ax_cnn.set_xlabel("Epoch")
    ax_cnn.set_ylabel("Cross-Entropy Loss")
    ax_cnn.grid(color=BORDER, linewidth=0.4, alpha=0.5)
    ax_cnn.text(
        0.97, 0.95, f"Final: {cnn_losses[-1]:.4f}",
        transform=ax_cnn.transAxes, fontsize=8,
        color=ORANGE, ha="right", va="top",
        bbox=dict(facecolor=DARK, edgecolor=BORDER, boxstyle="round,pad=0.3", alpha=0.8)
    )

    # ── 4. LSTM Training Loss (bottom-right) ───────────────────
    ax_lstm = fig.add_subplot(gs[1, 2])
    _panel(ax_lstm, "LSTM — Training Loss")
    lstm_losses = dl["LSTM"]["losses"]
    ax_lstm.plot(range(1, len(lstm_losses) + 1), lstm_losses,
                 color=PURPLE, linewidth=2, marker="o", markersize=4)
    ax_lstm.set_xlabel("Epoch")
    ax_lstm.set_ylabel("Cross-Entropy Loss")
    ax_lstm.grid(color=BORDER, linewidth=0.4, alpha=0.5)
    ax_lstm.text(
        0.97, 0.95, f"Final: {lstm_losses[-1]:.4f}",
        transform=ax_lstm.transAxes, fontsize=8,
        color=PURPLE, ha="right", va="top",
        bbox=dict(facecolor=DARK, edgecolor=BORDER, boxstyle="round,pad=0.3", alpha=0.8)
    )

    # ── Footer ──────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        "Dataset: 1,000-sample synthetic ECG · 5 AAMI classes (N, S, V, F, Q) · "
        "13 engineered features (time-domain · morphological · frequency-domain) · "
        "Next step: MIT-BIH Arrhythmia Database (PhysioNet)",
        ha="center", fontsize=7.5, color=MUTED
    )

    out_path = os.path.join(RESULTS_DIR, "results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"\nFigure saved → {out_path}")
    return out_path


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    print("=============================================")
    print("  ECG Arrhythmia Detection — ML Pipeline")
    print("  Python · PyTorch · scikit-learn · NumPy")
    print("=============================================\n")

    #load real data
    from mitbih_loader import load_mitbih_dataset
    X, y = load_mitbih_dataset(data_dir="./mitdb", max_records=5)
    #Synthetic data
    # X, y = generate_synthetic_dataset(num_samples=1000, seq_length=187)

    trad_results = train_traditional_models(X, y)
    dl_results   = train_deep_learning_models(X, y, epochs=15)

    print_summary(trad_results, dl_results)
    save_results_figure(X, trad_results, dl_results)

    print("\nPipeline validation complete.")
    print("Ready for follow-up evaluation on MIT-BIH Arrhythmia Database.")
