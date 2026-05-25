# ECG Arrhythmia Detection — ML Pipeline

A machine learning pipeline that classifies cardiac arrhythmias into **5 AAMI-standard categories** from ECG time-series data.

**Stack:** Python · PyTorch · scikit-learn · NumPy · Pandas · SciPy

---

## AAMI Class Definitions

| Class | Label | Description |
|-------|-------|-------------|
| N | Normal | Normal sinus rhythm |
| S | Supraventricular | Premature atrial/junctional beats |
| V | Ventricular | Premature ventricular contractions |
| F | Fusion | Fusion of normal + ventricular beats |
| Q | Unknown | Pacemaker / unclassifiable beats |

---

## Architecture

```
train.py             ← Main pipeline: data gen · training · evaluation · summary
models.py            ← PyTorch architectures (1D CNN · LSTM)
features.py          ← 13 hand-engineered features (time-domain · morphological · frequency-domain)
requirements.txt     ← Python dependencies
```

### Feature Engineering (13 features)

| Domain | Features |
|--------|----------|
| Time-domain (4) | mean_amp, std_amp, variance, ptp_amplitude |
| Morphological (5) | num_peaks, max_r_peak_amp, min_r_peak_amp, rr_mean, rr_std |
| Frequency-domain (4) | psd_mean, psd_std, dominant_freq, spectral_entropy |

### Models

| Model | Type | Library |
|-------|------|---------|
| Random Forest | Traditional ML | scikit-learn |
| SVM (RBF kernel) | Traditional ML | scikit-learn |
| 1D CNN | Deep Learning | PyTorch |
| LSTM | Deep Learning | PyTorch |

---

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python train.py
```

### Sample Output

```
=============================================
  ECG Arrhythmia Detection — ML Pipeline
  Python · PyTorch · scikit-learn · NumPy
=============================================

Generating synthetic dataset  (n=1000, seq_len=187) ...
Class distribution: {'N': 632, 'S': 90, 'V': 196, 'F': 66, 'Q': 16}

==================================================
  Traditional ML Models (Random Forest · SVM)
==================================================
...
Random Forest Accuracy: 61.00%
SVM Accuracy: 63.50%

==================================================
  Deep Learning Architectures (1D CNN · LSTM)
==================================================
...
1D CNN Accuracy: 63.50%
LSTM Accuracy:   63.50%

==================================================
  RESULTS SUMMARY — 1,000-sample synthetic dataset
==================================================
        Model  Accuracy (%)
          SVM         63.50
       1D CNN         63.50
         LSTM         63.50
Random Forest         61.00
```

> **Note on accuracy:** The synthetic dataset is class-imbalanced (N class ≈ 63% of samples), so baseline accuracy reflects class priors. Real MIT-BIH evaluation will use per-class F1 as the primary metric.

---

## Next Steps — MIT-BIH Evaluation

The current `train.py` uses a 1,000-sample synthetic generator to validate the full pipeline.
To run the planned clinically realistic evaluation:

1. Install `wfdb`:
   ```bash
   pip install wfdb
   ```

2. Download the MIT-BIH Arrhythmia Database from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/).

3. Replace `generate_synthetic_dataset()` in `train.py` with a `wfdb`-based loader:
   ```python
   import wfdb
   record = wfdb.rdrecord('mitdb/100')
   annotation = wfdb.rdann('mitdb/100', 'atr')
   ```

---

## References

- [MIT-BIH Arrhythmia Database — PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- [AAMI EC57 Standard for Arrhythmia Annotation](https://www.aami.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
