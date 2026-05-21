# ECG Arrhythmia Machine Learning Pipeline

This folder contains the **real** Python ML backend code for the ECG Arrhythmia detection system. It includes the PyTorch deep learning models, scikit-learn traditional ML models, and the NumPy/SciPy feature engineering pipeline.

## Structure
- `models.py`: Contains the PyTorch architecture for 1D CNN and LSTM models.
- `features.py`: SciPy and NumPy implementation of the 13 time-domain, morphological, and frequency-domain features.
- `train.py`: The main execution script to train and benchmark both traditional ML and Deep Learning architectures.
- `requirements.txt`: Python package dependencies.

## Usage
To run the ML pipeline locally after exporting to GitHub:

1. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training benchmark:
   ```bash
   python train.py
   ```

## Next Steps
The `train.py` currently uses a 1,000 sample synthetic data generator to validate the PyTorch and Sci-kit Learn pipelines. To run your planned evaluation:
1. Download the MIT-BIH Arrhythmia Database from PhysioNet.
2. Update the `generate_synthetic_dataset()` function in `train.py` to load the `.dat`/`.hea` files using the `wfdb` python package.
