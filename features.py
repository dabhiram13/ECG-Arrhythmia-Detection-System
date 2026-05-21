import numpy as np
from scipy.signal import find_peaks, welch

def extract_features(ecg_signal, fs=360):
    """
    Extracts 13 features spanning time-domain, frequency-domain, 
    and morphological characteristics from an ECG signal array.
    """
    features = {}
    
    # ---- Time-Domain Characteristics (4 features) ----
    features['mean_amp'] = np.mean(ecg_signal)
    features['std_amp'] = np.std(ecg_signal)
    features['variance'] = np.var(ecg_signal)
    features['ptp_amplitude'] = np.ptp(ecg_signal)  # Peak-to-peak amplitude
    
    # ---- Morphological Features (5 features) ----
    # Detect R-peaks assuming standard amplitude scaling
    peaks, _ = find_peaks(ecg_signal, distance=int(0.2*fs), height=np.mean(ecg_signal)) 
    
    features['num_peaks'] = len(peaks)
    if len(peaks) > 0:
        features['max_r_peak_amp'] = np.max(ecg_signal[peaks])
        features['min_r_peak_amp'] = np.min(ecg_signal[peaks])
    else:
        features['max_r_peak_amp'] = 0.0
        features['min_r_peak_amp'] = 0.0
        
    # RR Variability 
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        features['rr_mean'] = np.mean(rr_intervals)
        features['rr_std'] = np.std(rr_intervals)
    else:
        features['rr_mean'] = 0.0  
        features['rr_std'] = 0.0
        
    # ---- Frequency-Domain Analysis (4 features) ----
    # Power Spectral Density using Welch's method
    f, Pxx = welch(ecg_signal, fs=fs, nperseg=min(len(ecg_signal), 256))
    
    features['psd_mean'] = np.mean(Pxx)
    features['psd_std'] = np.std(Pxx)
    features['dominant_freq'] = f[np.argmax(Pxx)]
    
    # Spectral Entropy
    pxx_norm = Pxx / np.sum(Pxx)
    features['spectral_entropy'] = -np.sum(pxx_norm * np.log2(pxx_norm + 1e-12))
    
    return features
