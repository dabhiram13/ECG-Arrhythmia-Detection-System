import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from models import ECG1DCNN, ECGLSTM
from features import extract_features

# ==========================================
# 1. Dataset Handling
# ==========================================
def generate_synthetic_dataset(num_samples=1000, seq_length=187):
    """
    Generates a 1,000-sample synthetic dataset.
    Follow-up task: Replace this function with MIT-BIH PhysioNet loader.
    """
    print(f"Generating synthetic dataset (n={num_samples})...")
    X_raw = np.random.randn(num_samples, seq_length)
    # 5 AAMI Classes: N (Normal), S (Supraventricular), V (Ventricular), F (Fusion), Q (Unknown)
    y = np.random.randint(0, 5, size=(num_samples,)) 
    return X_raw, y

# ==========================================
# 2. Traditional Machine Learning Pipeline
# ==========================================
def train_traditional_models(X, y):
    print("\n--- Training Traditional ML Models ---")
    print("Extracting 13 engineered features for all samples...")
    X_features = np.array([list(extract_features(sig).values()) for sig in X])
    
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    # Random Forest
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds)*100:.2f}%")
    
    # Support Vector Machine
    print("Training SVM (RBF Kernel)...")
    svm = SVC(kernel='rbf', C=1.0)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    print(f"SVM Accuracy: {accuracy_score(y_test, svm_preds)*100:.2f}%")

# ==========================================
# 3. Deep Learning Pipeline (PyTorch)
# ==========================================
def train_deep_learning_models(X, y, epochs=5):
    print("\n--- Training Deep Learning Architectures ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Format shape for 1D CNN: (Batch, Channels, Seq_Len)
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1) 
    y_train_t = torch.LongTensor(y_train)
    
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_t = torch.LongTensor(y_test)
    
    model = ECG1DCNN(num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training 1D CNN...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")
        
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, predicted = torch.max(test_outputs.data, 1)
        acc = (predicted == y_test_t).sum().item() / y_test_t.size(0)
    print(f"1D CNN Accuracy: {acc*100:.2f}%\n")

if __name__ == "__main__":
    print("================================== ===")
    print(" ECG Arrhythmia ML Pipeline")
    print("=====================================")
    X, y = generate_synthetic_dataset()
    
    train_traditional_models(X, y)
    train_deep_learning_models(X, y)
    
    print("Pipeline validation complete! Ready for MIT-BIH evaluation.")
