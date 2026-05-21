import torch
import torch.nn as nn
import torch.nn.functional as F

class ECG1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for ECG Arrhythmia Classification.
    Designed to process 1D time-series signals directly.
    """
    def __init__(self, num_classes=5):
        super(ECG1DCNN, self).__init__()
        # Input shape: (Batch_size, 1, Sequence_length)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        
        # Assuming sequence length of 187 (MIT-BIH standard beat length zero-padded)
        # 187 -> pool -> 93 -> pool -> 46
        self.fc1 = nn.Linear(64 * 46, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten for dense layers
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ECGLSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) network to capture temporal dynamics
    of the ECG signal.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=5):
        super(ECGLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x expected shape: (batch_size, seq_length, input_size)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
