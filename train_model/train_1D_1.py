import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import welch
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter1d
import json


# --- Loss Function ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, targets):
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        return (self.alpha * (1 - pt) ** self.gamma * BCE).mean()

# --- Data Loading ---
def load_h5_data(file_path, max_samples=24000):
    with h5py.File(file_path, 'r') as f:
        waveforms = f['waveforms'][:max_samples]
        raw_labels = [s.decode('utf-8') for s in f['labels'][:max_samples]]
        labels = [eval(l) for l in raw_labels]
    return waveforms, labels

# --- Power Spectrum ---
def to_power_spectrum(signal):
    _, Pxx = welch(signal, fs=1e8, nperseg=1024, return_onesided=False)
    return Pxx

# --- Labels to Masks ---
def label_to_mask(intervals, f_grid):
    mask = np.zeros_like(f_grid, dtype=bool)
    for start, end in intervals:
        mask |= (f_grid >= start) & (f_grid <= end)
    return mask.astype(np.float32)

# --- Mask to Intervals ---
def mask_to_intervals(mask, f_grid, threshold=0.5):
    intervals = []
    active = False
    for i in range(len(mask)):
        if mask[i] > threshold and not active:
            start = f_grid[i]
            active = True
        elif mask[i] <= threshold and active:
            end = f_grid[i]
            intervals.append([round(start, 1), round(end, 1)])
            active = False
    if active:
        intervals.append([round(start, 1), round(f_grid[-1], 1)])
    return intervals

# --- Filter short intervals ---
def filter_short_intervals(intervals, min_width=0.5):
    return [i for i in intervals if i[1] - i[0] >= min_width]

# --- Compute IoU ---
def compute_iou(pred_mask, true_mask, threshold):
    pred_mask = (pred_mask > threshold).astype(bool)
    true_mask = true_mask.astype(bool)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union != 0 else 1.0

# --- Combined U-Net 1D Inspired CNN + MLP ---
def build_cnn_mlp(input_dim):
    return nn.Sequential(
        nn.Conv1d(1, 64, kernel_size=9, padding=4),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.Conv1d(64, 128, kernel_size=5, padding=2),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.Flatten(),
        nn.Linear((input_dim // 4) * 128, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, input_dim),
        nn.Sigmoid()
    )

# --- Training ---
def train_model(X_np, y_np, input_dim, epochs=50):
    X_train, X_val, y_train, y_val = train_test_split(X_np, y_np, test_size=0.1, random_state=42)
    X_train_tensor = torch.tensor(X_train[:, None, :], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = build_cnn_mlp(input_dim)
    loss_fn = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)

        with torch.no_grad():
            val_pred = model(torch.tensor(X_val[:, None, :], dtype=torch.float32)).numpy()
            best_iou = 0
            best_threshold = 0.0
            for t in np.arange(0.1, 0.6, 0.02):
                ious = [compute_iou(val_pred[i], y_val[i], threshold=t) for i in range(len(y_val))]
                avg_iou = np.mean(ious)
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    best_threshold = t
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Best IoU: {best_iou:.4f} @ Threshold {best_threshold:.2f}")

    return model

# --- Main ---
if __name__ == "__main__":
    file_path = "train.h5"
    waveforms, labels = load_h5_data(file_path, max_samples=24000)
    spectra = np.array([to_power_spectrum(w) for w in waveforms])
    spectra = (spectra - spectra.mean(axis=1, keepdims=True)) / (spectra.std(axis=1, keepdims=True) + 1e-8)
    f_grid = np.linspace(2400, 2500, spectra.shape[1])
    masks = np.array([label_to_mask(l, f_grid) for l in labels])

    model = train_model(spectra, masks, input_dim=spectra.shape[1], epochs=50)

    # Save predictions.txt for test_public.h5
    test_path = "test_public.h5"
    with h5py.File(test_path, 'r') as f:
        test_waveforms = f['waveforms'][:]
        test_questions = [s.decode('utf-8') for s in f['questions'][:]]

    test_spectra = np.array([to_power_spectrum(w) for w in test_waveforms])
    test_spectra = (test_spectra - test_spectra.mean(axis=1, keepdims=True)) / (test_spectra.std(axis=1, keepdims=True) + 1e-8)

    f_grid = np.linspace(2400, 2500, test_spectra.shape[1])
    pred_lines = ["id, prediction"]

    with torch.no_grad():
        pred_probs = model(torch.tensor(test_spectra[:, None, :], dtype=torch.float32)).numpy()
        for i in range(len(test_questions)):
            question = test_questions[i]
            mask = pred_probs[i]
            # extract freq range from question
            freqs = list(map(float, question.split("within the range")[-1].strip(". ").replace("MHz", "").split("-")))
            start_freq, end_freq = min(freqs), max(freqs)
            window_mask = (f_grid >= start_freq) & (f_grid <= end_freq)
            partial_mask = mask * window_mask
            intervals = mask_to_intervals(partial_mask, f_grid, threshold=0.3)
            filtered = filter_short_intervals(intervals)
            pred_lines.append(f"{i}, {json.dumps(filtered)}")

    with open("predictions.txt", "w") as f:
        f.write("\n".join(pred_lines))

    print("Saved predictions.txt with", len(pred_lines)-1, "samples.")
