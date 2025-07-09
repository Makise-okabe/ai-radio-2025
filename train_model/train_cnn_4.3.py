import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import welch
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter1d
import argparse
import re
import csv
import json


# ------------------ LOSS FUNCTIONS ------------------
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * BCE).mean()

        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (inputs * targets).sum(1)
        dice = (2. * intersection + self.smooth) / (inputs.sum(1) + targets.sum(1) + self.smooth)
        dice_loss = 1 - dice.mean()

        return 0.5 * focal_loss + 0.5 * dice_loss


# ------------------ UTILITY FUNCTIONS ------------------
def load_h5_data(file_path, max_samples=24000):
    with h5py.File(file_path, 'r') as f:
        waveforms = f['waveforms'][:max_samples]
        raw_labels = [s.decode('utf-8') for s in f['labels'][:max_samples]]
        labels = [eval(l) for l in raw_labels]
    return waveforms, labels

def to_power_spectrum(signal):
    _, Pxx = welch(signal, fs=1e8, nperseg=1024, return_onesided=False)
    return Pxx

def label_to_mask(intervals, f_grid):
    mask = np.zeros_like(f_grid, dtype=bool)
    for start, end in intervals:
        mask |= (f_grid >= start) & (f_grid <= end)
    return mask.astype(np.float32)

def mask_to_intervals(mask, f_grid, threshold=0.2):
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

def find_soft_overlaps(intervals, tolerance=0.2):
    overlaps = []
    n = len(intervals)
    for i in range(n):
        for j in range(i + 1, n):
            a_start, a_end = intervals[i]
            b_start, b_end = intervals[j]
            if max(a_start, b_start) < min(a_end, b_end) + tolerance:
                start = max(a_start, b_start)
                end = min(a_end, b_end)
                overlaps.append([start, end])
    return merge_intervals(overlaps)

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            prev[1] = max(prev[1], current[1])
        else:
            merged.append(current)
    return merged

def compute_iou(pred_mask, true_mask, threshold=0.2):
    pred_mask = (pred_mask > threshold).astype(bool)
    true_mask = true_mask.astype(bool)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union != 0 else 1.0


# ------------------ MODEL ------------------
def build_cnn_mlp(input_dim):
    return nn.Sequential(
        nn.Conv1d(1, 64, kernel_size=9, padding=4),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.Conv1d(64, 128, kernel_size=5, padding=2),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Conv1d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.Flatten(),
        nn.Linear((input_dim // 4) * 128, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, input_dim),
        nn.Sigmoid()
    )


# ------------------ TRAINING ------------------
def train_model(X_np, y_np, input_dim, epochs=50):
    X_train, X_val, y_train, y_val = train_test_split(X_np, y_np, test_size=0.1, random_state=42)
    X_train_tensor = torch.tensor(X_train[:, None, :], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = build_cnn_mlp(input_dim)
    loss_fn = FocalDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_threshold = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(X_val[:, None, :], dtype=torch.float32)).numpy()
            best_iou = 0
            for t in np.arange(0.1, 0.6, 0.02):
                ious = [compute_iou(val_pred[i], y_val[i], threshold=t) for i in range(len(y_val))]
                avg_iou = np.mean(ious)
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    best_threshold = t

        scheduler.step(best_iou)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}, Best IoU: {best_iou:.4f} @ T={best_threshold:.2f}")

    return model, best_threshold


# ------------------ INFERENCE ------------------
def parse_question(question):
    match = re.search(r'(\d+)\s*MHz\s*-\s*(\d+)\s*MHz', question)
    if not match:
        print("Warning: Could not parse frequency range from question.")
        return None, [2400.0, 2500.0]
    start_freq = float(match.group(1))
    end_freq = float(match.group(2))
    if "overlapping" in question.lower():
        return "interference", [start_freq, end_freq]
    else:
        return "occupancy", [start_freq, end_freq]

def crop_intervals(intervals, freq_range):
    start, end = freq_range
    return [[max(s, start), min(e, end)] for s, e in intervals if min(e, end) > max(s, start)]

def predict_single(iq_sample, question, model, f_grid, threshold=0.2):
    spectrum = to_power_spectrum(iq_sample)
    spectrum = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-8)
    model_input = torch.tensor(spectrum[None, None, :], dtype=torch.float32)
    with torch.no_grad():
        pred = model(model_input).squeeze().numpy()
    pred = uniform_filter1d(pred, size=3)
    task_type, freq_range = parse_question(question)
    occupied = mask_to_intervals(pred, f_grid, threshold=threshold)
    occupied = crop_intervals(occupied, freq_range)
    if task_type == "interference":
        overlapped = find_soft_overlaps(occupied, tolerance=0.2)
        return [[round(s, 1), round(e, 1)] for s, e in overlapped]
    else:
        return [[round(s, 1), round(e, 1)] for s, e in occupied]

def run_test_inference(test_h5_path, model, f_grid, threshold, output_file='predictions.txt'):
    with h5py.File(test_h5_path, 'r') as f:
        waveforms = f['waveforms'][:]
        questions = [q.decode('utf-8') for q in f['questions'][:]]

    with open(output_file, 'w') as csvfile:
        csvfile.write("id, prediction\n")
        for i, (sample, question) in enumerate(zip(waveforms, questions)):
            pred = predict_single(sample, question, model, f_grid, threshold)
            pred = merge_intervals(pred)
            csvfile.write(f"{i}, {json.dumps(pred)}\n")
            print(f"[{i}] {question} => {pred}")


    print(f"Saved predictions to {output_file}")


# ------------------ ENTRY POINT ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model_path', type=str, default='cnn_model.pt')
    parser.add_argument('--test_file', type=str, default='test_public.h5')
    parser.add_argument('--output', type=str, default='predictions.txt')
    args = parser.parse_args()

    input_dim = 1024
    f_grid = np.linspace(2400, 2500, input_dim)

    if args.mode == 'train':
        waveforms, labels = load_h5_data("train.h5", max_samples=24000)
        spectra = np.array([to_power_spectrum(w) for w in waveforms])
        spectra = (spectra - spectra.mean(axis=1, keepdims=True)) / (spectra.std(axis=1, keepdims=True) + 1e-8)
        masks = np.array([label_to_mask(l, f_grid) for l in labels])
        model, best_threshold = train_model(spectra, masks, input_dim=input_dim, epochs=50)
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

    elif args.mode == 'test':
        model = build_cnn_mlp(input_dim)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        run_test_inference(args.test_file, model, f_grid, threshold=0.2, output_file=args.output)
