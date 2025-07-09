import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import welch
import h5py
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

# --- Power Spectrum ---
def to_power_spectrum(signal):
    _, Pxx = welch(signal, fs=1e8, nperseg=1024, return_onesided=False)
    return Pxx

# --- Dual Mask Builder ---
def build_dual_masks(intervals, f_grid):
    count = np.zeros_like(f_grid)
    for start, end in intervals:
        count[(f_grid >= start) & (f_grid <= end)] += 1
    return (count >= 1).astype(float), (count >= 2).astype(float)

# --- Interval Extractor ---
def mask_to_intervals(mask, grid, threshold=0.3):
    intervals = []
    active = False
    for i in range(len(mask)):
        if mask[i] > threshold and not active:
            start = grid[i]
            active = True
        elif mask[i] <= threshold and active:
            end = grid[i]
            intervals.append([round(start, 1), round(end, 1)])
            active = False
    if active:
        intervals.append([round(start, 1), round(grid[-1], 1)])
    return [i for i in intervals if i[1] - i[0] >= 0.5]

# --- IoU Computation ---
def compute_iou(preds, targets, thresholds):
    best_iou = 0
    best_thresh = 0
    for t in thresholds:
        bin_preds = (preds > t).astype(float)
        inter = (bin_preds * targets).sum(axis=1)
        union = ((bin_preds + targets) >= 1).sum(axis=1)
        ious = inter / (union + 1e-6)
        avg_iou = ious.mean()
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_thresh = t
    return best_iou, best_thresh

# --- Model Definition ---
class MultiTaskCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
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
            nn.Dropout(0.3)
        )
        self.head_occ = nn.Linear(1024, input_dim)
        self.head_int = nn.Linear(1024, input_dim)

    def forward(self, x):
        x = self.shared(x)
        return torch.sigmoid(self.head_occ(x)), torch.sigmoid(self.head_int(x))

# --- Training ---
def train_model(train_path, input_dim, max_samples=24000):
    with h5py.File(train_path, 'r') as f:
        waves = f['waveforms'][:max_samples]
        raw_labels = [s.decode('utf-8') for s in f['labels'][:max_samples]]
        labels = [eval(l) for l in raw_labels]

    f_grid = np.linspace(2400, 2500, input_dim)
    X = np.array([to_power_spectrum(w) for w in waves])
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    Y_occ, Y_int = zip(*[build_dual_masks(l, f_grid) for l in labels])

    Xt, Xv, Yo, Yv, Yi, Yvi = train_test_split(X, Y_occ, Y_int, test_size=0.1, random_state=42)
    loader = DataLoader(TensorDataset(
    torch.tensor(Xt[:, None, :], dtype=torch.float32),
    torch.tensor(np.array(Yo), dtype=torch.float32),
    torch.tensor(np.array(Yi), dtype=torch.float32)
    ), batch_size=32, shuffle=True)


    model = MultiTaskCNN(input_dim)
    loss_fn = FocalLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_occ_iou, best_occ_thresh, best_int_iou, best_int_thresh = 0, 0.3, 0, 0.3
    best_model_state = None
    thresholds = np.linspace(0.1, 0.5, 9)

    for ep in range(1, 51):
        model.train(); total = 0
        for xb, y_occ, y_int in loader:
            po, pi = model(xb)
            loss = loss_fn(po, y_occ) + loss_fn(pi, y_int)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            po_val, pi_val = model(torch.tensor(Xv[:, None, :], dtype=torch.float32))
            po_val = po_val.numpy(); pi_val = pi_val.numpy()
            occ_iou, occ_thr = compute_iou(po_val, np.array(Yv), thresholds)
            int_iou, int_thr = compute_iou(pi_val, np.array(Yvi), thresholds)

        if occ_iou + int_iou > best_occ_iou + best_int_iou:
            best_occ_iou, best_occ_thresh = occ_iou, occ_thr
            best_int_iou, best_int_thresh = int_iou, int_thr
            best_model_state = model.state_dict()

        print(f"Epoch {ep:2d} | Loss: {total/len(loader):.4f} | Occ IoU: {occ_iou:.4f} (t={occ_thr:.2f}) | Int IoU: {int_iou:.4f} (t={int_thr:.2f})")

    torch.save({
        'model': best_model_state,
        'occ_thresh': best_occ_thresh,
        'int_thresh': best_int_thresh
    }, "model_multitask_best.pth")
    print("Best model saved with thresholds:", best_occ_thresh, best_int_thresh)

# --- Inference ---
def inference(test_path, model_path, output_path="predictions.txt"):
    with h5py.File(test_path, 'r') as f:
        waves = f['waveforms'][:]
        questions = [s.decode('utf-8') for s in f['questions'][:]]

    f_grid = np.linspace(2400, 2500, 1024)
    X = np.array([to_power_spectrum(w) for w in waves])
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    checkpoint = torch.load(model_path)
    model = MultiTaskCNN(1024)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    t_occ = checkpoint['occ_thresh']
    t_int = checkpoint['int_thresh']

    with torch.no_grad():
        po, pi = model(torch.tensor(X[:, None, :], dtype=torch.float32))
        po = po.numpy(); pi = pi.numpy()

    lines = ["id, prediction"]
    for i, q in enumerate(questions):
        head = pi[i] if "overlapping" in q.lower() else po[i]
        thresh = t_int if "overlapping" in q.lower() else t_occ
        f1, f2 = map(float, q.split("within the range")[-1].replace("MHz", "").strip(". ").split("-"))
        mask = (f_grid >= min(f1, f2)) & (f_grid <= max(f1, f2))
        pred = head * mask
        pred = uniform_filter1d(pred, size=3)
        intervals = mask_to_intervals(pred, f_grid, threshold=thresh)
        lines.append(f"{i}, {json.dumps(intervals)}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Predictions written to {output_path}")

if __name__ == '__main__':
    train_model("train.h5", input_dim=1024)
    inference("test_public.h5", "model_multitask_best.pth")
