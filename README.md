
# Radio Spectrum Sensing with 1D CNN

This repository provides a complete pipeline for detecting occupied and interfering frequency bands in the 2400–2500 MHz radio spectrum using deep learning. The model is designed for the AI+Radio Challenge and achieves high accuracy by transforming IQ samples into Power Spectral Density (PSD) features and training a CNN-MLP network.

---

## Project Structure

```
├── train_1D_1.py            # Main training & inference script
├── train.h5                 # Training dataset (IQ waveforms + labels)
├── test_public.h5           # Public test dataset with questions
├── predictions.txt          # Output predictions for submission
├── data_intro_en.md         # Dataset description and scoring rules
├── Model info.docx          # Technical documentation of model and environment
```

---

## Model Architecture

The model follows a **1D CNN + MLP** architecture:
- **Input**: 1024-bin normalized PSD vector (derived from 100,000-sample IQ signal)
- **Layers**:
  - Conv1D → BN → ReLU → MaxPool
  - Conv1D → BN → ReLU → MaxPool
  - Flatten → Dense → Dropout → Dense → Sigmoid
- **Output**: 1024-length probability mask (0–1), each bin corresponds to a frequency band
- **Loss Function**: Focal Loss (handles class imbalance)

---

## Dataset Details

### `train.h5`
- `waveforms`: 24,000 IQ samples, each 100,000 points, sampled at 100 MS/s
- `labels`: UTF-8 encoded frequency bands per sample, e.g.:
  ```
  '[[2402.0, 2422.0], [2432.0, 2472.0]]'
  ```

### `test_public.h5`
- `waveforms`: 1,000 samples like above
- `questions`: Natural language prompts (e.g. “Find the occupied frequency segments within 2420–2460 MHz”)

---

## Training Process

1. Convert each IQ waveform → 1024-bin PSD (Welch's method)
2. Normalize each PSD
3. Convert string-based labels into binary masks (float32)
4. Train for 50 epochs:
   - Batch size: 32
   - Optimizer: Adam
   - Learning rate: `1e-4`
   - Weight decay: `1e-4`
   - Evaluation: IoU across thresholds (0.1 to 0.6)

---

## Inference & Prediction

For each test sample:
1. Compute PSD and normalize
2. Use trained model to get a soft mask
3. Apply frequency window based on the question
4. Convert mask to intervals (with threshold + smoothing)
5. Filter short intervals (<0.5 MHz)
6. Output predictions in `predictions.txt`:

```csv
id, prediction
0, [[2452, 2462], [2475, 2477]]
1, [[2432, 2452]]
```

---

## Evaluation Metric

Intersection over Union (IoU):

```
IoU = |S_pred ∩ (S_true ∩ S_wd)| / |S_pred ∪ (S_true ∩ S_wd)|
```

Where:
- `S_pred`: Predicted intervals
- `S_true`: Ground truth intervals
- `S_wd`: Question-defined observation window

---

## Runtime Requirements

- Python ≥ 3.8
- Install dependencies:

```bash
pip install numpy scipy torch h5py scikit-learn matplotlib
```

---

## Run the Pipeline

To train and evaluate:

```bash
python train_1D_1.py
```

It will:
- Train on `train.h5`
- Generate `predictions.txt` for `test_public.h5`

---

## License

This project is released under the MIT License.

---

## Authors

Developed as part of the **AI+Radio Challenge 2025**. For inquiries or feedback, feel free to reach out via GitHub Issues.
