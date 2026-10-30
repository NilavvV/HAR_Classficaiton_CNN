# Human Activity Recognition with a 1D CNN

A small 1D convolutional network in PyTorch that classifies six activities
(walking, walking upstairs, walking downstairs, sitting, standing, laying) from raw
smartphone accelerometer and gyroscope signals in the UCI HAR dataset.

## Dataset

UCI Human Activity Recognition Using Smartphones. Each sample is a 2.56 s window of
9 inertial channels (body acc, body gyro, total acc; x/y/z) with 128 time steps.

Download it from the UCI Machine Learning Repository and unzip it so the layout is:

```
data/UCI HAR Dataset/
    train/   (subject_train.txt, y_train.txt, Inertial Signals/...)
    test/    (subject_test.txt,  y_test.txt,  Inertial Signals/...)
```
(note: This is not the actual layout of the dataset , here we organized it this way to fit it into our code)
The dataset is not included in this repository. To use a different location, set the
`HAR_DATA_DIR` environment variable or edit `DATA_DIR` at the top of the script.

Citation: Anguita, D., Ghio, A., Oneto, L., Parra, X., Reyes-Ortiz, J. L.
"A Public Domain Dataset for Human Activity Recognition Using Smartphones."
ESANN 2013.

## Model

Conv1d(9, 32, k=5) -> BatchNorm -> ReLU -> MaxPool(2)
-> Conv1d(32, 64, k=5) -> BatchNorm -> ReLU -> MaxPool(2)
-> Flatten -> Dropout(0.3) -> Linear(2048, 6)

Trained with Adam (lr 1e-3), batch size 64, 20 epochs, cross-entropy loss.

## Evaluation protocol

- 20% of the training subjects are held out as a validation set (split by subject,
  not by window, so no person appears in both).
- Normalization statistics come from the training split only.
- The best epoch is selected on validation accuracy.
- The official test set is evaluated once, at the end.

## Results

Fill these in after running the script or notebook.

| Metric | Value |
|---|---|
| Best validation accuracy | 0.9867 (epoch 4) |
| Test accuracy |  0.9257 |

The confusion matrix is saved to `results/confusion_matrix.png`.
<img width="691" height="637" alt="Screenshot 2026-10-30 at 4 57 28 PM" src="https://github.com/user-attachments/assets/1d7b662d-31b3-4e66-a470-b3d5b71ccf67" />


## Run it

```bash
pip install -r requirements.txt
python CNN_HAR.py
```

Or open `CNN_HAR.ipynb` and use Restart Kernel and Run All.
