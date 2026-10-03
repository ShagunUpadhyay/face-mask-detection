# Real-Time Face Mask Detection

A real-time face mask detector that uses OpenCV (Haar Cascade) for face detection
and a MobileNetV2 transfer-learning classifier built with TensorFlow/Keras.

## Demo

![demo](screenshots/demo.png)

## How it works

1. A Haar Cascade detects faces in each webcam frame.
2. Each face is cropped, resized to 224×224 and preprocessed with MobileNetV2's `preprocess_input`.
3. The classifier predicts **Mask** or **No Mask**.
4. A green (mask) or red (no mask) box with the confidence score is drawn on the frame.

## Model

- Base: MobileNetV2 pretrained on ImageNet (frozen)
- Head: AveragePooling → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(2, Softmax)
- Loss: categorical cross-entropy, optimizer: Adam (lr = 1e-4)
- Augmentation: flip, rotation, zoom, translation

## Dataset

<!-- TODO: fill this in -->
Dataset: [name and link here]
- With mask: XXXX images
- Without mask: XXXX images
- Split: 80% train / 20% validation

Expected folder layout (not included in this repo):

```
dataset/
├── with_mask/
└── without_mask/
```

## Results

<!-- TODO: replace with the real numbers printed by train_mask_detector.py -->
| Metric | Value |
|--------|-------|
| Validation accuracy | XX% |
| Precision | XX |
| Recall | XX |

![training plot](training_plot.png)

## Installation

```bash
git clone https://github.com/<your-username>/face-mask-detection.git
cd face-mask-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train the model (creates `mask_detector.keras`, `labels.json`, `training_plot.png`):

```bash
python train_mask_detector.py --dataset dataset --epochs 10
```

Run the webcam detector (press `q` to quit):

```bash
python detect_mask_video.py
```

Optional arguments: `--camera 1` to use another webcam, `--model path/to/model.keras`.

## Project structure

```
face-mask-detection/
├── train_mask_detector.py
├── detect_mask_video.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations

- Haar Cascade struggles with side-facing faces and sometimes misses faces already wearing masks.
- Accuracy depends on lighting and on how diverse the training data is.
- Not intended for safety-critical use.

## Future improvements

- Replace Haar Cascade with a DNN-based face detector (e.g. SSD or MediaPipe)
- Fine-tune the top layers of MobileNetV2
- Deploy as a web app (Streamlit / Flask)

## Tech stack

Python, TensorFlow/Keras, OpenCV, NumPy, scikit-learn, Matplotlib

## Author

Your Name · [SHAGUN UPADHYAY](https://github.com/ShagunUpadhyay)
