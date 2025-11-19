# Circle vs. Square Classification with CNN

This repository contains a convolutional neural network (CNN) that classifies simple images into two classes: **circles** and **squares**.  
The project was originally prepared as an assignment for the *Neural Networks and Deep Learning* course.

Author: **Michał Kaczmarek**  

---

## 1. Problem Description

- **Task:** Binary image classification  
- **Classes:**
  - `0` – square  
  - `1` – circle  
- **Input:** RGB images resized to `64x64`  
- **Output:** Probability that the image contains a circle (sigmoid output)

The model is trained and evaluated in two variants:

1. **Without data augmentation**
2. **With data augmentation** (rotation, shifting, zooming, flipping, etc.)

Both models are evaluated on the same test set.

---

## 2. Dataset

- **Source:** http://www.kasprowski.pl/datasets/circrec.zip  
- The script automatically:
  - downloads the ZIP file (if not present),
  - extracts it to the `circrec/` directory,
  - loads images and their labels based on filename convention:  
    `img_<number>_<class>.jpg`, where `<class>` ∈ {0, 1}.

---

## 3. Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── main.py
└── reports/
    └── figures/
```


- `src/main.py` — full pipeline (download, preprocessing, training, evaluation)  
- `reports/figures/` — training curves, confusion matrices, and sample images

---

## 4. Installation

```bash
pip install -r requirements.txt
```

## 5. How to Run

```
python src/main.py
```
The script will:

- Download and extract the dataset
- Load and preprocess all images
- Split data into train/test (50/50, stratified)
- Train two CNN models:
  - without augmentation
  - with augmentation
- Save all plots to `reports/figures/`
- Evaluate both models:
  - confusion matrix
  - Cohen’s Kappa score
  
---
## 6. Model Architecture

Both models use the same CNN:

- Conv2D(32, 3×3) + MaxPooling2D  
- Conv2D(64, 3×3) + MaxPooling2D  
- Conv2D(128, 3×3) + MaxPooling2D  
- Flatten  
- Dense(128, ReLU)  
- Dropout(0.5)  
- Dense(1, sigmoid)

**Loss:** binary crossentropy  
**Optimizer:** Adam (lr=1e-4)  
**Metric:** accuracy  
**Callbacks:** early stopping + checkpointing  

---

## 7. Evaluation

For both models the script produces:

- training/validation accuracy curves  
- training/validation loss curves  
- confusion matrices  
- Cohen’s Kappa score  

Because the dataset is simple and uniform, the non-augmented model may perform slightly better on the validation set — augmented samples differ more from the fixed test set.

## 8. Results

- Test accuracy (both models): 100%
- Cohen’s Kappa: 1.00 for both models
- Confusion matrices and training curves are available in `reports/figures/`.
