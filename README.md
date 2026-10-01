# Corn Leaf Disease Classification: AlexNet vs EfficientNetB0

Image classification of corn leaves into four classes (**Blight**, **Common Rust**, **Gray Leaf Spot**, **Healthy**). An AlexNet-style CNN trained from scratch is used as the baseline, then replaced with an EfficientNetB0 transfer learning model with class weighting to handle class imbalance.

**Result:** test accuracy improved from **84.29% to 93.97%** and macro F1 from **0.7982 to 0.9226**, with the largest gain on the hardest class (Gray Leaf Spot recall: 0.46 to 0.83).

## Key Objectives

- **EDA:** Check class distribution, file formats, image dimensions and aspect ratios, color modes, and RGB histograms per class.
- **Data Splitting:** Stratified 70/15/15 train/validation/test split per class.
- **Preprocessing:** Resize to 224×224, normalize pixels, and apply data augmentation on the training set.
- **Baseline CNN:** An AlexNet-style architecture trained from scratch.
- **Modified Model:** EfficientNetB0 with ImageNet weights (frozen base), a regularized classification head, and class weights.
- **Evaluation:** Accuracy, macro precision, macro recall, and macro F1 on the test set, plus per-class reports and confusion matrices.

## Dataset

Corn leaf disease images, 4,188 in total.

| Class | Images |
|-------|--------|
| Blight | 1,146 |
| Common Rust | 1,306 |
| Gray Leaf Spot | 574 |
| Healthy | 1,162 |

**EDA observations**
- Class imbalance: Gray Leaf Spot has far fewer images than the other classes.
- Most images (3,852) are 256×256. The remaining images in the disease classes have varied sizes and aspect ratios (some up to 5184 px), while all Healthy images are 256×256.
- Color modes: 4,183 RGB, 4 RGBA, 1 CMYK. The data loader converts all images to 3-channel RGB.
- RGB histograms show Healthy leaves dominated by the green channel, while diseased classes show shifts in the red/blue channels and more dark pixels (notably Common Rust).

**Split (stratified per class)**

| Class | Train | Val | Test |
|-------|-------|-----|------|
| Blight | 802 | 172 | 172 |
| Common Rust | 914 | 196 | 196 |
| Gray Leaf Spot | 401 | 86 | 87 |
| Healthy | 812 | 175 | 175 |
| **Total** | **2,929** | **629** | **630** |

## Preprocessing

- Image size 224×224, batch size 32, pixel values rescaled to [0, 1]
- Data augmentation (training set only): horizontal flip, rotation (0.15), zoom (0.15), translation (0.1), contrast (0.1)

## Models

### 1. Baseline: AlexNet-style CNN (from scratch)
- Conv 96 (11×11, stride 4) → MaxPool → Conv 256 (5×5) → MaxPool → Conv 384 → Conv 384 → Conv 256 (3×3) → MaxPool
- Flatten → Dense 4096 → Dropout 0.5 → Dense 4096 → Dropout 0.5 → Softmax (4 classes)
- 46.8M parameters, Adam (lr 1e-3), 20 epochs

### 2. Modified: EfficientNetB0 (transfer learning)
- EfficientNetB0 (ImageNet weights, frozen) → Global Average Pooling → Dropout 0.5 → Dense 128 (ReLU, L2 1e-4) → Dropout 0.5 → Softmax
- **Class weights (balanced):** Blight 0.913, Common Rust 0.801, Gray Leaf Spot 1.826, Healthy 0.902
- Adam (lr 1e-3), up to 30 epochs, Early Stopping (patience 10, best weights restored), ReduceLROnPlateau (factor 0.5, patience 3)

## Results

Test set (630 images):

| Metric | Baseline AlexNet | Modified EfficientNetB0 | Improvement |
|--------|------------------|-------------------------|-------------|
| Accuracy | 0.8429 | **0.9397** | +0.0968 |
| Precision (Macro) | 0.8159 | **0.9211** | +0.1052 |
| Recall (Macro) | 0.7937 | **0.9242** | +0.1305 |
| F1 Score (Macro) | 0.7982 | **0.9226** | +0.1244 |

**Per-class results (F1 / Recall)**

| Class | Baseline F1 | Baseline Recall | Modified F1 | Modified Recall |
|-------|-------------|-----------------|-------------|-----------------|
| Blight | 0.7828 | 0.8488 | 0.9038 | 0.9012 |
| Common Rust | 0.9125 | 0.8776 | 0.9871 | 0.9796 |
| Gray Leaf Spot | 0.5442 | 0.4598 | 0.8136 | 0.8276 |
| Healthy | 0.9532 | 0.9886 | 0.9858 | 0.9886 |

## Key Findings

- **Baseline:** reached a validation F1 of 0.7887 with minor fluctuations in late epochs. Its main weakness was Gray Leaf Spot, with a recall of only 0.46, meaning more than half of those leaves were missed.
- **Modified model:** the EfficientNetB0 model gave smoother training curves and a validation F1 of 0.9158. Gray Leaf Spot recall rose to 0.83 and its F1 from 0.54 to 0.81. Since transfer learning and class weighting were introduced together, their individual contributions were not separated.
- **Remaining gap:** Gray Leaf Spot is still the weakest class (F1 0.81), compared with 0.90 to 0.99 for the other classes.

## Conclusion

This project compared a CNN trained from scratch with a transfer learning approach for corn leaf disease classification on a relatively small and imbalanced dataset. Using a pretrained EfficientNetB0 with a regularized head and class weights raised test accuracy to 93.97% and macro F1 to 0.9226, and made detection of the minority Gray Leaf Spot class much more reliable.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras (EfficientNetB0 from Keras Applications)
- **Evaluation & Utilities:** scikit-learn
- **Data Handling & Visualization:** NumPy, Pandas, Matplotlib, Seaborn, Pillow
- **Environment:** Kaggle Notebooks
