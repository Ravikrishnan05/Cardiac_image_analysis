# 🫀 Automated Cardiac Segmentation with U-Net  
**A Deep Learning Approach for Semantic Segmentation in Cardiac MRI**

This project implements a complete deep learning pipeline for semantic segmentation of cardiac structures from 2D MRI slices. Using the **ACDC MICCAI 2017** dataset, we deploy a **U-Net architecture** to accurately segment:
- 🫁 Right Ventricle (RV)  
- 🧠 Myocardium (MYO)  
- ❤️ Left Ventricle (LV)  

The goal is to **automate cardiac structure extraction**, a critical step in diagnosing cardiovascular diseases and computing clinical measures like **ejection fraction**.

---

## 📽️ Demo
The animation below shows the model's predictions on unseen validation data:
- Left: Input MRI slice  
- Center: Ground truth mask  
- Right: Model prediction  

![Model Output GIF](https://via.placeholder.com/800x250.png?text=Input+MRI+vs+Ground+Truth+vs+Model+Prediction+GIF)

---

## ✨ Key Features
- ✅ **Improved U-Net** with Batch Normalization & Dropout in every block  
- 🧠 **Hybrid Loss**: Weighted Cross-Entropy + Dice Loss for pixel accuracy + shape coherence  
- ⚖️ **Class Imbalance Handling** with Median Frequency Balancing  
- 📊 **Stratified Data Splitting** based on class dominance per slice  
- 🛠️ **Keras Training Suite**: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`

---

## 📁 Dataset
- **Source**: ACDC Challenge Dataset – MICCAI 2017  
- **Location**: `/kaggle/input/automated-cardiac-diagnosis-challenge-miccai17/`  
- **Format**: NIfTI `.nii` files  
- **Structure**:
  - 100 patient folders
  - Each contains 2D MRI slices and corresponding masks  
- **Classes**:
  - `0`: Background  
  - `1`: Right Ventricle (RV)  
  - `2`: Myocardium (MYO)  
  - `3`: Left Ventricle (LV)  

---

## 🛠️ Methodology

### 1. 📦 Preprocessing
- Loaded 3D `.nii` volumes using `nibabel`
- Extracted 2D slices and filtered irrelevant ones (foreground > 0.1%)
- Standardized input:
  - Resized to `128x128` (linear for images, nearest for masks)
  - Normalized pixel values to `[0,1]`
  - One-hot encoded masks to `(H, W, 4)`

### 2. 🧠 Model Architecture: U-Net (Improved)
- **Encoder**: 3x3 Conv → BatchNorm → ReLU → Dropout → MaxPool  
- **Decoder**: Transposed Conv → Skip Connections → Conv blocks  
- **Skip Connections**: Fuse encoder & decoder feature maps to recover spatial details

![U-Net Architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)
<p align="center">Original U-Net architecture diagram by Ronneberger et al.</p>

### 3. � Custom Loss Function

Total_Loss = Weighted_Categorical_CrossEntropy + Weighted_Dice_Loss

WCCE: Penalizes misclassified pixels with class weighting
Dice: Optimizes overlap; crucial for shape integrity
Combined Loss improves both spatial and pixel-level performance



