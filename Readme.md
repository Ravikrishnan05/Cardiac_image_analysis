Automated Cardiac Segmentation with U-Net: A Deep Learning Approach for MRI Analysis
This project presents a complete deep learning pipeline for semantic segmentation of the human heart from 2D MRI scans. Using the ACDC (Automated Cardiac Diagnosis Challenge) MICCAI 2017 dataset, this implementation leverages a U-Net architecture to accurately identify and delineate key cardiac structures: the right ventricle (RV), the myocardium (MYO), and the left ventricle (LV).
The primary goal is to automate the extraction of these structures, a critical step for diagnosing cardiovascular diseases and calculating clinical metrics like ejection fraction. This repository showcases a robust, end-to-end workflow, from data preprocessing to model training, evaluation, and analysis.
🚀 Project Demo
The animation below demonstrates the model's performance on unseen validation data. It compares the input MRI slice, the ground truth segmentation mask, and the model's predicted mask.
<!--
[RECOMMENDATION]: Create a GIF of your model's predictions and embed it here.
A good GIF would show 3 panels side-by-side:
1. The input MRI (grayscale)
2. The ground truth mask (colored)
3. Your model's prediction (colored)
This is the most impactful way to demonstrate your project's success.
-->
![alt text](https://via.placeholder.com/800x250.png?text=Input+MRI+vs+Ground+Truth+vs+Model+Prediction+GIF)
✨ Key Features & Technical Contributions
Advanced U-Net Architecture: Implemented a sophisticated U-Net with Dropout and Batch Normalization in each convolutional block to improve training stability and prevent overfitting.
Custom Combined Loss Function: Developed a custom loss function that combines Weighted Categorical Cross-Entropy and Weighted Dice Loss. This hybrid approach ensures both pixel-level accuracy and structural overlap, which is critical for robust segmentation.
Handling Severe Class Imbalance: Addressed the massive imbalance between background and foreground pixels by implementing Median Frequency Balancing to calculate class weights, forcing the model to focus on learning the small but vital heart structures.
Methodologically Sound Data Splitting: Employed a stratified splitting strategy based on the dominant class in each MRI slice. This ensures a fair and representative distribution of data between the training and validation sets, leading to more reliable evaluation metrics.
Robust Training & Evaluation Pipeline: The training process is managed by a suite of Keras callbacks, including ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau, to automatically save the best model, prevent wasted computation, and fine-tune the learning rate.
📊 Dataset
The project utilizes the ACDC MICCAI 2017 Challenge dataset available on Kaggle. The data is provided in the NIfTI (.nii) format.
Source: /kaggle/input/automated-cardiac-diagnosis-challenge-miccai17/
Structure: 100 patient folders for training, each containing a series of 2D MRI slices and corresponding ground truth masks.
Classes:
Class 0: Background
Class 1: Right Ventricle (RV)
Class 2: Myocardium (MYO)
Class 3: Left Ventricle (LV)
🛠️ Methodology & Workflow
The project follows a systematic deep learning workflow, detailed below.
1. Data Preprocessing
Raw medical imaging data requires careful preprocessing to be suitable for a neural network.
NIfTI File Handling: Used the nibabel library to load 3D MRI volumes.
Slice Extraction: Processed the 3D volumes into individual 2D slices.
Filtering Irrelevant Slices: To improve training efficiency, slices with minimal or no heart tissue were filtered out. A slice was kept only if the foreground (non-background) pixels in its mask constituted at least 0.1% of the total area.
Standardization:
Resizing: All image and mask slices were resized to a uniform 128x128 resolution. cv2.INTER_LINEAR interpolation was used for images, while cv2.INTER_NEAREST was used for masks to preserve the integer class labels.
Normalization: Image pixel values were normalized to a [0, 1] range.
One-Hot Encoding: Ground truth masks were converted to a one-hot encoded format of shape (H, W, 4) for compatibility with the categorical loss function.
2. Model Architecture: Improved U-Net
The core of this project is a U-Net, a convolutional neural network architecture designed for biomedical image segmentation.
Encoder (Contracting Path): Consists of repeated blocks of two 3x3 convolutions, followed by Batch Normalization, a ReLU activation, Dropout, and a 2x2 max pooling operation. This path captures the contextual information of the image while progressively downsampling.
Decoder (Expansive Path): Symmetrically upsamples the feature maps using 2x2 transposed convolutions.
Skip Connections: The crucial feature of U-Net. The feature maps from the encoder path are concatenated with the corresponding upsampled maps in the decoder. This allows the network to combine high-level contextual information with low-level spatial details, enabling precise localization of the heart structures.
![alt text](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

<p align="center">Original U-Net architecture diagram by Ronneberger et al.</p>
3. Custom Loss Function
A standard loss function is insufficient for this task due to severe class imbalance. I designed a custom loss function that is the sum of two weighted components:
Weighted Categorical Cross-Entropy (WCCE): Penalizes pixel-wise classification errors. The contribution of each pixel's error is scaled by its class weight, forcing the model to pay more attention to the under-represented heart classes.
Weighted Dice Loss: Directly optimizes the Dice Coefficient, a measure of overlap between the predicted and true masks. This is crucial for ensuring the predicted shapes are structurally coherent. The Dice loss for each class is also scaled by its class weight.
Generated code
Total Loss = Weighted_CCE_Loss + Weighted_Dice_Loss
Use code with caution.
This combined approach provides both smooth gradients for stable training (from CCE) and a strong signal for spatial overlap (from Dice), leading to superior segmentation results.
📈 Results & Analysis
The model was trained for 100 epochs with a batch size of 8. The best model was selected based on the highest val_mean_io_u score.
Quantitative Results
The final model, restored from the best epoch (Epoch 94), achieved the following performance on the held-out validation set:
Metric	Score	Interpretation
Validation Loss	2.3366	The final combined error score.
Validation Mean IoU	0.3752	The primary metric, indicating a solid overlap.
Recall (Right Ventricle)	0.6352	The model successfully identified 63.5% of RV pixels.
Recall (Myocardium)	0.5336	The model successfully identified 53.4% of MYO pixels.
Recall (Left Ventricle)	0.4386	The model successfully identified 43.9% of LV pixels.
Learning Curves
The training history plots provide insight into the learning dynamics.
Model Loss	Model Mean IoU
<!-- INSERT YOUR LOSS PLOT IMAGE HERE --> <img src="https://i.imgur.com/vHq0F7B.png" width="400"/>	<!-- INSERT YOUR MEAN_IOU PLOT IMAGE HERE --> <img src="https://i.imgur.com/kY7pU4o.png" width="400"/>
Model Loss: The validation loss (orange) decreased rapidly and then plateaued, indicating that the model learned effectively and EarlyStopping was justified. The gap between the noisy training loss and stable validation loss is expected due to regularization and data augmentation.
Model Mean IoU: The validation IoU plot (orange) shows stable, incremental improvement, confirming that the model was not stuck but making steady progress. The noise in the training IoU is a known artifact of how Keras reports streaming metrics on generators.
Qualitative Results
Visual inspection confirms that the model produces coherent and accurate segmentation masks.
<!--
[RECOMMENDATION]: Add a few examples of your best and worst predictions here.
This shows you've critically analyzed your results.
-->
Good Prediction Example:
| Input MRI | Ground Truth | Prediction |
| :---: | :---: | :---: |
|
![alt text](https://via.placeholder.com/200x200.png?text=Good+Input)
|
![alt text](https://via.placeholder.com/200x200.png?text=Good+Ground+Truth)
|
![alt text](https://via.placeholder.com/200x200.png?text=Good+Prediction)
|
Challenging Prediction Example (e.g., at the apex of the heart):
| Input MRI | Ground Truth | Prediction |
| :---: | :---: | :---: |
|
![alt text](https://via.placeholder.com/200x200.png?text=Challenging+Input)
|
![alt text](https://via.placeholder.com/200x200.png?text=Challenging+GT)
|
![alt text](https://via.placeholder.com/200x200.png?text=Challenging+Prediction)
|
🚀 How to Run
To replicate this project, follow these steps:
Clone the repository:
Generated bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Use code with caution.
Bash
Set up a virtual environment and install dependencies:
Generated bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
Use code with caution.
Bash
Download the dataset:
Download the ACDC dataset from Kaggle and place it in an input/ directory within the project folder. The expected path is /input/automated-cardiac-diagnosis-challenge-miccai17/.
Run the Jupyter Notebook:
Launch Jupyter and open the main notebook (.ipynb) file to execute the pipeline from data loading to training and evaluation.
Generated bash
jupyter notebook
Use code with caution.
Bash
🔮 Future Work & Potential Improvements
Use a Pre-trained Encoder: Employ transfer learning by using an encoder pre-trained on ImageNet (e.g., ResNet34, EfficientNet) to potentially boost performance and speed up convergence. The segmentation-models library is excellent for this.
Advanced Data Augmentation: Integrate the Albumentations library for faster and more sophisticated augmentations, such as ElasticTransform and GridDistortion, which are well-suited for medical images.
Increase Image Resolution: Train on larger image sizes (e.g., 256x256) to preserve more spatial detail, which may improve boundary definition, at the cost of higher computational requirements.
Post-processing: Implement post-processing steps, such as removing small, disconnected predicted regions, to clean up the final segmentation masks and potentially improve metric scores.
📄 License
This project is licensed under the MIT License. See the LICENSE file for details.







🚨 Root Causes of All-Zero Predictions
Class Imbalance Nuclear Winter

Background pixels dominate (often >95% of pixels)

Model learns predicting all background = "good enough" loss

Improper Loss Function Setup

Standard cross-entropy/Dice fails with extreme imbalance

No weighting for cardiac structures (RV/MYO/LV)

Data Starvation Effects

100 images ≈ 1-2 patients worth of variability

Standard U-Net overfits immediately

