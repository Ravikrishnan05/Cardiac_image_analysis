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

