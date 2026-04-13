From Trees to Transformers: Call Quality Prediction- 
This repository contains the implementation of a comparative study evaluating the performance of ensemble tree-based models and deep learning architectures for predicting customer-perceived voice call quality. The project introduces a Fused Model architecture that combines a Random Forest regressor with an FT-Transformer (Feature Tokenizer Transformer) trained on residuals to achieve superior predictive accuracy

📌 Project Overview
The core objective is to determine if hybrid systems can outperform standalone models on tabular datasets. The study utilizes the Voice Call Quality Customer Experience dataset from the Telecom Regulatory Authority of India (TRAI).

Key Findings:
The Fused Model (RF + FT-Transformer) outperformed both standalone models.Standalone tree-based models (Random Forest) continue to show high robustness on moderate-sized tabular datasets compared to standalone deep learning.Combining these architectures allows the system to capture both broad non-linear structures and fine-grained feature interactions.

🏗️ Model Architectures:
Three distinct configurations were evaluated:
1.)Random Forest (Baseline): An ensemble of 300 decision trees trained directly on customer ratings.
2.)FT-Transformer (Standalone): A deep learning architecture designed for tabular data using multi-head self-attention and feature tokenization.
3.)Fused Model: A hybrid system where the Random Forest provides a base prediction, and the FT-Transformer is trained to predict the remaining residuals (errors). The final output is the sum of both predictions.


📊 Results (Final Evaluation on Test Set)
The models were evaluated using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R².

Model	RMSE 	MAE 	R² 
Fused (RF + FT-Transformer)	0.6876	0.5976	0.7834
Random Forest (RF)	0.6970	0.6059	0.7774
FT-Transformer (Standalone)	0.7048	0.6246	0.7724

🛠️ Technical Stack & Methodology
1.) Language: Python
2.) Deep Learning Framework: PyTorch (FT-Transformer implemented from scratch)
3.) Data Processing: Pandas, Scikit-learn (StandardScaler, PCA) Feature Engineering:Principal Component Analysis (PCA) for geographic coordinates (latitude/longitude).Ordinal encoding for call quality categories.One-hot encoding for nominal features like operator and network type.
4.) Data Source: Programmatic extraction via the Government of India's Open Data Portal (data.gov.in) API.

📂 Dataset Description
The dataset includes metrics from September 2017:Features: State, Operator (Airtel, Jio, etc.), Network Type (2G/3G/4G), Call Drop Category, and Geographic coordinates.Target: rating (Continuous numerical score assigned by customers).

🚀 Future Work
1.) Hyperparameter optimization using tools like Optuna.
2.) Incorporating longitudinal data to analyze temporal trends.
3.) Exploring iterative boosting with deep learning "weak learners".

📑 References
1.) Vaswani et al. (2017) - Attention Is All You Need.
2.) Gorishniy et al. (2021) - Revisiting Deep Learning Models for Tabular Data.
3.) Grinsztajn et al. (2022) - Why tree-based models still outperform deep learning on tabular data.
4.) Breiman (2001) - Random Forests





