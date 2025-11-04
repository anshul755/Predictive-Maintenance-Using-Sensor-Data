# 🚀 Predictive Maintenance Using Sensor Data

## Machine Learning-Based Estimation of Remaining Useful Life (RUL) in Turbofan Engines

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive machine learning project for predicting the Remaining Useful Life (RUL) of NASA turbofan engines using the C-MAPSS dataset. This project implements and compares multiple regression models to achieve accurate RUL predictions for predictive maintenance applications.

## 📊 Project Overview

Predictive maintenance is a proactive approach that uses data-driven techniques to predict equipment failure before it occurs. This project focuses on estimating the RUL of turbofan engines by analyzing sensor data and operational settings, enabling cost-effective maintenance scheduling.

### Key Highlights
- **Best Model Performance**: XGBoost with **RMSE: 19.85** and **R²: 0.77**
- **Multiple ML Models**: Comparison of 7 different regression algorithms
- **Complete Pipeline**: From data loading to model evaluation
- **NASA C-MAPSS Dataset**: Industry-standard benchmark dataset (FD001)

## 🎯 Results Summary

| Model | Test RMSE | Test R² Score |
|-------|-----------|---------------|
| Linear Regression | 22.91 | 0.70 |
| SVR | 21.83 | 0.72 |
| Decision Tree | 17.51 | 0.74 |
| Random Forest | 20.73 | 0.75 |
| Hypertuned Random Forest | 20.08 | 0.77 |
| Ridge Regression | 20.08 | 0.77 |
| **XGBoost** | **19.85** | **0.77** |

![Model Comparison](https://www.dropbox.com/scl/fi/e2vja0cad4ddnihhceoer/Models-Comparision.png?rlkey=fdysqypkou941bcv9dczlkz07&st=zd2hnk7l&raw=1)

## 🗂️ Repository Structure

```
Predictive-Maintenance-Using-Sensor-Data/
├── CMaps/                     # Nasa CMapss Dataset
├── Models/                    # Saved trained model artifacts
├── RUL_ML.ipynb               # Main Jupyter notebook (complete analysis)
├── train_FD001.txt            # Training data with run-to-failure cycles
├── test_FD001.txt             # Test sensor data
├── RUL_FD001.txt              # Ground truth RUL values for test set
├── ML.docx                    # Detailed project documentation
└── README.md                  # This file
```

## 📁 Dataset Information

**Dataset**: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) - FD001

**Characteristics**:
- **Training trajectories**: 100 engines
- **Test trajectories**: 100 engines
- **Fault modes**: 1 (least complex scenario)
- **Operational settings**: 3
- **Sensor readings**: 21

**Data Structure**:
Each row contains:
1. Engine unit number
2. Time (in cycles)
3. Three operational settings
4. 21 sensor measurements (temperature, pressure, vibration, etc.)

**Dataset Source**: [Kaggle - NASA C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps/data)

## 🔄 Machine Learning Workflow

![Machine Learning Workflow for Predictive Maintenance (RUL Prediction)](https://www.dropbox.com/scl/fi/np5omkgic1rau5bulobhm/Screenshot-2025-11-04-015543.png?rlkey=qvme86u6dji8lal5xdouwnfw4&st=cjag42dp&raw=1)

## 🛠️ Methodology

### 1. **Data Loading**
   - Import training and test datasets
   - Load ground truth RUL values

### 2. **Data Analysis**
   - Exploratory Data Analysis (EDA)
   - Statistical summaries
   - Missing value detection
   - Feature distribution analysis

### 3. **Data Preprocessing**
   - Feature engineering
   - RUL calculation for training data
   - Data normalization/scaling
   - Train-test split management

### 4. **Data Visualization**
   - Sensor reading patterns over time
   - Correlation heatmaps
   - Feature importance plots
   - Engine degradation trends

### 5. **Model Building**
   Implemented and compared 7 regression models:
   - Linear Regression
   - Support Vector Regression (SVR)
   - Decision Tree Regressor
   - Random Forest Regressor
   - Hypertuned Random Forest
   - Ridge Regression (L2 Regularization)
   - XGBoost Regressor

### 6. **Model Evaluation**
   - RMSE (Root Mean Square Error)
   - R² Score (Coefficient of Determination)
   - Cross-validation
   - Hyperparameter tuning

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/anshul755/Predictive-Maintenance-Using-Sensor-Data.git
cd Predictive-Maintenance-Using-Sensor-Data
```

2. **Install required packages**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost jupyter
```

### Required Libraries

```python
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- xgboost
- jupyter
```

### Usage

1. **Open the Jupyter Notebook**
```bash
jupyter notebook RUL_ML.ipynb
```

2. **Run the cells sequentially** to:
   - Load and explore the data
   - Preprocess and visualize patterns
   - Train multiple ML models
   - Compare model performances
   - Generate predictions

3. **Access saved models** from the `Models/` directory for inference

## 📈 Key Findings

- **XGBoost** emerged as the best-performing model with the lowest RMSE (19.85) and highest R² (0.77)
- **Decision Tree** showed surprisingly good performance with the lowest RMSE among non-ensemble methods (17.51)
- **Ensemble methods** (Random Forest, XGBoost) generally outperformed linear models
- **Hyperparameter tuning** improved model performance significantly
- Sensor data patterns show clear degradation trends over engine lifecycle

## 🎓 Contributers
- Anshul Patel [Github](https://github.com/anshul755)
- Jainil Patel [Github](https://github.com/JainilPatel2502)


## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/anshul755/Predictive-Maintenance-Using-Sensor-Data/issues).

## 🙏 Acknowledgments

- NASA for providing the C-MAPSS dataset
- Kaggle community for dataset hosting and discussions