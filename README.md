# Climate & Health Risk Prediction 🌍

A supervised machine learning project for the [Zindi Climate Risk & Health Prediction Challenge](https://zindi.world/competitions/climate-risk-health-prediction-challenge). The project combines mortality records with demographic, geographic, and climate features to predict whether a recorded death is classified as climate-sensitive.

## 🎯 Objective

Predict `is_climate_sensitive` using demographic and environmental data, leveraging machine learning to identify patterns associated with climate-sensitive mortality.

## 📊 Dataset

- Training data: 3,146 records
- Test data: 1,030 records
- Supplementary climate data: 4,176 records

Climate features are merged using `ID`. Features include age, gender, location, temperature, rainfall, NDVI, and engineered temporal and environmental variables.

## ⚙️ Methodology

1. Data loading and integration
2. Exploratory data analysis
3. Feature engineering and preprocessing
4. Stratified 5-fold cross-validation
5. CatBoost model training and evaluation
6. Test prediction and submission generation

## 🤖 Model

**CatBoostClassifier**
- Iterations: 10,000
- Learning rate: 0.03
- Evaluation metric: ROC-AUC
- Early stopping: 150 rounds
- Cross-validation: 5 folds

## 📈 Validation Results

| Metric | Score |
|---|---:|
| Accuracy | 0.7495 |
| Precision | 0.7671 |
| Recall | 0.8832 |
| F1-Score | 0.8211 |
| ROC-AUC | 0.8191 |
| Weighted Score | 0.8203 |

The challenge score combines F1-Score (60%) and ROC-AUC (40%). The reported F1-Score and weighted score use a threshold of 0.45; the final submission uses the required **0.5 threshold** for binary predictions.

## 🔑 Top Features

1. `year`
2. `age`
3. `age_cat`
4. `ndvi_90d`
5. `tavg_30d`

Feature importance reflects predictive contribution, not causation.

## 🛠️ Technologies

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · CatBoost

## 📁 Repository Structure

```text
climate-related-deaths/
├── climate-related-deaths.ipynb
├── README.md
└── submission.csv
```

## 🚀 Getting Started

1. Clone the repository.
2. Install the required libraries.
3. Download the competition datasets from Zindi.
4. Update dataset paths in the notebook.
5. Run the notebook to train the model and generate predictions.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn catboost
```

## ⚠️ Limitations

The model identifies patterns in available data but does not establish that climate conditions caused individual deaths. Results depend on data quality, coverage, and representativeness.

## 👤 Author

**Elkanah Rutto**
