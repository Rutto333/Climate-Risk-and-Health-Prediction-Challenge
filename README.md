# Climate-Linked Mortality Prediction

## Overview

This project explores whether a recorded death belongs to a
**climate-sensitive category** using supervised machine learning. It
combines health, demographic, geographic, and climate-related features
to estimate the probability that a death is climate-sensitive.

The project is designed for a multi-metric machine learning challenge in
which the final score combines **F1-Score** and **ROC-AUC**. The goal is
to build a model that identifies climate-sensitive cases while also
ranking positive cases above negative cases effectively.

> **Important:** Predictive patterns do not establish that climate
> conditions caused an individual death. Findings should be interpreted
> as associations in the available data, with attention to data quality,
> bias, and uncertainty.

## Problem Statement

Given a set of records containing health, demographic, geographic, and
environmental information, predict:

1.  A binary class indicating whether the death is climate-sensitive.
2.  A probability representing the model's estimated likelihood that the
    death is climate-sensitive.

## Evaluation Metric

The leaderboard score is a weighted combination of two metrics:

  ------------------------------------------------------------------------
  Metric                                      Weight What it measures
  --------------------- ---------------------------- ---------------------
  F1-Score                                       60% Balance between
                                                     precision and recall
                                                     for the positive
                                                     class

  ROC-AUC                                        40% How well predicted
                                                     probabilities rank
                                                     positive cases above
                                                     negative cases
  ------------------------------------------------------------------------

The combined score is:

\[ `\text{Final Score}`{=tex} = 0.60
`\times `{=tex}`\text{F1-Score}`{=tex} + 0.40
`\times `{=tex}`\text{ROC-AUC}`{=tex} \]

### Prediction and submission rules

-   `TargetF1` must contain binary predictions generated using the
    **default probability threshold of 0.5**.
-   Do **not** tune or apply a different threshold for the binary
    predictions.
-   `TargetRAUC` must contain the predicted probability for the
    climate-sensitive class (class `1`).
-   Include the test record identifier in the `ID` column.
-   Preserve the required column names and submission format.

Example:

  ID            TargetF1   TargetRAUC
  ----------- ---------- ------------
  ID_XYZ4AL            0         0.45
  ID_XTLAUX            0         0.32

## Workflow

1.  **Data understanding** --- inspect the training and test data,
    target distribution, data types, missing values, and duplicate
    records.
2.  **Exploratory data analysis** --- examine mortality labels alongside
    demographic, geographic, and climate variables.
3.  **Data preprocessing** --- handle missing values, encode categorical
    features, and scale numerical features when required by the model.
4.  **Feature engineering** --- where supported by the available
    columns, derive useful climate features such as temperature range,
    rainfall intensity, recent-to-long-term rainfall ratios, and
    temporal changes.
5.  **Model training** --- train and compare suitable supervised
    classifiers.
6.  **Validation** --- use a reproducible validation strategy and assess
    both F1-Score and ROC-AUC.
7.  **Probability generation** --- generate class-1 probabilities for
    the test dataset.
8.  **Submission preparation** --- create the required two prediction
    columns and export them with the record ID.

## Potential Models

Models worth comparing include:

-   Logistic Regression as a baseline.
-   Random Forest.
-   XGBoost.
-   CatBoost.
-   Support Vector Machine (SVM), with preprocessing where needed.

Model choice should be based on validation performance, data
characteristics, and practical constraints. Use pipelines where possible
to keep preprocessing consistent between training and inference.

## Example Submission Code

The following example assumes that `test_df` contains the identifier
column `ID` and that `model` is already trained. Replace `ID` and the
feature-selection logic with the actual column names used in the
dataset.

``` python
import pandas as pd

# Keep the test identifiers
submission = pd.DataFrame({
    "ID": test_df["ID"],
    "TargetRAUC": model.predict_proba(X_test)[:, 1],
})

# Required default threshold: 0.5
submission["TargetF1"] = (
    submission["TargetRAUC"] >= 0.5
).astype(int)

# Required column order
submission = submission[["ID", "TargetF1", "TargetRAUC"]]

submission.to_csv("submission.csv", index=False)
```

Confirm that the model supports `predict_proba`. If it does not, use an
appropriate probability-calibration approach or a model that provides
probabilities. Fit any calibration method using training/validation data
only; never fit it using the test labels.

## Recommended Validation

-   Use a stratified train/validation split when appropriate for the
    target distribution.
-   Consider cross-validation for a more reliable estimate of
    performance.
-   Calculate F1-Score using binary predictions at the fixed threshold
    of 0.5.
-   Calculate ROC-AUC using predicted probabilities, not binary
    predictions.
-   Check for data leakage, especially from identifiers, post-outcome
    fields, or variables that directly encode the target.
-   If records are grouped by location or time, consider validation
    splits that test generalization to unseen locations or periods.

## Suggested Project Structure

``` text
climate-linked-mortality/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── exploratory-analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   └── train.py
├── submission.csv
├── requirements.txt
└── README.md
```

This is a suggested structure; adapt it to the files and scripts in your
repository.

## Reproducibility

-   Set random seeds where supported.
-   Keep preprocessing and feature engineering consistent across
    training and test data.
-   Record validation scores and model parameters.
-   Avoid using the test set to select features, tune models, or
    calibrate probabilities.
-   Verify the final submission has no missing predictions and that all
    probabilities lie between `0` and `1`.

## Limitations and Responsible Use

Climate-sensitive mortality classification is a data-driven prediction
task, not proof of causation. Results may reflect underreporting,
differences in access to healthcare, geographic coverage, demographic
imbalances, or other unobserved factors. The model should not be used as
the sole basis for individual medical decisions or policy decisions
without further validation and domain expertise.

## Author

**Elkanah Rutto**

Update this section with your preferred contact details and links to the
project repository.
