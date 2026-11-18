# Heart Failure Prediction

Trains five scikit-learn classifiers on clinical data to predict heart disease, then writes each model's predictions for an unlabeled test set side by side so they can be compared.

Built for the DSAI club induction task (2022).

## What it does

- **Data:** 700 labeled training rows and 218 unlabeled test rows. There are 11 clinical features, including age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, exercise-induced angina, ST depression (Oldpeak) and ST slope. The target is `HeartDisease` (0/1).
- **Preprocessing:** one-hot encodes the 5 categorical features with a `ColumnTransformer` and standardizes all features with `StandardScaler`.
- **Models:** k-nearest neighbors (k=5), logistic regression, random forest (10 trees, entropy), decision tree (entropy) and an RBF-kernel SVM.
- **Output:** `Heart Failure Predictions.csv`, with one column of predictions per model for the 218 test patients.

**Results:** the test labels weren't provided, and the script doesn't score the models on held-out data, so no accuracy is reported.

## Tech stack

Python · pandas · NumPy · scikit-learn

## How to run

```sh
pip install numpy pandas scikit-learn matplotlib
```

The script reads and writes files using hardcoded `D:/` paths. Change them to the repo folder first (the output path is `'D:/ Heart Failure Predictions.csv'`, with a space after the slash). For example:

```sh
sed -i.bak 's|D:/ |./|; s|D:/|./|g' Heart_Failure_Model.py
python3 Heart_Failure_Model.py
```

This writes `Heart Failure Predictions.csv` to the current folder.
