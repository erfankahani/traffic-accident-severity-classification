# Traffic Accident Severity Classification with Machine Learning

**Data Mining course project · Group 23 · Instructor: Dr. Nadi · 2025–2026**

This project predicts the **severity of traffic accidents** (4 classes) from a real-world accident dataset. It runs a complete data mining pipeline: exploratory data analysis, cleaning, encoding, feature engineering, feature selection, model training, hyperparameter tuning and evaluation. Four models are compared: an Artificial Neural Network, a Decision Tree, a Random Forest and XGBoost.

**Best model: XGBoost, with 86.49% test accuracy and a Macro F1 score of 0.577.**

---

## Dataset

- `group_23_train.csv`: 100,000 accident records with 34 anonymized features (`f1`–`f34`) and the target `y` (severity, classes 1–4).
- The records come from the [US Accidents dataset on Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).
- The features mix several types: numerical (latitude, longitude, distance, temperature, humidity, pressure, visibility, wind speed), categorical (city, state, weather, wind direction, day/night), boolean (road conditions), datetime and free text (accident description).
- The target is **highly imbalanced**. Class 2 makes up about 79% of the records, and classes 1 and 4 are rare.

Project rules: use at most **25 features** in the final model, use **no external data**, and compare **at least 3 models**.

---

## Methodology

### 1. Exploratory Data Analysis
- Feature types, the target distribution, and a Pearson correlation heatmap for the numerical features.
- Histograms and boxplots for numerical features, bar charts for categorical features, and count plots for boolean features.
- A missing-value analysis (`f4` is 43.9% missing, `f20` 28.5% and `f15` 25.9%) and IQR outlier detection.

### 2. Cleaning and Preprocessing
- Duplicate records were removed.
- Missing values: numerical features were filled with the **median**, categorical features with `"Unknown"`, boolean features with the **most frequent value**. Datetime features were split into hour, month and day of week.
- Outliers were detected but **kept**. Because the features are anonymized, there was no way to tell a valid extreme value from an error.
- Encoding: **one-hot** for low-cardinality categories, **frequency encoding** for high-cardinality categories. TF-IDF was tested on the text feature `f6`, but `f6` was dropped because it would have added too many features for the 25-feature limit.
- Split: **80% train / 10% validation / 10% test**, stratified by the target.
- Scaling: standardization (mean 0, standard deviation 1), applied after the split to avoid data leakage.

### 3. Feature Selection
- **Filter methods:** Mutual Information and Fisher Score. Each feature was ranked by its average position in the two rankings, and the top 25 features were kept. These features are used by the ANN.
- **Embedded methods:** feature importance from the tree-based models (Decision Tree, Random Forest, XGBoost), trained on the **unscaled** data so the importances stay interpretable.
- Wrapper methods and PCA were considered but not used. Wrapper methods would have needed about 675 training runs, and PCA would have replaced the original features with components that are hard to interpret.

### 4. Models and Class Imbalance
All models were tuned on the **validation Macro F1 score**, because accuracy is misleading on imbalanced data. Every model also uses class weighting.

| Model | Feature set | Selected configuration |
|---|---|---|
| ANN (MLP) | Filter (scaled) | hidden layers (128, 64, 32), ReLU, Adam, learning rate 0.001, batch size 32, early stopping, manual class weights {1: 10, 2: 0.5, 3: 2, 4: 5} |
| Decision Tree | Embedded (unscaled) | criterion entropy, max_depth 20, class_weight balanced |
| Random Forest | Embedded (unscaled) | 100 trees, criterion entropy, min_samples_split 5, min_samples_leaf 5, max_features sqrt, class_weight balanced |
| XGBoost | Embedded (unscaled) | 600 trees, max_depth 5, learning rate 0.05, min_child_weight 3, subsample 0.8, colsample_bytree 0.8, gamma 0.1, sample weighting |

---

## Results (test set, 9,998 samples)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| ANN | 75.89% | 0.473 | 0.78 |
| Decision Tree | 77.95% | 0.487 | 0.80 |
| Random Forest | 84.57% | 0.546 | 0.85 |
| **XGBoost** | **86.49%** | **0.577** | **0.87** |

XGBoost's F1 score per class: class 1: 0.32 · class 2: 0.92 · class 3: 0.74 · class 4: 0.33.

The most important features across the tree-based models were **f5** (distance), **f1_Source2**, **f4**, **f7_freq**, **f10_freq** and **f2_month**.

**Takeaways**
- Boosting (XGBoost) gave the best balance between overall accuracy and performance on the minority classes.
- Fully automatic inverse-frequency class weights over-penalized the majority class. Manually adjusted weights produced a better Macro F1.
- Using two preprocessing outputs worked best: scaled data for the ANN and unscaled data for the tree-based models.

---

## Repository Structure

```
├── DM_PROJECT_NOTEBOOK.ipynb   # EDA, preprocessing, feature engineering, feature selection, data export
├── Filter_methods_ANN.ipynb    # ANN trained on the filter-selected features
├── Embedded_methods.ipynb      # Decision Tree, Random Forest and XGBoost
├── group_23_train.csv          # Dataset
├── Decision_Tree_Final.png     # Visualization of the final decision tree
├── requirements.txt
├── REPOTRS/
│   ├── DM_PROJECT_REPORT.pdf   # Full project report (also as .docx)
reference
```

---

## How to Run

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run **`DM_PROJECT_NOTEBOOK.ipynb`** first. It reads `group_23_train.csv` and creates two folders:
   - `Data Export Embedded Methods/`: unscaled data for the tree-based models
   - `Data Export Filter Method/`: scaled data with the filter-selected features, for the ANN
3. Then run **`Filter_methods_ANN.ipynb`** and **`Embedded_methods.ipynb`** in any order.

The generated `Data Export*` folders are not in the repository because they are large and can be recreated with step 2.

---

## Team

| Member | Contributions |
|---|---|
| **Seyed Erfan Kahani** | Data cleaning and preprocessing, feature engineering, Decision Tree / Random Forest / XGBoost models, hyperparameter tuning, evaluation and the related report sections |
| **Maryam Safaei** | Introduction and abstract, exploratory data analysis,Phase 2 progress report |

Both members reviewed the final results and prepared the final report together.

See [`REPOTRS/DM_PROJECT_REPORT.pdf`](REPOTRS/DM_PROJECT_REPORT.pdf) for the full analysis, figures and discussion.
