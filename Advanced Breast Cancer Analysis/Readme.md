# Breast Cancer Analysis: An Advanced Data Exploration and Machine Learning Approach

## Project Overview

This project presents an in-depth analysis of breast cancer data using advanced statistical methods and machine learning models. The primary objective is to explore the clinical and molecular features associated with breast cancer, identify significant patterns, and develop predictive models to assist in early detection and prognosis.

The dataset used in this analysis, `METABRIC_RNA_Mutation.csv`, contains extensive clinical and genetic data on breast cancer patients, making it a rich source for comprehensive exploration and modeling.

## Research Publication

The findings from this analysis have been published in a peer-reviewed journal. You can read the full research paper here:

**[Read the Published Paper](<https://biomedres.us/pdfs/BJSTR.MS.ID.009130.pdf>)**

## Key Features & Analysis

### 1. Data Visualization
- **Clinical Data Exploration**: Visualized the distribution of key clinical attributes such as tumor size, lymph node status, and patient demographics.
- **Correlation Analysis**: Explored the relationships between various clinical and genetic features to identify potential predictors of breast cancer outcomes.

### 2. Advanced Statistical Analysis
- **Normality Tests**: Performed statistical tests like the D'Agostino-Pearson test to assess the normality of the data distributions.
- **Correlation Matrix**: Generated and visualized the correlation matrix to identify the most influential features.

### 3. Machine Learning Modeling
- **Model Selection**: Implemented and compared several classifiers including Logistic Regression, Random Forest, Support Vector Machine (SVM), and XGBoost.
- **Hyperparameter Tuning**: Applied GridSearchCV and cross-validation techniques to optimize model performance.
- **Performance Evaluation**: Used metrics such as accuracy, F1 score, and ROC-AUC to evaluate model effectiveness.

### 4. Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Applied PCA to reduce the dimensionality of the data, capturing the most significant variance in the dataset.

### 5. Anomaly Detection
- **Isolation Forest**: Implemented Isolation Forest for detecting and analyzing outliers, which may correspond to rare or unique clinical cases.

## Technologies & Libraries Used

- **Programming Language**: Python
- **Data Manipulation**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, yellowbrick, venn3
- **Machine Learning**: scikit-learn, XGBoost, kmapper
- **Statistical Analysis**: statsmodels

## Results & Findings

- **Significant Correlations**: Identified key clinical features that correlate strongly with patient outcomes, providing insights into the factors most associated with breast cancer prognosis.
- **Model Performance**: The Random Forest classifier achieved the highest accuracy, making it the most effective model for this dataset. The ROC-AUC scores confirmed its robustness.
- **PCA Insights**: The PCA revealed that a small number of components could explain a significant portion of the variance, allowing for a more streamlined analysis.

## How to Run the Notebook

### Prerequisites

Ensure that you have Python installed, along with the following libraries:

```bash
pip install numpy pandas scipy matplotlib seaborn yellowbrick scikit-learn xgboost
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Open the notebook:
   ```bash
   jupyter notebook breast_cancer.ipynb
   ```
3. Execute the cells in sequence to perform the analysis.

## Conclusion

This analysis provides a comprehensive exploration of breast cancer data, combining visualization, statistical methods, and machine learning to yield valuable insights. The findings underscore the importance of certain clinical features in predicting patient outcomes and demonstrate the effectiveness of machine learning models in this domain. Future work could expand on these results by incorporating additional data or exploring other advanced modeling techniques.

