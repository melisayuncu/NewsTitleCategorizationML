# News Title Classification using Machine Learning

This project focuses on classifying news headlines into categories using various machine learning algorithms. The study includes handling class imbalance, feature extraction, feature selection, and model comparison.

---

## Project Overview

News classification is an important Natural Language Processing (NLP) task. In this project, news headlines are automatically categorized using supervised machine learning techniques.

The pipeline includes:
- Data preprocessing
- Handling class imbalance
- Feature extraction with TF-IDF
- Feature selection using Chi-square
- Model training and evaluation
- Performance comparison

---

## Dataset

The dataset consists of news headlines belonging to different categories.

The dataset is imbalanced:
- Politics & Government: 9930 samples  
- Education: 481 samples  

To address this issue, **class weights** are used in the models.

---

## Methodology

### Data Split
- Train/Test Split: 80% / 20%

---

### Feature Extraction
TF-IDF vectorization is applied:

```python
TfidfVectorizer(max_features=10000, stop_words='english')
```

## Models Used
- Logistic Regression
- Naive Bayes
- Linear SVC
- Random Forest
- K-Nearest Neighbors (with normalization)
- Decision Tree
- XGBoost

Hyperparameter optimization was applied to all algorithms before evaluation.

## Key Insights
- Class imbalance significantly affects model performance
- TF-IDF + Chi-square improves efficiency and accuracy
- Logistic Regression performed well after optimization
- Feature selection reduces dimensionality without major performance loss

## Future Work
- Deep learning models (LSTM, BERT)
- More balanced dataset
- Advanced text preprocessing techniques

## Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn
