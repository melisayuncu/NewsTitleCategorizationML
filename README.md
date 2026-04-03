# NewsTitleCategorizationML
News title classification using machine learning models with TF-IDF, feature selection, and class imbalance handling.
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
