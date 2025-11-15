# 📰 Fake News Classification Using Machine Learning

This project focuses on building a machine learning model to classify news articles as **Real** or **Fake** using Natural Language Processing (NLP) and traditional machine learning techniques. The dataset contains two separate files: one with real news and one with fake news. After labeling, merging, cleaning, and vectorizing the text, a classification model is trained to detect misinformation effectively.

---

## 📌 Project Overview

The main objective of this project is to detect fake news articles based on their textual content.  
The workflow includes:

- Loading and merging datasets  
- Adding labels (1 = Real, 0 = Fake)  
- Cleaning and preprocessing raw text  
- Converting text into numerical features using **TF-IDF**  
- Training a machine learning classifier  
- Evaluating performance using accuracy and classification metrics  

This project is ideal for learning **text classification**, **NLP**, and **model evaluation**.

---

## 🗂 Dataset Description

The dataset contains two CSV files:

- **True.csv** — Real news articles  
- **Fake.csv** — Fake news articles  

Each file includes:

- `title`  
- `text`  
- `subject`  
- `date`

After loading, a new column `label` is added:

- Real news → `1`  
- Fake news → `0`

The two datasets are then concatenated and shuffled.

---

## 🧹 Data Preprocessing

Several preprocessing steps were applied to clean the text:

- Removing URLs  
- Removing punctuation & special characters  
- Converting all text to lowercase  
- Basic regex cleaning  
- Shuffling and resetting index  

The cleaned text is transformed using **TfidfVectorizer** with up to 5000 features.

---

## 🤖 Model Training

The dataset is split into **80% training** and **20% testing**.

A **Logistic Regression** classifier is used because it performs well on high-dimensional sparse text data.

Training pipeline:

1. Fit TF-IDF vectorizer on training data  
2. Train Logistic Regression model  
3. Predict on the test set  
4. Evaluate performance  

---

## 📊 Evaluation Metrics

The model is evaluated using:

- **Accuracy Score**  
- **Precision, Recall, F1-Score**  
- **Classification Report**  

These metrics help assess how well the model distinguishes fake news from real news.

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- TfidfVectorizer  
- Logistic Regression  
- Jupyter Notebook  

---

## 🚀 Future Improvements

Potential enhancements:

- Leveraging deep learning models (LSTM, GRU, Bi-LSTM)  
- Using transformer models such as **BERT**  
- Implementing lemmatization & stemming  
- Improving feature extraction with Word2Vec or GloVe  
- Deploying the model as a web application (Flask, FastAPI, Streamlit)  

---

## 📁 Project Structure

```
│── Fake.csv
│── True.csv
│── notebook.ipynb
│── README.md
└── model.pkl (optional)
```

---

## ✨ Summary

This project demonstrates a full pipeline for detecting fake news using NLP and classic machine learning.  
It covers everything from data loading to preprocessing, model training, evaluation, and potential improvements.

---

## 📬 Contact

If you'd like to discuss the project or collaborate, feel free to reach out!

