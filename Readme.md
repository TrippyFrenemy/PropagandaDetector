# Text Classification Project

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Files Description](#files-description)

## Overview
This project is a text classification application built using FastAPI for the web interface and various machine learning models for text classification tasks. It preprocesses textual data and trains models to classify text into different categories, such as propaganda or non-propaganda.

## Setup
1. **Clone the repository**:
    ```bash
    git clone https://github.com/TrippyFrenemy/PropagandaDetector
    cd PropagandaDetector
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the FastAPI server**:
    ```bash
    uvicorn main:app --reload
    ```

## Usage
1. **Access the web interface**:
   Open your web browser and navigate to `http://127.0.0.1:8000/`.

2. **Submit text for classification**:
   Enter the text you want to classify in the form and click submit. The result will show whether the text is classified as propaganda or non-propaganda.

## Files Description

### FastAPI Files
- `main.py`: Entry point for the FastAPI application. It sets up the routes and handles the text submission and prediction.
- `config.py`: Configuration file for setting up paths and other configuration variables.
- `templates/`: Contains HTML templates for the web interface.

### Classification Files
- `classification.py`: Handles the main classification logic including training and saving models.
- `data_manipulating/preprocessing.py`: Contains functions for text preprocessing including tokenization, lemmatization, and stop words removal.
- `data_manipulating/model.py`: Functions to save and load machine learning models and vectorizers.
- `data_manipulating/over_sampling.py`: Handles over-sampling techniques to balance the dataset.
- `models_back/`: Directory containing pre-trained models and vectorizers.
- `worker.py`: Script to test the trained models on a given set of sentences.
- `word2vec.py`: Script for working with Word2Vec embeddings. (NOT WORKING, in test)
- `utils/`: Utility scripts for various tasks such as converting text to CSV, drawing reports, and using Google Translate.

### Other Files
- `.env`: Environment variables file.
- `.gitignore`: Specifies which files and directories to ignore in version control.
- `requirements.txt`: Lists the Python packages required to run the project.

## Preprocessing and Classification Example
### Preprocessing
The `preprocessing.py` script includes the following key function:
```python
def preprocess_text(text, lemma):
    # Preprocess the text
    # ...
    return text

def preprocess_corpus(corpus, lemma=True):
    processed_corpus = [preprocess_text(text, lemma) for text in tqdm(corpus)]
    return processed_corpus
```

### Classification
The `classification.py` script includes the following key steps:
1. Load and preprocess the data.
2. Train various classifiers (Logistic Regression, Random Forest, Decision Tree, Naive Bayes, KNN).
3. Save the trained models and vectorizer.
4. Generate and print classification reports.

Example usage:
```python
# Load data
data = pd.read_csv('your_folder/your_csv_dataset', sep=";", encoding="utf-8")
data["PreText"] = preprocess_corpus(data["Text"])

# Split data to train and test datasets 
X_train, X_test, y_train, y_test = train_test_split(data["PreText"], data["Class"], test_size=0.1, random_state=20)

# Train and save vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
save_vectorizer(vectorizer, "your_path/file_name_without_extension")

# Train and save models
logreg = LogisticRegression(random_state=20, max_iter=8000, solver="saga", penalty="elasticnet", l1_ratio=0.5)
logreg.fit(X_train_tfidf, y_train)
save_model(logreg, "your_path/file_name_without_extension")
```

## Conclusion
This project provides a comprehensive pipeline for text classification tasks, from data preprocessing to model training and web interface deployment. It leverages the power of FastAPI for serving the models and various machine learning techniques for classification.