# Propaganda Detection System

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Classification Approaches](#classification-approaches)
- [Technical Implementation](#technical-implementation)
- [Files Description](#files-description)

## Overview
This project implements a sophisticated propaganda detection system using multiple classification approaches and deep learning techniques. It provides both basic classification capabilities and advanced propaganda technique identification, built using FastAPI for the web interface and various machine learning architectures for text analysis tasks.

## Project Structure
```
.
├── data_manipulating/
│   ├── embeddings.py          # Text embedding generation
│   ├── feature_extractor.py   # Feature extraction utilities
│   ├── manipulate_models.py   # Model handling functions
│   ├── over_sampling.py       # Data balancing techniques
│   └── preprocessing.py       # Text preprocessing pipeline
├── pipelines/
│   ├── cascade_classification.py    # Cascade classification model
│   ├── classification.py           # Base classification pipeline
│   └── improved_cascade.py         # Enhanced cascade model
├── utils/
│   ├── add_data_to_csv.py    # Dataset manipulation
│   ├── combine_csv.py        # Dataset combining utilities
│   ├── draw_report.py        # Visualization tools
│   └── google_translate.py    # Translation services
├── templates/                # HTML templates
├── main.py                  # FastAPI application
├── config.py               # Configuration settings
└── requirements.txt        # Project dependencies
```

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

4. **Set up environment variables**:
Create a `.env` file with the following configuration:
```
MODEL_PATH=models
CASCADE_PATH=cpm_v5
IMPROVED_CASCADE_PATH=icpm_v2
FOREST_PATH=models/forest_propaganda_detector_v2
TFIDF_PATH=models/tfidf_propaganda_detector_v2
```

5. **Run the FastAPI server**:
```bash
uvicorn main:app --reload
```

## Usage
The system provides multiple endpoints for propaganda detection:

1. **Basic Classification** (`/`):
   - Detects propaganda using traditional machine learning approaches
   - Returns binary classification with confidence scores
   - Highlights influential words in the text

2. **Improved Classification** (`/improved_classification`):
   - Uses cascade architecture for detailed analysis
   - Identifies specific propaganda techniques
   - Provides confidence levels for each detection

3. **Data Addition** (`/add`):
   - Allows adding new labeled data to the training set
   - Supports both text input and file upload
   - Automatically handles data validation and preprocessing

## Classification Approaches

### 1. Basic Classification Pipeline
The traditional approach using TF-IDF features and ensemble methods:
- Logistic Regression with elastic net regularization
- Random Forest with optimized parameters
- Decision Trees and KNN as complementary models
- Feature importance analysis and visualization

### 2. Cascade Classification Pipeline
A sophisticated two-stage model:
```python
class CascadePropagandaPipeline:
    def __init__(self, model_path, model_name, ...):
        self.binary_model = BinaryPropagandaModel(...)
        self.technique_model = TechniquePropagandaModel(...)
```
- First stage: Binary propaganda detection
- Second stage: Specific technique identification
- Attention mechanisms for improved accuracy
- Multi-head architecture for complex pattern recognition

### 3. Improved Cascade Pipeline
Enhanced version with additional features:
- Advanced feature extraction using BERT embeddings
- Hierarchical attention mechanisms
- Fragment-aware classification
- Improved confidence scoring system

## Technical Implementation

### Embedding Generation
The system supports multiple embedding approaches:
```python
class BERTEmbedder:
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
```

### Feature Extraction
Advanced feature extraction includes:
- Linguistic features (sentence structure, complexity)
- Emotional markers (sentiment, subjectivity)
- Rhetorical devices (repetition, alliteration)
- Statistical characteristics (vocabulary richness)

### Model Architecture
The cascade model uses a sophisticated neural architecture:
- CNN layers for local feature extraction
- BiLSTM for sequence understanding
- Multi-head attention for context awareness
- Hierarchical classification for technique identification

## Files Description

### Core Classification Files
- `embeddings.py`: Implements various text embedding approaches including BERT, Sentence-BERT, and Doc2Vec
- `feature_extractor.py`: Comprehensive feature extraction pipeline with linguistic and statistical analysis
- `cascade_classification.py`: Implementation of the cascade classification model
- `improved_cascade.py`: Enhanced version of the cascade model with additional capabilities

### Utility Modules
- `manipulate_models.py`: Functions for model serialization, loading, and management
- `preprocessing.py`: Text preprocessing pipeline including cleaning, tokenization, and normalization
- `over_sampling.py`: Implementation of various data balancing techniques
- `google_translate.py`: Multi-language support through translation services

### Data Management
- `combine_csv.py`: Tools for dataset manipulation and combination
- `add_data_to_csv.py`: Functions for adding new data to the training set
- `draw_report.py`: Visualization and reporting utilities

### Web Interface
- `main.py`: FastAPI application implementing the web interface
- `templates/`: HTML templates for the web interface
- `config.py`: Configuration management and environment variables

## Model Training and Evaluation
To train the models:
```python
pipeline = CascadePropagandaPipeline(
    model_path="models",
    model_name="cpm_v5",
    batch_size=32,
    num_epochs=30,
    learning_rate=2e-5,
    warmup_steps=1000,
    max_length=512,
    class_weights=True
)

metrics = pipeline.train_and_evaluate("path_to_dataset.csv")
```
