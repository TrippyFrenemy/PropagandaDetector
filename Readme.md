# Ukrainian Propaganda Detection System

## Table of Contents
- [Overview](#overview)
- [New Features for Ukrainian Language Support](#new-features-for-ukrainian-language-support)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Classification Approaches](#classification-approaches)
- [Technical Implementation](#technical-implementation)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Files Description](#files-description)

## Overview
This project implements a sophisticated propaganda detection system specifically enhanced for Ukrainian language text analysis. It builds upon the previous multilingual models and introduces specialized components designed for Ukrainian text processing and classification. The system uses a cascade approach for propaganda detection, first identifying whether a text contains propaganda, and then determining specific propaganda techniques used.

## New Features for Ukrainian Language Support

### Ukrainian-Specific Processing
- Integration of Ukrainian language stopword removal
- Ukrainian-specific tokenization and lemmatization using spaCy's Ukrainian model
- Sentiment dictionary for Ukrainian text analysis
- Enhanced feature extraction optimized for Ukrainian language constructs
- Support for Cyrillic character encodings and Ukrainian-specific linguistic patterns

### Enhanced Model Architecture
- Ukrainian-specific text embeddings
- Modified CNN filter designs for Ukrainian morphological structures
- Attention mechanism refinements for Ukrainian syntactic patterns
- Improved BiLSTM sequence handling for Ukrainian text
- Advanced feature extraction for Ukrainian rhetorical devices

### Multilingual Capabilities
- Automatic language detection
- Optional translation pipeline for cross-language analysis
- Training on combined Ukrainian and English datasets
- Parallel feature extraction for both languages

## Project Structure
```
.
├── data_manipulating/
│   ├── feature_extractor.py       # Base feature extraction utilities
│   ├── feature_extractor_ua.py    # Ukrainian-specific feature extraction
│   ├── manipulate_models.py       # Model handling functions
│   ├── preprocessing.py           # Base text preprocessing pipeline
│   ├── preprocessing_ua.py        # Ukrainian text preprocessing
│   └── stopwords_ua.txt           # Ukrainian stopwords dictionary
├── pipelines/
│   ├── cascade_classification.py  # Base cascade classification model
│   ├── enhanced_cascade.py        # Enhanced Ukrainian cascade model
│   ├── improved_cascade.py        # Improved cascade architecture
│   └── enhanced_smote.py          # SMOTE-enhanced balancing pipeline
├── utils/
│   ├── add_data_to_csv.py         # Dataset manipulation
│   ├── combine_csv.py             # Dataset combining utilities
│   ├── draw_report.py             # Visualization tools
│   └── translate.py               # Translation services
├── templates/                     # HTML templates for web interface
├── main.py                        # FastAPI application
├── config.py                      # Configuration settings
└── requirements.txt               # Project dependencies
```

## Setup
1. **Clone the repository**:
```bash
git clone https://github.com/TrippyFrenemy/UkrainianPropagandaDetector
cd UkrainianPropagandaDetector
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

4. **Download required language models**:
```bash
python -m spacy download uk_core_news_sm
python -m spacy download en_core_web_sm
```

5. **Set up environment variables**:
Create a `.env` file with the following configuration:
```
MODEL_PATH=models
CASCADE_PATH=cpm_v5
IMPROVED_CASCADE_PATH=icpm_v2
UA_CASCADE_PATH=ecpm_ua_v1
FOREST_PATH=models/forest_propaganda_detector_v2
TFIDF_PATH=models/tfidf_propaganda_detector_v2
```

6. **Run the FastAPI server**:
```bash
uvicorn main:app --reload
```

## Usage
The system provides multiple endpoints for propaganda detection:

1. **Ukrainian Classification** (`/cascade_classification`):
   - Detects propaganda in Ukrainian text
   - Identifies specific propaganda techniques
   - Provides confidence levels for each detection
   - Handles both Ukrainian and English text with automatic language detection

2. **Multilingual Analysis** (`/`):
   - Supports cross-language propaganda detection
   - Automatically translates text when needed
   - Uses traditional machine learning approaches as a baseline

3. **Data Addition** (`/add`):
   - Allows adding new labeled data to the Ukrainian training set
   - Supports both text input and file upload
   - Performs automatic language validation for Ukrainian text

## Classification Approaches

### Enhanced Cascade Classification for Ukrainian
The core innovation is the `EnhancedCascadePropagandaPipeline` which features:
- Ukrainian language preprocessing with custom stopwords and lemmatization
- Enhanced feature extraction for Ukrainian text specifics
- Improved attention mechanism for Ukrainian syntactic patterns
- Balanced handling of Ukrainian-specific propaganda techniques

```python
class EnhancedCascadePropagandaPipeline(CascadePropagandaPipeline):
    def __init__(self, *args, use_extra_features=True, use_ukrainian=False, **kwargs):
        super().__init__(*args, use_extra_features=use_extra_features, use_ukrainian=use_ukrainian, **kwargs)
        self.use_extra_features = use_extra_features
        self.use_ukrainian = use_ukrainian
        if self.use_ukrainian:
            self.nlp = spacy.load("uk_core_news_sm")
```

### Enhanced Binary Propaganda Model
A specialized model for detecting propaganda in Ukrainian text:
```python
class EnhancedBinaryPropagandaModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, lstm_hidden, k_range, use_extra_features=True):
        # Ukrainian-optimized architecture with attention mechanism
        # Feature encoder with emphasis on important Ukrainian-specific features
        # Combined classifier with gradient clipping and regularization
```

## Technical Implementation

### Ukrainian Feature Extraction
The system implements specialized feature extraction for Ukrainian text:
```python
class TextFeatureExtractorUA(EnhancedPreprocessorUA):
    def __init__(self, nlp):
        super().__init__(nlp)
        self.__vowels_ua = set("аеєиіїоуюя")  # Ukrainian vowels for syllable counting
        
    def extract_linguistic_features(self, text):
        # Ukrainian-specific linguistic features extraction
        
    def extract_emotional_features(self, text):
        # Emotional analysis adapted for Ukrainian language
```

### Ukrainian Preprocessing
Enhanced preprocessing pipeline for Ukrainian text:
```python
class PreprocessorUA:
    def __init__(self, nlp):
        # Use Ukrainian stopwords
        with open('../data_manipulating/stopwords_ua.txt', 'r', encoding='utf-8') as f:
            self.__stop_words = set(line.strip().lower() for line in f if line.strip())
        # Load Ukrainian spaCy model
        self.nlp = nlp
```

### Automatic Language Detection and Translation
The system automatically handles mixed-language inputs:
```python
def predict(self, texts: List[str], print_results: bool = True):
    # Store original texts
    text_base = texts
    
    # Optional translation if needed
    # if not self.use_ukrainian:
    #     texts = translate_corpus(texts)
        
    # Process with appropriate model...
```

## Model Architecture

### Enhanced Binary Classification
The Ukrainian propaganda detection model features:
- CNN with different kernel sizes for identifying key Ukrainian-specific patterns
- BiLSTM for sequence analysis optimized for Ukrainian text
- Attention mechanism with 8 heads for contextual understanding
- Feature encoder emphasizing important Ukrainian linguistic characteristics
- Combined classifier with residual connections and layered architecture

### Technique Classification
The system identifies specific propaganda techniques with:
- Fragment-aware classification for Ukrainian text
- Weighted confidence levels for detected techniques
- Hierarchical analysis of text segments
- Threshold-based detection with calibrated confidence

## Training and Evaluation
To train the Ukrainian model:
```python
pipeline = EnhancedCascadePropagandaPipeline(
    model_path="models",
    model_name="ecpm_ua_v1",
    batch_size=32,
    num_epochs_binary=10,
    num_epochs_technique=10,
    learning_rate=2e-5,
    warmup_steps=1000,
    max_length=512,
    class_weights=True,
    binary_k_range=[2, 3, 4, 5, 6, 7],
    technique_k_range=[3, 4, 5],
    dataset_distribution=0.9,
    use_ukrainian=True
)

metrics = pipeline.train_and_evaluate("path_to_ukrainian_dataset.csv")
```

## Files Description

### Core Ukrainian Classification Files
- `preprocessing_ua.py`: Ukrainian text preprocessing with stopwords removal and lemmatization
- `feature_extractor_ua.py`: Feature extraction specifically designed for Ukrainian text
- `enhanced_cascade.py`: Implementation of the enhanced cascade model with Ukrainian support
- `stopwords_ua.txt`: Comprehensive list of Ukrainian stopwords for text preprocessing

### Ukrainian Data Processing
- `add_data_to_csv.py`: Tools for adding new Ukrainian data to the training set
- `translate.py`: Translation services for cross-language analysis and testing
- `sentiment_ua.csv`: Ukrainian sentiment dictionary for emotional text analysis

### Web Interface
- `main.py`: FastAPI application with endpoints for Ukrainian text analysis
- `templates/`: HTML templates for the web interface
- `config.py`: Configuration for Ukrainian model paths and parameters