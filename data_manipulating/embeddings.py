from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import torch
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from tqdm import tqdm

from data_manipulating.preprocessing import EnhancedPreprocessor, Preprocessor


class BERTEmbedder(BaseEstimator, TransformerMixin):
    """
    A custom transformer that generates BERT embeddings for text data.
    Compatible with scikit-learn pipeline.
    """

    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling of token embeddings, weighted by attention mask.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for text in tqdm(X, desc="Generating BERT embeddings"):
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                output = self.model(**encoded)
                embedding = self._mean_pooling(output, encoded['attention_mask'])
                embeddings.append(embedding.cpu().numpy()[0])

        return np.array(embeddings)


class SBERTEmbedder(BaseEstimator, TransformerMixin):
    """
    A custom transformer that generates Sentence-BERT embeddings.
    Optimized for sentence-level semantic similarity.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(X, show_progress_bar=True)


class Doc2VecEmbedder(BaseEstimator, TransformerMixin):
    """
    A custom transformer that learns and generates Doc2Vec embeddings.
    Captures document-level semantic relationships.
    """

    def __init__(self, vector_size=300, min_count=2, epochs=20):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def fit(self, X, y=None):
        # Tokenize and tag documents
        tagged_data = [
            TaggedDocument(nltk.word_tokenize(text.lower()), [i])
            for i, text in enumerate(X)
        ]

        # Train Doc2Vec model
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=4
        )
        self.model.build_vocab(tagged_data)
        self.model.train(
            tagged_data,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs
        )
        return self

    def transform(self, X):
        return np.array([
            self.model.infer_vector(nltk.word_tokenize(text.lower()))
            for text in tqdm(X, desc="Generating Doc2Vec embeddings")
        ])


# Example usage in your existing code:
def create_embeddings_pipeline(embedding_type='bert', **kwargs):
    """
    Creates a pipeline with the specified embedding type.

    Args:
        embedding_type: One of 'bert', 'sbert', or 'doc2vec'
        **kwargs: Additional arguments for the embedder
    """
    if embedding_type == 'bert':
        return BERTEmbedder(**kwargs)
    elif embedding_type == 'sbert':
        return SBERTEmbedder(**kwargs)
    elif embedding_type == 'doc2vec':
        return Doc2VecEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


# Modified data preparation code:
from sklearn.pipeline import Pipeline


def prepare_data_with_embeddings(data, embedding_type='bert', use_enhanced_preprocessing=True, **kwargs):
    """
    Prepares the data using the specified embedding approach with preprocessing.

    Args:
        data: DataFrame containing the text data
        embedding_type: Type of embeddings to use
        use_enhanced_preprocessing: Whether to use EnhancedPreprocessor instead of base Preprocessor
        **kwargs: Additional arguments for the embedder
    """
    # Initialize appropriate preprocessor
    preprocessor = EnhancedPreprocessor() if use_enhanced_preprocessing else Preprocessor()
    # Create embedding pipeline
    embedder = create_embeddings_pipeline(embedding_type, **kwargs)

    # Create preprocessing wrapper for scikit-learn compatibility
    class PreprocessorWrapper(BaseEstimator, TransformerMixin):
        def __init__(self, preprocessor):
            self.preprocessor = preprocessor

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return self.preprocessor.preprocess_corpus(X)

    # Create full pipeline with preprocessing
    pipeline = Pipeline([
        ('preprocessor', PreprocessorWrapper(preprocessor)),
        ('embedder', embedder)
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data["PreText"],
        data["Class"],
        test_size=0.1,
        random_state=42
    )

    # Fit and transform training data
    X_train_embedded = pipeline.fit_transform(X_train)
    X_test_embedded = pipeline.transform(X_test)

    return X_train_embedded, X_test_embedded, y_train, y_test, pipeline
