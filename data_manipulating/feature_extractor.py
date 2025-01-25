import re
import statistics
from collections import Counter
from typing import Dict, List, Tuple, Any

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
from textblob import TextBlob
from data_manipulating.preprocessing import EnhancedPreprocessor


class TextFeatureExtractor(EnhancedPreprocessor):
    """
    Advanced text feature extractor that builds upon the base Preprocessor class.
    Adds linguistic, emotional, rhetorical, and statistical text features.
    """

    def __init__(self):
        super().__init__()
        # Initialize CMU pronunciation dictionary for syllable counting
        self.prondict = cmudict.dict()

        # Load required NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/cmudict')
        except LookupError:
            nltk.download('cmudict')

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word using CMU pronunciation dictionary."""
        try:
            return len([ph for ph in self.prondict[word.lower()][0] if ph.strip('0123456789')])
        except KeyError:
            # Fallback syllable counting for words not in dictionary
            return len(re.findall('[aeiou]+', word.lower()))

    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features including sentence length, word complexity,
        and readability metrics.
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        # Calculate average sentence length
        sent_lengths = [len(word_tokenize(sent)) for sent in sentences]
        avg_sent_length = statistics.mean(sent_lengths) if sent_lengths else 0
        sent_length_std = statistics.stdev(sent_lengths) if len(sent_lengths) > 1 else 0

        # Calculate word complexity metrics
        word_lengths = [len(word) for word in words]
        syllable_counts = [self._count_syllables(word) for word in words]

        return {
            'avg_sentence_length': avg_sent_length,
            'sentence_length_std': sent_length_std,
            'avg_word_length': statistics.mean(word_lengths) if word_lengths else 0,
            'avg_syllables_per_word': statistics.mean(syllable_counts) if syllable_counts else 0,
            'unique_words_ratio': len(set(words)) / len(words) if words else 0,
        }

    def extract_emotional_features(self, text: str) -> Dict[str, float]:
        """
        Extract emotional characteristics using TextBlob sentiment analysis
        and emotion word detection.
        """
        blob = TextBlob(text)

        # Basic sentiment analysis
        sentiment_scores = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
        }

        # Count emotional punctuation
        sentiment_scores.update({
            'exclamation_ratio': text.count('!') / len(text) if text else 0,
            'question_ratio': text.count('?') / len(text) if text else 0,
            'ellipsis_ratio': text.count('...') / len(text) if text else 0,
        })

        return sentiment_scores

    def extract_rhetorical_features(self, text: str) -> Dict[str, float]:
        """
        Detect rhetorical devices like repetition, alliteration, and metaphorical language.
        """
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)

        # Analyze word repetition
        word_counts = Counter(words)
        repeated_words = {word: count for word, count in word_counts.items() if count > 1}

        # Detect alliteration (words starting with same sound in close proximity)
        alliterations = 0
        for i in range(len(words) - 2):
            if len(words[i]) > 0 and len(words[i + 1]) > 0:
                if words[i][0] == words[i + 1][0]:
                    alliterations += 1

        return {
            'repetition_ratio': len(repeated_words) / len(set(words)) if words else 0,
            'alliteration_ratio': alliterations / len(words) if words else 0,
            'avg_sentence_similarity': self._calculate_sentence_similarity(sentences),
        }

    def _calculate_sentence_similarity(self, sentences: List[str]) -> float:
        """Calculate average similarity between consecutive sentences."""
        if len(sentences) < 2:
            return 0.0

        similarities = []
        for i in range(len(sentences) - 1):
            blob1 = TextBlob(sentences[i])
            blob2 = TextBlob(sentences[i + 1])
            # Calculate word overlap between consecutive sentences
            words1 = set(word.lower() for word in blob1.words)
            words2 = set(word.lower() for word in blob2.words)
            similarity = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
            similarities.append(similarity)

        return statistics.mean(similarities)

    def extract_statistical_features(self, text: str) -> Dict[str, float]:
        """
        Extract statistical text characteristics including vocabulary richness
        and distribution patterns.
        """
        words = word_tokenize(text.lower())
        word_counts = Counter(words)

        # Calculate vocabulary richness metrics
        vocabulary_size = len(set(words))
        total_words = len(words)

        # Type-Token Ratio (TTR)
        ttr = vocabulary_size / total_words if total_words > 0 else 0

        # Calculate frequency distribution statistics
        frequencies = list(word_counts.values())

        return {
            'type_token_ratio': ttr,
            'hapax_legomena_ratio': len(
                [w for w, c in word_counts.items() if c == 1]) / total_words if total_words > 0 else 0,
            'avg_word_frequency': statistics.mean(frequencies) if frequencies else 0,
            'frequency_std': statistics.stdev(frequencies) if len(frequencies) > 1 else 0,
        }

    def extract_all_features(self, text: str) -> Dict[str, Dict[str, float]]:
        """
        Extract all available features for a given text.
        """
        return {
            'linguistic': self.extract_linguistic_features(text),
            'emotional': self.extract_emotional_features(text),
            'rhetorical': self.extract_rhetorical_features(text),
            'statistical': self.extract_statistical_features(text)
        }

    def process_text_with_features(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process text using base preprocessing and extract additional features.
        """
        # Perform base preprocessing
        processed_text = self._Preprocessor__preprocess_text(text, lemma=True)

        # Extract features from original text (before preprocessing)
        features = self.extract_all_features(text)

        return processed_text, features

    def process_corpus_with_features(self, corpus: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process a corpus of texts and extract features for each document.
        """
        processed_texts = []
        features_list = []

        for text in corpus:
            processed_text, features = self.process_text_with_features(text)
            processed_texts.append(processed_text)
            features_list.append(features)

        return processed_texts, features_list


# Example usage:
if __name__ == "__main__":
    extractor = TextFeatureExtractor()

    # Example text
    text = """The sun was setting behind the mountains, casting long shadows 
              across the valley. Birds sang their evening songs, creating a 
              peaceful atmosphere. The air was crisp and clean, filled with 
              the scent of pine trees."""

    # Process text and extract features
    processed_text, features = extractor.process_text_with_features(text)

    print("Processed text:", processed_text)
    print("\nExtracted features:")
    for category, category_features in features.items():
        print(f"\n{category.upper()} FEATURES:")
        for feature, value in category_features.items():
            print(f"{feature}: {value}")
