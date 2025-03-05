import statistics
from collections import Counter
from typing import Dict, List, Tuple, Any

from data_manipulating.preprocessing_ua import EnhancedPreprocessorUA


class TextFeatureExtractorUA(EnhancedPreprocessorUA):
    """
    Розширений екстрактор ознак тексту (для української мови), побудований на EnhancedPreprocessorUA.
    Додає лінгвістичні, емоційні, риторичні та статистичні ознаки тексту.
    """

    def __init__(self, nlp):
        super().__init__(nlp)
        # Для української: визначаємо набір голосних для підрахунку складів
        self.prondict = None  # відсутній словник вимови для української
        self.__vowels_ua = set("аеєиіїоуюя")
        # self.__nlp = spacy.load('uk_core_news_sm')

    def _count_syllables(self, word: str) -> int:
        """Рахує кількість складів у слові (українською мовою) шляхом підрахунку голосних літер."""
        return sum(1 for char in word.lower() if char in self.__vowels_ua)

    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Виділяє лінгвістичні ознаки: довжина речень, довжина/складність слів, показники читабельності.
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)
        words = [token.text for token in doc]
        # Average sentence length (words per sentence)
        sent_lengths = [len([token for token in sent]) for sent in sentences]
        avg_sent_length = statistics.mean(sent_lengths) if sent_lengths else 0
        sent_length_std = statistics.stdev(sent_lengths) if len(sent_lengths) > 1 else 0
        # Word complexity metrics
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
        Виділяє емоційні характеристики за допомогою лексичного аналізу тональності та розділових знаків.
        """
        sentiment = self._EnhancedPreprocessorUA__analyze_sentiment(text)
        sentiment_scores = {
            'polarity': sentiment['polarity'],
            'subjectivity': sentiment['subjectivity'],
        }
        # Punctuation-based emotion markers
        sentiment_scores.update({
            'exclamation_ratio': text.count('!') / len(text) if len(text) > 0 else 0,
            'question_ratio': text.count('?') / len(text) if len(text) > 0 else 0,
            'ellipsis_ratio': text.count('...') / len(text) if len(text) > 0 else 0,
        })
        return sentiment_scores

    def extract_rhetorical_features(self, text: str) -> Dict[str, float]:
        """
        Визначає риторичні засоби: повторення та алітерацію.
        """
        doc = self.nlp(text)
        words = [token.text.lower() for token in doc]
        sentences = list(doc.sents)
        # Word repetition
        word_counts = Counter(words)
        repeated_words = {word: count for word, count in word_counts.items() if count > 1}
        # Alliteration (consecutive words starting with same letter)
        alliterations = 0
        for i in range(len(words) - 1):
            if words[i] and words[i + 1] and words[i][0] == words[i + 1][0]:
                alliterations += 1
        return {
            'repetition_ratio': len(repeated_words) / len(set(words)) if words else 0,
            'alliteration_ratio': alliterations / len(words) if words else 0,
            'avg_sentence_similarity': self._calculate_sentence_similarity(sentences),
        }

    def _calculate_sentence_similarity(self, sentences: List[str]) -> float:
        """Обчислює середню подібність між послідовними реченнями (спільні слова)."""
        if len(sentences) < 2:
            return 0.0
        similarities = []
        for i in range(len(sentences) - 1):
            sent1 = sentences[i]
            sent2 = sentences[i + 1]
            words1 = {token.text.lower() for token in sent1 if token.is_alpha}
            words2 = {token.text.lower() for token in sent2 if token.is_alpha}
            if not words1 or not words2:
                similarities.append(0)
            else:
                similarities.append(len(words1 & words2) / len(words1 | words2))
        return statistics.mean(similarities)

    def extract_statistical_features(self, text: str) -> Dict[str, float]:
        """
        Виділяє статистичні характеристики: багатство словникового запасу та розподіл частот.
        """
        doc = self.nlp(text)
        words = [token.text.lower() for token in doc]
        word_counts = Counter(words)
        vocabulary_size = len(set(words))
        total_words = len(words)
        # Type-Token Ratio (TTR)
        ttr = vocabulary_size / total_words if total_words > 0 else 0
        frequencies = list(word_counts.values())
        return {
            'type_token_ratio': ttr,
            'hapax_legomena_ratio': len(
                [w for w, c in word_counts.items() if c == 1]) / total_words if total_words > 0 else 0,
            'avg_word_frequency': statistics.mean(frequencies) if frequencies else 0,
            'frequency_std': statistics.stdev(frequencies) if len(frequencies) > 1 else 0,
        }

    def extract_all_features(self, text: str) -> Dict[str, Dict[str, float]]:
        """Виділяє всі категорії ознак для даного тексту."""
        return {
            'linguistic': self.extract_linguistic_features(text),
            'emotional': self.extract_emotional_features(text),
            'rhetorical': self.extract_rhetorical_features(text),
            'statistical': self.extract_statistical_features(text)
        }

    def process_text_with_features(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Обробляє текст базовим препроцесингом та виділяє всі ознаки.
        Повертає (опрацьований текст, словник ознак).
        """
        # Perform base preprocessing (lemmatization, stopword removal)
        processed_text = self._PreprocessorUA__preprocess_text(text, lemma=True)
        # Extract features from the original text
        features = self.extract_all_features(text)
        return processed_text, features


# Example usage:
if __name__ == "__main__":
    extractor = TextFeatureExtractorUA()

    # Пример текста на украинском языке
    text = """Сонце сідало за горами, відкидаючи довгі тіні
              на долину. Птахи співали свої вечірні пісні, створюючи
              атмосферу спокою. Повітря було свіжим і чистим, наповненим
              ароматом соснових дерев."""

    # Обработка текста и извлечение характеристик
    processed_text, features = extractor.process_text_with_features(text)

    print("Processed text:", processed_text)
    print("\nExtracted features:")
    for category, category_features in features.items():
        print(f"\n{category.upper()} FEATURES:")
        for feature, value in category_features.items():
            print(f"{feature}: {value}")
