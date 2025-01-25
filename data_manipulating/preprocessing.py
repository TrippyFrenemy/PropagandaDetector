import re

import nltk
from tqdm import tqdm
import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import emoji


class Preprocessor:
    def __init__(self):
        self.__stop_words = set(stopwords.words('english'))
        # Load the spaCy English model
        self.__nlp = spacy.load('en_core_web_sm')

    # Функция для очистки и предобработки текста на англ языке
    def __preprocess_text(self, text, lemma):
        if lemma:
            # Обработка текста используя spaCy
            doc = self.__nlp(text)

            # Приведение токенов к лемматизации и удаления собственных названий
            lemmatized_tokens = [token.lemma_ for token in doc if not token.ent_type_]

            # Сборка предложения обратно из токенов
            text = ' '.join(lemmatized_tokens)

        # Приведение к нижнему регистру
        text = text.lower()

        # Удаление специальных символов и цифр
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d', ' ', text)

        # Удаление всех одиночных символов
        text = re.sub(r'\s+[a-z]\s+', ' ', text)

        text.replace("\n", "")
        text.replace("\t", "")

        # Удаление начальных и конечных пробелов
        text = text.strip()

        # Токенизация
        tokens = word_tokenize(text)

        # Удаление стоп-слов и стемминг
        tokens = [word for word in tokens if word not in self.__stop_words]

        # Сборка предложения обратно из токенов
        text = ' '.join(tokens)

        return text

    def preprocess_corpus(self, corpus, lemma=True):
        processed_corpus = [self.__preprocess_text(text, lemma) for text in tqdm(corpus)]
        return processed_corpus


class EnhancedPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()
        self.__stop_words = set(stopwords.words('english'))
        # Load the spaCy English model with all pipeline components
        self.__nlp = spacy.load('en_core_web_sm')
        # Initialize NLTK's VADER sentiment analyzer
        self.__sia = SentimentIntensityAnalyzer()

    def __extract_emotional_markers(self, text):
        """
        Extract and analyze emotional markers from text including emojis and punctuation.
        Returns the cleaned text and emotional features dictionary.
        """
        # Extract emojis and their descriptions
        emojis_list = [c for c in text if c in emoji.EMOJI_DATA]
        emoji_descriptions = [emoji.demojize(e) for e in emojis_list]

        # Count emotional punctuation
        exclamation_count = text.count('!')
        question_count = text.count('?')

        # Create emotional features dictionary
        emotional_features = {
            'emoji_count': len(emojis_list),
            'emoji_descriptions': emoji_descriptions,
            'exclamation_count': exclamation_count,
            'question_count': question_count
        }

        # Remove emojis from text for further processing
        cleaned_text = ''.join(c for c in text if c not in emoji.EMOJI_DATA)

        return cleaned_text, emotional_features

    def __analyze_syntax(self, doc):
        """
        Analyze syntactic structure of the text using spaCy.
        Returns dictionary with syntactic features.
        """
        syntactic_features = {
            'sentence_count': len(list(doc.sents)),
            'dependency_patterns': [],
            'pos_tags': {}
        }

        # Extract dependency patterns
        for token in doc:
            if token.dep_ != '':
                pattern = f'{token.head.text} --{token.dep_}--> {token.text}'
                syntactic_features['dependency_patterns'].append(pattern)

        # Count POS tags
        for token in doc:
            pos = token.pos_
            syntactic_features['pos_tags'][pos] = syntactic_features['pos_tags'].get(pos, 0) + 1

        return syntactic_features

    def __extract_named_entities(self, doc):
        """
        Extract named entities using spaCy's NER.
        Returns dictionary with entity information.
        """
        named_entities = {
            'entities': [],
            'entity_counts': {}
        }

        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            named_entities['entities'].append(entity_info)
            named_entities['entity_counts'][ent.label_] = \
                named_entities['entity_counts'].get(ent.label_, 0) + 1

        return named_entities

    def __analyze_sentiment(self, text):
        """
        Perform sentiment analysis using both VADER and TextBlob.
        Returns dictionary with sentiment scores.
        """
        # VADER sentiment analysis
        vader_scores = self.__sia.polarity_scores(text)

        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment

        sentiment_features = {
            'vader': vader_scores,
            'textblob': {
                'polarity': textblob_sentiment.polarity,
                'subjectivity': textblob_sentiment.subjectivity
            }
        }

        return sentiment_features

    def preprocess_text(self, text, include_features=True):
        """
        Enhanced text preprocessing with optional feature extraction.

        Args:
            text (str): Input text to process
            include_features (bool): Whether to include additional features in output

        Returns:
            tuple: (processed_text, features_dict) if include_features=True
                  processed_text only if include_features=False
        """
        # Extract emotional markers
        text, emotional_features = self.__extract_emotional_markers(text)

        # Process with spaCy
        doc = self.__nlp(text)

        # Extract features if requested
        if include_features:
            features = {
                'emotional': emotional_features,
                'syntactic': self.__analyze_syntax(doc),
                'named_entities': self.__extract_named_entities(doc),
                'sentiment': self.__analyze_sentiment(text)
            }

        # Perform base preprocessing using parent class method
        processed_text = super()._Preprocessor__preprocess_text(text, lemma=True)

        return (processed_text, features) if include_features else processed_text

    def preprocess_corpus(self, corpus, include_features=True):
        """
        Process entire corpus with enhanced preprocessing.

        Args:
            corpus (list): List of texts to process
            include_features (bool): Whether to include additional features

        Returns:
            tuple: (processed_texts, features_list) if include_features=True
                  processed_texts only if include_features=False
        """
        results = [self.preprocess_text(text, include_features) for text in tqdm(corpus)]

        if include_features:
            processed_texts, features_list = zip(*results)
            return processed_texts, features_list
        else:
            return results


if __name__ == "__main__":
    # Example usage:
    preprocessor = EnhancedPreprocessor()

    # Process single text with features
    text = "I love this product! 😊 The CEO John Smith announced great results."
    processed_text, features = preprocessor.preprocess_text(text)
    print(processed_text, features)

    # Process corpus
    corpus = [
        "Orange is orange.",
        "Why We should support our great nation.",
        "Our country stands on the threshold of the greatest achievements in history. While malicious critics try to undermine our progress, we must remain united and steadfast. Our leader, continuing a long tradition of outstanding government, strives tirelessly for the benefit of every citizen.",
        "Our leader is the best of the best all over the world.",
        "The Everyday Struggle of Ordinary People Against Feminist Influence.".lower(),
        "In these difficult times, when national and international enemies threaten our way of life, our indomitable leader stands guard over stability and development. Thanks to his wise and decisive actions, our country's economy is experiencing unprecedented growth.",
        "We stand on the threshold of the greatest challenges of our time. Our society, our traditions and our history are under attack from those who want to see us weakened and divided. Do not be deceived: our enemies are both internal traitors and external aggressors.",
        "Pamela Geller and Robert Spencer co founded anti Muslim group Stop Islamization of America.",
        "Children loves chocolate.",
        "Little Kate always looks forward to the weekend to go for a walk in the nearby park.",
        "Make sure you use the correct instances of models and vectorizers for testing that match the number of features used in the trained models.",
        "Save the crowbar from the ship to blow up the Japanese.",
        "Destroy this crazy beast. Enroll in the US Army.",
        "Fulfill the five-year plan not in five years, but in four.",
        "Death to capital or death under the heel of capital!",
        "Workers of the world, unite! For a worldwide October!",
        '\"\"\"Working together, we\'ll get it taken care of,\"\" Trump said.\"']

    processed_texts, features_list = preprocessor.preprocess_corpus(corpus)
    print(processed_texts, features_list)
