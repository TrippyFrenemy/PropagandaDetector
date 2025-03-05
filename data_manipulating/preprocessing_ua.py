import csv
import re

import emoji
from tqdm import tqdm


class PreprocessorUA:
    def __init__(self, nlp):
        # Використання українських стоп-слів
        with open('../data_manipulating/stopwords_ua.txt', 'r', encoding='utf-8') as f:
            self.__stop_words = set(line.strip().lower() for line in f if line.strip())
        # Завантаження української моделі spaCy
        self.nlp = nlp

    # Функція для очищення та підготовчої обробки тексту українською мовою
    def __preprocess_text(self, text, lemma):
        if lemma:
            # Обробка тексту з використанням spaCy
            doc = self.nlp(text)
            # Лематизація токенів та видалення іменованих сутностей
            lemmatized_tokens = [token.lemma_ for token in doc if not token.ent_type_]
            # Збирання речення назад з токенів
            text = ' '.join(lemmatized_tokens)
        # Приведення до нижнього регістру
        text = text.lower()
        # Видалення спеціальних символів та цифр
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d', ' ', text)
        # Видалення всіх одиночних літер
        text = re.sub(r'\s+[\u0400-\u04FF]\s+', ' ', text)
        text = text.replace("\n", " ").replace("\t", " ")
        # Видалення початкових та кінцевих пробілів
        text = text.strip()
        # Токенізація (поділ на слова)
        tokens = text.split()
        # Видалення стоп-слів
        tokens = [word for word in tokens if word not in self.__stop_words]
        # Збирання речення назад з токенів
        text = ' '.join(tokens)
        return text

    def preprocess_corpus(self, corpus, lemma=True):
        processed_corpus = [self.__preprocess_text(text, lemma) for text in tqdm(corpus)]
        return processed_corpus


class EnhancedPreprocessorUA(PreprocessorUA):
    def __init__(self, nlp):
        super().__init__(nlp)
        # Завантаження тонального словника для аналізу тональності
        self.__sentiment_dict = {}
        # self.__nlp = spacy.load('uk_core_news_sm')
        with open('../data_manipulating/sentiment_ua.csv', 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader, None)  # пропустити заголовок
            for word, value in reader:
                self.__sentiment_dict[word.lower()] = int(value)

    def __extract_emotional_markers(self, text):
        """
        Витягує та аналізує емоційні маркери з тексту (емодзі та розділові знаки).
        Повертає очищений від емодзі текст та словник емоційних ознак.
        """
        emojis_list = [c for c in text if c in emoji.EMOJI_DATA]
        emoji_descriptions = [emoji.demojize(e) for e in emojis_list]
        exclamation_count = text.count('!')
        question_count = text.count('?')
        emotional_features = {
            'emoji_count': len(emojis_list),
            'emoji_descriptions': emoji_descriptions,
            'exclamation_count': exclamation_count,
            'question_count': question_count
        }
        cleaned_text = ''.join(c for c in text if c not in emoji.EMOJI_DATA)
        return cleaned_text, emotional_features

    def __analyze_syntax(self, doc):
        """
        Аналізує синтаксичну структуру тексту за допомогою spaCy.
        Повертає словник синтаксичних ознак.
        """
        syntactic_features = {
            'sentence_count': len(list(doc.sents)),
            'dependency_patterns': [],
            'pos_tags': {}
        }
        for token in doc:
            if token.dep_:
                pattern = f'{token.head.text} --{token.dep_}--> {token.text}'
                syntactic_features['dependency_patterns'].append(pattern)
        for token in doc:
            pos = token.pos_
            syntactic_features['pos_tags'][pos] = syntactic_features['pos_tags'].get(pos, 0) + 1
        return syntactic_features

    def __extract_named_entities(self, doc):
        """
        Витягує іменовані сутності за допомогою NER spaCy.
        Повертає словник з інформацією про сутності.
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
            named_entities['entity_counts'][ent.label_] = named_entities['entity_counts'].get(ent.label_, 0) + 1
        return named_entities

    def __analyze_sentiment(self, text):
        """
        Виконує аналіз тональності тексту на основі тонального словника.
        Повертає словник з показниками тональності (полярність, суб'єктивність та ін.).
        """
        doc = self.nlp(text)
        words = [token.lemma_.lower() for token in doc if token.is_alpha]
        pos_count = sum(1 for w in words if w in self.__sentiment_dict and self.__sentiment_dict[w] > 0)
        neg_count = sum(1 for w in words if w in self.__sentiment_dict and self.__sentiment_dict[w] < 0)
        score = sum(self.__sentiment_dict[w] for w in words if w in self.__sentiment_dict)
        polarity = score / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0
        subjectivity = (pos_count + neg_count) / len(words) if len(words) > 0 else 0
        sentiment_features = {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'positive_words': pos_count,
            'negative_words': neg_count
        }
        return sentiment_features

    def preprocess_text(self, text, include_features=True):
        """
        Покращена обробка тексту з додатковим виділенням ознак.
        Повертає (опрацьований текст, словник ознак) якщо include_features=True, інакше опрацьований текст.
        """
        text, emotional_features = self.__extract_emotional_markers(text)
        doc = self.nlp(text)
        if include_features:
            features = {
                'emotional': emotional_features,
                'syntactic': self.__analyze_syntax(doc),
                'named_entities': self.__extract_named_entities(doc),
                'sentiment': self.__analyze_sentiment(text)
            }
        processed_text = super()._PreprocessorUA__preprocess_text(text, lemma=True)
        return (processed_text, features) if include_features else processed_text

    def preprocess_corpus(self, corpus, include_features=True):
        """
        Обробляє весь корпус текстів з розширеним препроцесингом.
        Повертає (список опрацьованих текстів, список ознак) якщо include_features=True, інакше список опрацьованих текстів.
        """
        results = [self.preprocess_text(text, include_features) for text in tqdm(corpus)]
        if include_features:
            processed_texts, features_list = zip(*results)
            return processed_texts, features_list
        else:
            return results


if __name__ == "__main__":
    # Пример использования:
    preprocessor = EnhancedPreprocessorUA()

    # Обработка отдельного текста с характеристиками
    text = "Я люблю цей продукт! 😊 Директор Іван Петренко оголосив чудові результати."
    processed_text, features = preprocessor.preprocess_text(text)
    print(processed_text)
    print(features)

    # Обработка корпуса
    corpus = [
        "Апельсин є помаранчевим.",
        "Чому ми повинні підтримувати нашу велику націю.",
        "Наша країна стоїть на порозі найбільших досягнень в історії. У той час як зловмисні критики намагаються підірвати наш прогрес, ми повинні залишатися об'єднаними і непохитними.",
        "Наш лідер - найкращий з найкращих у всьому світі.",
        "Щоденна боротьба звичайних людей проти впливу фемінізму."
    ]

    processed_texts, features_list = preprocessor.preprocess_corpus(corpus)
    print(processed_texts)
    print(features_list)
