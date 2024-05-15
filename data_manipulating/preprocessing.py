import re
from tqdm import tqdm
import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor


stop_words = set(stopwords.words('english'))
# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')


# Функция для очистки и предобработки текста на англ языке
def preprocess_text(text, lemma):

    if lemma:
        # Обработка текста используя spaCy
        doc = nlp(text)

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
    tokens = [word for word in tokens if word not in stop_words]

    # Сборка предложения обратно из токенов
    text = ' '.join(tokens)

    return text


def preprocess_corpus(corpus, lemma=True):
    processed_corpus = [preprocess_text(text, lemma) for text in tqdm(corpus)]
    return processed_corpus
