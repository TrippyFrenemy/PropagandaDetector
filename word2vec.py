import os

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from data_manipulating.preprocessing import preprocess_corpus
from data_manipulating.model import save_model
from config import MODEL_PATH, LAST_NAME
from utils.draw_report import draw_report


def vectorize(sentence):
    words = sentence.split()
    words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(words_vecs) == 0:
        return np.zeros(w2v_model.vector_size)
    words_vecs = np.array(words_vecs)
    return words_vecs.mean(axis=0)


def vectorize_corpus(data):
    return np.array([vectorize(sentence) for sentence in data])


if MODEL_PATH == "models_back":
    raise Exception("Переписываешь резервную копию")
if os.path.exists(f"{MODEL_PATH}/word2vec_{LAST_NAME}.model"):
    answer = input(f"Хотите перезаписать данный путь \"{LAST_NAME}\"? y/n: ").lower()
    if answer != "y":
        raise Exception("Перезаписываете существующий файл")
else:
    print(LAST_NAME, "\n")

# Загрузка данных
data = pd.read_csv('datasets/propaganda_on_sentence_level.csv', sep=";", encoding="utf-8", index_col=False)

print(data.head())
print(data["Class"].value_counts())

data["PreText"] = preprocess_corpus(data["Text"])
# data.to_csv('datasets/propaganda_on_article_level_preprocessed.csv', index=False, sep=";", encoding="utf-8")
data['Class'].replace({'propaganda': 1, 'non-propaganda': 0}, inplace=True)

# data = over_sampling(data)

data = data[data["PreText"] != ""]
data = data[data["PreText"].apply(lambda x: len(x.split()) > 3) | (data["Class"] == 1)]

print(data.head())
print(data["Class"].value_counts())

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data["PreText"], data["Class"], test_size=0.1, random_state=20)

# Обучение Word2Vec модели
sentences = [sentence.split() for sentence in data["PreText"]]

# Загрузка модели
# w2v_model = Word2Vec.load("models/word2vec_model.model")
w2v_model = Word2Vec(sentences, vector_size=500)
# Сохранение модели
w2v_model.save(f"{MODEL_PATH}/word2vec_{LAST_NAME}.model")
X_train = vectorize_corpus(X_train)
X_test = vectorize_corpus(X_test)

# Обучение модели
logreg = LogisticRegression(random_state=20, max_iter=8000, solver="saga", penalty="elasticnet", l1_ratio=0.5)
logreg.fit(X_train, y_train)

save_model(logreg, f"{MODEL_PATH}/word2vec_logic_regretion_{LAST_NAME}")

# Предсказание на тестовых данных
y_pred_logreg = logreg.predict(X_test)

# Вывод отчета о классификации
print(classification_report(y_test, y_pred_logreg, zero_division=0))

draw_report("Logic Regression Word2vec", y_test, y_pred_logreg, "logic_regretion_", "png")
