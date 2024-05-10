import os.path

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from data_manipulating.preprocessing import preprocess_corpus
from data_manipulating.model import save_model, save_vectorizer

from config import MODEL_PATH, LAST_NAME
from utils.draw_report import draw_report

if MODEL_PATH == "models_back":
    raise Exception("Переписываешь резервную копию")
if os.path.exists(f"{MODEL_PATH}/tfidf_{LAST_NAME}.joblib"):
    answer = input(f"Хотите перезаписать данный путь \"{LAST_NAME}\"? y/n: ").lower()
    if answer != "y":
        raise Exception("Перезаписываете существующий файл")
else:
    print(LAST_NAME, "\n")

# Загрузка данных
data = pd.read_csv('datasets/propaganda_on_sentence_level.csv', sep=";", encoding="utf-8")
# data = pd.read_csv('datasets/propaganda_on_article_level_preprocessed.csv', sep=";", encoding="Windows-1251")

# Первые пять данных столбца
print(data.head())
print(data["Class"].value_counts())

# Обработка текста
data["PreText"] = preprocess_corpus(data["Text"])

# Сохранение обработаных данных
data.to_csv('datasets/propaganda_on_sentence_level_preprocessed.csv', index=False, sep=";", encoding="utf-8")
# data.to_csv('datasets/propaganda_on_article_level_preprocessed.csv', index=False, sep=";", encoding="Windows-1251")
data['Class'].replace({'propaganda': 1, 'non-propaganda': 0}, inplace=True)

# data = over_sampling(data)

# Удаление пустых значений
data = data[data["PreText"] != ""]
data = data[data["PreText"].apply(lambda x: len(x.split()) > 3) | (data["Class"] == 1)]

print(data.head())
print(data["Class"].value_counts())

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data["PreText"], data["Class"], test_size=0.1, random_state=20)

# Создание векторизатора TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# save_vectorizer(vectorizer, "models/tfidf_sentence_nooversampling")
save_vectorizer(vectorizer, f"{MODEL_PATH}/tfidf_{LAST_NAME}")
# print(vectorizer.vocabulary_)

# Создание и обучение модели логистической регрессии
logreg = LogisticRegression(random_state=20, max_iter=8000, solver="saga", penalty="elasticnet", l1_ratio=0.5)
logreg.fit(X_train_tfidf, y_train)

# Создание и обучение модели случайного леса
forest = RandomForestClassifier(n_estimators=100, random_state=20)
forest.fit(X_train_tfidf, y_train)

# Создание и обучение модели дерева решений
tree = DecisionTreeClassifier(random_state=20)
tree.fit(X_train_tfidf, y_train)

navbay = MultinomialNB()
navbay.fit(X_train_tfidf, y_train)

# Создание и обучение модели к ближайших соседов
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train_tfidf, y_train)

# model = load_model("models/logic_regretion")
save_model(logreg, f"{MODEL_PATH}/logic_regretion_{LAST_NAME}")
save_model(forest, f"{MODEL_PATH}/forest_{LAST_NAME}")
save_model(tree, f"{MODEL_PATH}/tree_{LAST_NAME}")
save_model(navbay, f"{MODEL_PATH}/naivebayes_{LAST_NAME}")
save_model(knn, f"{MODEL_PATH}/knn_{LAST_NAME}")

# Предсказание на тестовых данных
y_pred_logreg = logreg.predict(X_test_tfidf)
y_pred_forest = forest.predict(X_test_tfidf)
y_pred_tree = tree.predict(X_test_tfidf)
y_pred_navbay = navbay.predict(X_test_tfidf)
y_pred_knn = knn.predict(X_test_tfidf)

# threshold = 0.45
#
# Предсказание на тестовых данных
# y_pred_logreg = (logreg.predict_proba(X_test_tfidf)[:, 1] >= threshold).astype(int)
# y_pred_forest = (forest.predict_proba(X_test_tfidf)[:, 1] >= threshold).astype(int)
# y_pred_tree = (tree.predict_proba(X_test_tfidf)[:, 1] >= threshold).astype(int)
# y_pred_navbay = (navbay.predict_proba(X_test_tfidf)[:, 1] >= threshold).astype(int)

# Вывод отчета о классификации
print("Logreg:\n", classification_report(y_test, y_pred_logreg))
print("\nForest:\n", classification_report(y_test, y_pred_forest))
print("\nTree:  \n", classification_report(y_test, y_pred_tree))
print("\nNavbay:\n", classification_report(y_test, y_pred_navbay))
print("\nKNN:   \n", classification_report(y_test, y_pred_knn))

# Вывод графиков отчета о классификации
draw_report("Logic Regression", y_test, y_pred_logreg, "logic_regretion_", "png")
draw_report("Forest", y_test, y_pred_forest, "forest_", "png")
draw_report("Tree", y_test, y_pred_tree, "tree_", "png")
draw_report("Naive Bayes", y_test, y_pred_navbay, "naivebayes_", "png")
draw_report("K Near Neighbours", y_test, y_pred_knn, "knn_", "png")



