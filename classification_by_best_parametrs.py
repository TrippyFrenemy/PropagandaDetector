import os.path

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from data_manipulating.preprocessing import Preprocessor
from data_manipulating.model import save_model, save_vectorizer

from config_classification import MODEL_PATH, LAST_NAME
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

# Первые пять данных столбца
print(data.head())
print(data["Class"].value_counts())

preprocessing = Preprocessor()
# Обработка текста
data["PreText"] = preprocessing.preprocess_corpus(data["Text"])

# Замена классовых данных
data['Class'].replace({'propaganda': 1, 'non-propaganda': 0}, inplace=True)

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
save_vectorizer(vectorizer, f"{MODEL_PATH}/tfidf_{LAST_NAME}")

# Определение параметров для подбора
param_grid_logreg = {
    'solver': ['saga'],
    'penalty': ['elasticnet'],
    'l1_ratio': [0.1, 0.5, 0.9],
    'max_iter': [1000, 3000, 5000],
    'random_state': [20]
}
param_grid_forest = {
    'n_estimators': [100, 200, 300, 500, 800],
    'criterion': ['gini', 'entropy'],
    'max_depth': [100, 200, 300, 500, 800, None],
    'random_state': [20]
}
param_grid_tree = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'random_state': [20]
}
param_grid_knn = {
    'n_neighbors': [3, 5, 10, 15],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
}
param_grid_navbay = {

}


# Функция для автоматического поиска лучших параметров
def find_best_model(model, param_grid, X_train, y_train):
    try:
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='f1')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
    except Exception as ex:
        print(f"Ошибка при подборе параметров для {model.__class__.__name__}: {ex}")
        return None, None


# Поиск лучших параметров и обучение моделей
models_params = [
    (LogisticRegression(), param_grid_logreg, "logic_regretion"),
    (RandomForestClassifier(), param_grid_forest, "forest"),
    (DecisionTreeClassifier(), param_grid_tree, "tree"),
    (MultinomialNB(), param_grid_navbay, "naivebayes"),
    (KNeighborsClassifier(), param_grid_knn, "knn"),
]

best_models = {}

for model, param_grid, model_name in models_params:
    best_model, best_params = find_best_model(model, param_grid, X_train_tfidf, y_train)
    if best_model is not None:
        best_models[model_name] = (best_model, best_params)
        save_model(best_model, f"{MODEL_PATH}/{model_name}_{LAST_NAME}")

# Предсказание на тестовых данных и вывод отчета о классификации
for model_name, (model, params) in best_models.items():
    y_pred = model.predict(X_test_tfidf)
    print(f"\n{model_name.capitalize()}:\n", classification_report(y_test, y_pred))
    draw_report(model_name.capitalize(), y_test, y_pred, model_name + "_", "png")


# Вывод лучших параметров
print("Best Parameters:")
for model_name, (model, params) in best_models.items():
    print(f"{model_name.capitalize()}: {params}")


# Вывод путей сохраненных моделей
print("\nSaved Models:")
for model_name in best_models.keys():
    print(f"{model_name.capitalize()}: {MODEL_PATH}/{model_name}_{LAST_NAME}")
