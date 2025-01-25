import numpy
from gensim.models import Word2Vec
from data_manipulating.preprocessing import Preprocessor
from data_manipulating.manipulate_models import load_model, load_vectorizer
from config_classification import MODEL_PATH, LAST_NAME
from utils.google_translate import check_lang_corpus, translate_corpus

test_texts_ua = [
    # Нейтральні тексти
    "Сьогодні чудова сонячна погода, ідеально для прогулянки в парку.",
    "Нова книгарня відкрилась на розі вулиці Шевченка.",
    "У неділю відбудеться фестиваль вуличної їжі в центрі міста.",
    "Діти люблять гратися на новому дитячому майданчику.",
    "Цього року врожай яблук був особливо вдалим.",

    # Тексти з елементами пропаганди
    "Наш лідер - найвеличніший керівник за всю історію, тільки вороги батьківщини можуть це заперечувати.",
    "Всі наші проблеми через підступні дії іноземних агентів, які намагаються знищити нашу велику країну.",
    "Тільки повна лояльність до партії забезпечить світле майбутнє для наших дітей.",
    "Зрадники народу не заслуговують на милосердя, їх треба нещадно викривати та карати.",
    "Наша армія - найсильніша у світі, жоден ворог не встоїть перед нашою міццю.",

    # Маніпулятивні тексти
    "Звичайні громадяни страждають через підступні плани світової еліти.",
    "Альтернативні джерела енергії - це змова проти традиційної промисловості.",
    "Тільки справжні патріоти підтримують нашу політику, всі інші - вороги держави.",
    "Міжнародні організації намагаються підірвати наш суверенітет своїми брехливими звітами.",
    "Опозиція діє в інтересах іноземних спецслужб, це давно всім відомо.",

    # Тексти з прихованою пропагандою
    "Наші традиційні цінності під загрозою через вплив західної культури.",
    "Молодь забуває свою історію через підступну пропаганду ворогів.",
    "Справжні герої завжди підтримують чинну владу без жодних сумнівів.",
    "Критики нашої політики отримують фінансування з-за кордону.",
    "Реформи освіти знищують нашу унікальну систему виховання молоді."
]
print(test_texts_ua)

y_pred_test = [0] * 5 + [1] * 15
output = ' '.join(str(num) for num in y_pred_test)
output = "[" + output.strip() + "]"

if check_lang_corpus(test_texts_ua):
    text = translate_corpus(test_texts_ua)

preprocessing = Preprocessor()
text = preprocessing.preprocess_corpus(text)
print(text)

# Создание векторизатора TF-IDF
vectorizer = load_vectorizer(f"{MODEL_PATH}/tfidf_{LAST_NAME}")

X_test_tfidf = vectorizer.transform(text)

threshold = 0.45

logreg = load_model(f"{MODEL_PATH}/logic_regretion_{LAST_NAME}")
forest = load_model(f"{MODEL_PATH}/forest_{LAST_NAME}")
tree = load_model(f"{MODEL_PATH}/tree_{LAST_NAME}")
knn = load_model(f"{MODEL_PATH}/knn_{LAST_NAME}")

# Предсказание на тестовых данных
y_pred_logreg = logreg.predict_proba(X_test_tfidf)[:, 1]
y_pred_forest = forest.predict_proba(X_test_tfidf)[:, 1]
y_pred_tree = tree.predict_proba(X_test_tfidf)[:, 1]
y_pred_knn = knn.predict_proba(X_test_tfidf)[:, 1]

print("expected:   ", output)
print("logreg:     ", y_pred_logreg)
print("forest:     ", y_pred_forest)
print("tree:       ", y_pred_tree)
print("knn:        ", y_pred_knn)
print()

# Предсказание на тестовых данных
y_pred_logreg = (logreg.predict_proba(X_test_tfidf)[:, 1] >= threshold).astype(int)
y_pred_forest = (forest.predict_proba(X_test_tfidf)[:, 1] >= threshold).astype(int)
y_pred_tree = (tree.predict_proba(X_test_tfidf)[:, 1] >= threshold).astype(int)
y_pred_knn = (knn.predict_proba(X_test_tfidf)[:, 1] >= 0.5).astype(int)

print("expected:   ", output)
print("logreg:     ", y_pred_logreg)
print("forest:     ", y_pred_forest)
print("tree:       ", y_pred_tree)
print("knn:        ", y_pred_knn)
print()

print("vectorizer: ", vectorizer.get_params())
print("logreg:     ", logreg.get_params())
print("forest:     ", forest.get_params())
print("tree:       ", tree.get_params())
print("knn:        ", knn.get_params())
print()
