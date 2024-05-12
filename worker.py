import numpy
from gensim.models import Word2Vec
from data_manipulating.preprocessing import preprocess_corpus
from data_manipulating.model import load_model, load_vectorizer
from config_classification import MODEL_PATH, LAST_NAME

text = [
    "Orange is orange",
    "Why We should support our great nation",
    "Our country stands on the threshold of the greatest achievements in history. While malicious critics try to undermine our progress, we must remain united and steadfast. Our leader, continuing a long tradition of outstanding government, strives tirelessly for the benefit of every citizen.",
    "Our leader is the best of the best all over the world",
    "The Everyday Struggle of Ordinary People Against Feminist Influence".lower(),
    "In these difficult times, when national and international enemies threaten our way of life, our indomitable leader stands guard over stability and development. Thanks to his wise and decisive actions, our country's economy is experiencing unprecedented growth.",
    "We stand on the threshold of the greatest challenges of our time. Our society, our traditions and our history are under attack from those who want to see us weakened and divided. Do not be deceived: our enemies are both internal traitors and external aggressors.",
    "Pamela Geller and Robert Spencer co founded anti Muslim group Stop Islamization of America",
    "Children loves chocolate",
    "Little Kate always looks forward to the weekend to go for a walk in the nearby park.",
    "Make sure you use the correct instances of models and vectorizers for testing that match the number of features used in the trained models",
    "Save the crowbar from the ship to blow up the Japanese",
    "Destroy this crazy beast. Enroll in the US Army",
    "Fulfill the five-year plan not in five years, but in four",
    "Death to capital or death under the heel of capital!",
    "Workers of the world, unite! For a worldwide October!",
    "Why we fight - for our children`s bread"]
print(text)

y_pred_test = [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1]
output = ' '.join(str(num) for num in y_pred_test)
output = "[" + output.strip() + "]"

text = preprocess_corpus(text)
print(text)

# Создание векторизатора TF-IDF
vectorizer = load_vectorizer(f"{MODEL_PATH}/tfidf_{LAST_NAME}")

X_test_tfidf = vectorizer.transform(text)

threshold = 0.45

logreg = load_model(f"{MODEL_PATH}/logic_regretion_{LAST_NAME}")
forest = load_model(f"{MODEL_PATH}/forest_{LAST_NAME}")
tree = load_model(f"{MODEL_PATH}/tree_{LAST_NAME}")
naivebayes = load_model(f"{MODEL_PATH}/naivebayes_{LAST_NAME}")
knn = load_model(f"{MODEL_PATH}/knn_{LAST_NAME}")

# Предсказание на тестовых данных
y_pred_logreg = logreg.predict_proba(X_test_tfidf)[:, 1]
y_pred_forest = forest.predict_proba(X_test_tfidf)[:, 1]
y_pred_tree = tree.predict_proba(X_test_tfidf)[:, 1]
y_pred_naivebayes = naivebayes.predict_proba(X_test_tfidf)[:, 1]
y_pred_knn = knn.predict_proba(X_test_tfidf)[:, 1]

print("expected:   ", output)
print("logreg:     ", y_pred_logreg)
print("forest:     ", y_pred_forest)
print("tree:       ", y_pred_tree)
print("naivebayes: ", y_pred_naivebayes)
print("knn:        ", y_pred_knn)
print()

# Предсказание на тестовых данных
y_pred_logreg = (logreg.predict_proba(X_test_tfidf)[:, 1] >= threshold).astype(int)
y_pred_forest = (forest.predict_proba(X_test_tfidf)[:, 1] >= threshold).astype(int)
y_pred_tree = (tree.predict_proba(X_test_tfidf)[:, 1] >= threshold).astype(int)
y_pred_naivebayes = (naivebayes.predict_proba(X_test_tfidf)[:, 1] >= threshold).astype(int)
y_pred_knn = (knn.predict_proba(X_test_tfidf)[:, 1] >= 0.5).astype(int)

print("expected:   ", output)
print("logreg:     ", y_pred_logreg)
print("forest:     ", y_pred_forest)
print("tree:       ", y_pred_tree)
print("naivebayes: ", y_pred_naivebayes)
print("knn:        ", y_pred_knn)
print()

print("vectorizer: ", vectorizer.get_params())
print("logreg:     ", logreg.get_params())
print("forest:     ", forest.get_params())
print("tree:       ", tree.get_params())
print("naivebayes: ", naivebayes.get_params())
print("knn:        ", knn.get_params())
print()
