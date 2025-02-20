import numpy
from gensim.models import Word2Vec
from data_manipulating.preprocessing import Preprocessor
from data_manipulating.manipulate_models import load_model, load_vectorizer
from config_classification import MODEL_PATH, LAST_NAME
from pipelines.cascade_classification import logger
from utils.translate import check_lang_corpus, translate_corpus

test_texts_ua = [
        "Сьогодні чудова сонячна погода, ідеально для прогулянки в парку.",
        "Нова книгарня відкрилась на розі вулиці Шевченка.",
        "У неділю відбудеться фестиваль вуличної їжі в центрі міста.",
        "Діти люблять гратися на новому дитячому майданчику.",
        "Цього року врожай яблук був особливо вдалим.",
        "На місцевому ринку з'явилися свіжі овочі та фрукти.",
        "Міський парк прикрасили новими лавками та ліхтарями.",
        "Розклад руху автобусів змінено на зимовий період.",
        "В універмазі триває сезонний розпродаж зимового одягу.",
        "Місцева бібліотека поповнилась новими виданнями.",
        "Ремонт дороги планують завершити до кінця місяця.",
        "У зоопарку народилося троє тигренят.",
        "Студенти готуються до зимової сесії в університеті.",
        "На центральній площі встановили новорічну ялинку.",
        "Фермери зібрали рекордний урожай пшениці.",
        "У місцевому театрі відбудеться прем'єра нової вистави.",
        "Відкрився новий спортивний комплекс із басейном.",
        "Метеорологи прогнозують сонячну погоду на вихідні.",
        "У парку встановили нові гойдалки для дітей.",
        "Місцева пекарня випікає свіжий хліб щоранку.",
        "На озері можна побачити диких качок.",
        "Волонтери прибрали територію міського пляжу.",
        "У ботанічному саду розквітли рідкісні орхідеї.",
        "Школярі відвідали виставку сучасного мистецтва.",
        "На стадіоні пройдуть змагання з легкої атлетики.",
        "Місцеві майстри проводять майстер-класи з гончарства.",
        "У кінотеатрі показують нові українські фільми.",
        "Розпочався сезон збору полуниці.",
        "Відкрився новий магазин органічних продуктів.",
        "На набережній встановили нові велопарковки.",

        "Наш лідер - найвеличніший керівник за всю історію, тільки вороги батьківщини можуть це заперечувати.",
        "Всі наші проблеми через підступні дії іноземних агентів, які намагаються знищити нашу велику країну.",
        "Тільки повна лояльність до партії забезпечить світле майбутнє для наших дітей.",
        "Зрадники народу не заслуговують на милосердя, їх треба нещадно викривати та карати.",
        "Наша армія - найсильніша у світі, жоден ворог не встоїть перед нашою міццю.",
        "Єдність партії та народу - запорука нашої непереможності.",
        "Вороги намагаються підірвати наш дух, але ми сильніші за їхні підступні плани.",
        "Тільки сліпі не бачать величі наших досягнень під керівництвом мудрого лідера.",
        "Наш народ має особливу місію у світовій історії.",
        "Зовнішні вороги бояться нашої могутності та єдності.",
        "Критики влади - це агенти іноземного впливу.",
        "Наша ідеологія - єдиний шлях до процвітання.",
        "Патріоти підтримують кожне рішення керівництва.",
        "Опозиція працює на знищення нашої держави.",
        "Тільки сильна влада може захистити народ від ворогів.",
        "Звичайні громадяни страждають через підступні плани світової еліти.",
        "Альтернативні джерела енергії - це змова проти традиційної промисловості.",
        "Тільки справжні патріоти підтримують нашу політику, всі інші - вороги держави.",
        "Міжнародні організації намагаються підірвати наш суверенітет своїми брехливими звітами.",
        "Опозиція діє в інтересах іноземних спецслужб, це давно всім відомо.",
        "Незалежні ЗМІ поширюють дезінформацію про нашу країну.",
        "Екологічні активісти працюють на закордонні фонди.",
        "Реформатори хочуть знищити нашу економічну незалежність.",
        "Наші традиційні цінності під загрозою через вплив західної культури.",
        "Молодь забуває свою історію через підступну пропаганду ворогів.",
        "Критики нашої політики отримують фінансування з-за кордону.",
        "Реформи освіти знищують нашу унікальну систему виховання молоді.",
        "Інтернет-технології відволікають молодь від справжніх цінностей.",
        "Необхідно захищати дітей від шкідливого впливу масової культури.",
        "Історичні події потрібно трактувати правильно, а не так, як нав'язують вороги.",
    ]
print(test_texts_ua)

y_pred_test = [0] * 15 + [1] * 45
output = ' '.join(str(num) for num in y_pred_test)
output = "[" + output.strip() + "]"

if check_lang_corpus(test_texts_ua):
    text = translate_corpus(test_texts_ua)

preprocessing = Preprocessor()
text = preprocessing.preprocess_corpus(text)
print(text)

# Создание векторизатора TF-IDF
vectorizer = load_vectorizer(f"../{MODEL_PATH}/tfidf_{LAST_NAME}")

X_test_tfidf = vectorizer.transform(text)

threshold = 0.45

logreg = load_model(f"../{MODEL_PATH}/logic_regretion_{LAST_NAME}")
forest = load_model(f"../{MODEL_PATH}/forest_{LAST_NAME}")
tree = load_model(f"../{MODEL_PATH}/tree_{LAST_NAME}")
knn = load_model(f"../{MODEL_PATH}/knn_{LAST_NAME}")

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

matches = sum(1 for p, b in zip(y_pred_test, y_pred_forest) if p == b)
total = len(y_pred_test)

# Вычисляем точность
accuracy = matches / total

logger.info(f"Совпадений: {matches}/{total}")
logger.info(f"Точность: {accuracy:.2%}")

# Дополнительная статистика
true_positives = sum(1 for p, b in zip(y_pred_test, y_pred_forest) if p == 1 and b == 1)
true_negatives = sum(1 for p, b in zip(y_pred_test, y_pred_forest) if p == 0 and b == 0)
false_positives = sum(1 for p, b in zip(y_pred_test, y_pred_forest) if p == 0 and b == 1)
false_negatives = sum(1 for p, b in zip(y_pred_test, y_pred_forest) if p == 1 and b == 0)

logger.info(f"\nДетальная статистика:")
logger.info(f"Правильно определена пропаганда (True Positives): {true_positives}")
logger.info(f"Правильно определены чистые тексты (True Negatives): {true_negatives}")
logger.info(f"Ложные срабатывания (False Positives): {false_positives}")
logger.info(f"Пропущенная пропаганда (False Negatives): {false_negatives}")
