import json
import logging
import os

from pipelines.cascade_classification import CascadePropagandaPipeline
from pipelines.enhanced_cascade import EnhancedCascadePropagandaPipeline
from pipelines.improved_cascade import ImprovedCascadePropagandaPipeline
from utils.translate import check_lang_corpus, translate_corpus


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def result_output(predict, pipeline, text, pipeline_name, idx):
    # Initialize pipeline
    logger.info(f"Running predictions on test texts...\nVersion of {pipeline_name.upper()} PIPELINE v{idx}")
    results, formatted = pipeline.predict(text, True)

    binary_scores = [1 if result['propaganda_probability'] * 100 > 50 else 0 for result in results]
    scores = [round(result['propaganda_probability'] * 100) for result in results]

    matches = sum(1 for p, b in zip(predict, binary_scores) if p == b)
    total = len(predict)

    # Вычисляем точность
    accuracy = matches / total

    logger.info(f"Всего: {binary_scores}")
    logger.info(f"Совпадений: {matches}/{total}")
    logger.info(f"Точность: {accuracy:.2%}")

    # Дополнительная статистика
    true_positives = sum(1 for p, b in zip(predict, binary_scores) if p == 1 and b == 1)
    true_negatives = sum(1 for p, b in zip(predict, binary_scores) if p == 0 and b == 0)
    false_positives = sum(1 for p, b in zip(predict, binary_scores) if p == 0 and b == 1)
    false_negatives = sum(1 for p, b in zip(predict, binary_scores) if p == 1 and b == 0)

    logger.info(f"\nДетальная статистика:")
    logger.info(f"Правильно определена пропаганда (True Positives): {true_positives}")
    logger.info(f"Правильно определены чистые тексты (True Negatives): {true_negatives}")
    logger.info(f"Ложные срабатывания (False Positives): {false_positives}")
    logger.info(f"Пропущенная пропаганда (False Negatives): {false_negatives}")


def main():
    text = [
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

    if check_lang_corpus(text):
        text = translate_corpus(text)

    predict = [0] * 30 + [1] * 30
    logger.info(predict)

    for idx in range(1, 3):
        try:
            # binary_k_range = [3, 4, 5] if idx < 5 else [3, 5, 7]
            # technique_k_range = [2, 3, 4, 5] if idx < 5 else [3 , 5]

            pipeline = EnhancedCascadePropagandaPipeline(
                model_path="../models",
                model_name=f"ecpm_v{idx}",
                batch_size=32,
                num_epochs_binary=20,
                num_epochs_technique=10,
                learning_rate=2e-5,
                warmup_steps=1000,
                max_length=512,
                class_weights=True,
                # binary_k_range=binary_k_range,
                # technique_k_range=technique_k_range,
            )

            result_output(predict, pipeline, text, "Enhanced", idx)

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    for idx in range(2, 6):
        try:
            binary_k_range = [3, 4, 5] if idx < 5 else [3, 5, 7]
            technique_k_range = [2, 3, 4, 5] if idx < 5 else [3, 5]

            pipeline = CascadePropagandaPipeline(
                model_path="../models",
                model_name=f"cpm_v{idx}",
                batch_size=32,
                num_epochs_binary=10,
                num_epochs_technique=10,
                learning_rate=2e-5,
                warmup_steps=1000,
                max_length=512,
                class_weights=True,
                binary_k_range=binary_k_range,
                technique_k_range=technique_k_range,
            )

            result_output(predict, pipeline, text, "Cascade", idx)

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    for idx in range(1, 2):
        try:
            binary_k_range = [3, 4, 5] if idx < 3 else [3, 5, 7]
            technique_k_range = [2, 3, 4, 5] if idx < 3 else [3, 5]
            pipeline = ImprovedCascadePropagandaPipeline(
                model_path="../models",
                model_name=f"icpm_v{idx}",
                batch_size=32,
                num_epochs_binary=10,
                num_epochs_technique=10,
                learning_rate=2e-5,
                warmup_steps=1000,
                max_length=512,
                class_weights=True,
                binary_k_range=binary_k_range,
                technique_k_range=technique_k_range,
            )

            result_output(predict, pipeline, text, "Improved", idx)

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise


if __name__ == '__main__':
    main()
