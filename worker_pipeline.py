import json
import logging
import os

from pipelines.cascade_classification import CascadePropagandaPipeline
from pipelines.hierarchical_classification import HierarchicalClassificationPipeline
from pipelines.improved_cascade import ImprovedCascadePropagandaPipeline


def save_config_to_json(config, filename):
    """Save configuration to JSON file"""
    os.makedirs('classification_configs', exist_ok=True)
    file_path = os.path.join('classification_configs', filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(__name__)

    text = [
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

    print(text)
    try:
        pipeline = CascadePropagandaPipeline(
            model_path="models",
            model_name=f"cpm_v5",
            batch_size=32,
            num_epochs=30,
            learning_rate=2e-5,
            warmup_steps=1000,
            max_length=512,
            class_weights=True,
        )
        # Initialize pipeline
        logger.info(f"Running predictions on test texts...\nVersion of CASCADE PIPELINE v5")
        results, formatted = pipeline.predict(text, True)

        pipeline.print_params()

        config = pipeline.get_params()
        save_config_to_json(config, f'cascade_v5_1.json')

        logger.warning(f"Params of the CASCADE PIPELINE v5\n{config}")
        logger.info(formatted)


    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


    # for idx in range(2, 6):
    #     try:
    #         pipeline = CascadePropagandaPipeline(
    #             model_path="models",
    #             model_name=f"cpm_v{idx}",
    #             batch_size=32,
    #             num_epochs=10,
    #             learning_rate=2e-5,
    #             warmup_steps=1000,
    #             max_length=512,
    #             class_weights=True,
    #         )
    #
    #         # Initialize pipeline
    #         logger.info(f"Running predictions on test texts...\nVersion of CASCADE PIPELINE v{idx}")
    #         results, formatted = pipeline.predict(text, True)
    #
    #         pipeline.print_params()
    #
    #         config = pipeline.get_params()
    #         save_config_to_json(config, f'cascade_v{idx}.json')
    #
    #         logger.warning(f"Params of the CASCADE PIPELINE v{idx}\n{config}")
    #         logger.info(formatted)
    #
    #
    #     except Exception as e:
    #         logger.error(f"Prediction failed: {str(e)}")
    #         raise
    #
    # for idx in range(1, 3):
    #     try:
    #         pipeline = ImprovedCascadePropagandaPipeline(
    #             model_path="models",
    #             model_name=f"icpm_v{idx}",
    #             batch_size=32,
    #             num_epochs=10,
    #             learning_rate=2e-5,
    #             warmup_steps=1000,
    #             max_length=512,
    #             class_weights=True,
    #         )
    #
    #         # Initialize pipeline
    #         logger.info(f"Running predictions on test texts...\nVersion of IMPROVED CASCADE PIPELINE v{idx}")
    #         results, formatted = pipeline.predict(text, True)
    #
    #         config = pipeline.get_params(detailed=True)
    #         save_config_to_json(config, f'improve_v{idx}.json')
    #
    #         logger.warning(f"Params of the IMPROVED CASCADE PIPELINE v{idx}\n{config}")
    #         logger.info(formatted)
    #
    #     except Exception as e:
    #         logger.error(f"Prediction failed: {str(e)}")
    #         raise
    #
    # try:
    #     pipeline = HierarchicalClassificationPipeline(
    #         model_path="models",
    #         model_name=f"hpm_v1",
    #         batch_size=32,
    #         num_epochs=10,
    #         learning_rate=2e-5,
    #         warmup_steps=1000,
    #         max_length=512,
    #         class_weights=True,
    #     )
    #
    #     # Initialize pipeline
    #     logger.info(f"Running predictions on test texts...\nVersion of HIERARCHICAL PIPELINE v1")
    #     results = pipeline.predict(text)
    #
    #     print("\nPrediction Results:")
    #     print("-" * 50)
    #     for text, preds, probs in zip(
    #             text,
    #             results['predictions'],
    #             results['probabilities']
    #     ):
    #         print(f"\nText: {text}")
    #         print("Detected Propaganda Techniques:")
    #         if 'non-propaganda' in preds:
    #             print("No propaganda detected")
    #         else:
    #             for technique in preds:
    #                 prob = probs.get(technique, 0.0)
    #                 print(f"  - {technique}: {prob:.3f}")
    #         print("-" * 30)
    #
    # except Exception as e:
    #     logger.error(f"Prediction failed: {str(e)}")
    #     raise


if __name__ == '__main__':
    main()
