import os.path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tqdm import tqdm

from data_manipulating.manipulate_models import save_model, save_vectorizer, load_model, load_vectorizer
from data_manipulating.feature_extractor import TextFeatureExtractor


class ClassificationPipeline:
    def __init__(self, model_path="models", model_name="enhanced_model"):
        self.model_path = model_path
        self.model_name = model_name

        self.feature_extractor = TextFeatureExtractor()

        # Настройка TF-IDF с оптимальными параметрами
        self.vectorizer = TfidfVectorizer(
            max_features=500,  # Уменьшаем количество признаков
            norm='l2',
            sublinear_tf=True,
        )

        self.scaler = StandardScaler()

        # Словарь для маппинга названий признаков
        self.advanced_features_mapping = {
            'linguistic_avg_sentence_length': 'Средняя длина предложения',
            'linguistic_sentence_length_std': 'Стандартное отклонение длины предложений',
            'linguistic_avg_word_length': 'Средняя длина слова',
            'linguistic_avg_syllables_per_word': 'Среднее число слогов в слове',
            'linguistic_unique_words_ratio': 'Доля уникальных слов',
            'emotional_polarity': 'Полярность текста',
            'emotional_subjectivity': 'Субъективность текста',
            'emotional_exclamation_ratio': 'Доля восклицательных знаков',
            'emotional_question_ratio': 'Доля вопросительных знаков',
            'emotional_ellipsis_ratio': 'Доля многоточий',
            'rhetorical_repetition_ratio': 'Доля повторов',
            'rhetorical_alliteration_ratio': 'Доля аллитераций',
            'rhetorical_avg_sentence_similarity': 'Схожесть предложений',
            'statistical_type_token_ratio': 'Лексическое разнообразие',
            'statistical_hapax_legomena_ratio': 'Доля редких слов',
            'statistical_avg_word_frequency': 'Средняя частота слов',
            'statistical_frequency_std': 'Разброс частот слов'
        }

        # Определение моделей и их параметров
        self.models_params = {
            'logistic': (
                LogisticRegression(),
                {
                    'solver': ['saga'],
                    'penalty': ['elasticnet'],
                    'l1_ratio': [0.1, 0.5],
                    'C': [0.1, 1.0, 10.0],
                    'max_iter': [7000, 10000],
                    'tol': [1e-4, 1e-5],
                    'random_state': [42]
                }
            ),
            'forest': (
                RandomForestClassifier(),
                {
                    'n_estimators': [100, 200, 300],
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_features': ['sqrt', 'log2', None],
                    'max_leaf_nodes': [10, 50, 100, 200, 500, None],
                    'max_depth': [100, 200, None],
                    'random_state': [42]
                }
            ),
            'knn': (
                KNeighborsClassifier(),
                {
                    'n_neighbors': [3, 5, 10],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                }
            )
        }

    def analyze_feature_importance(self, model, feature_names):
        """Анализ важности признаков"""
        importance_dict = {'feature': [], 'importance': [], 'type': []}

        if hasattr(model, 'coef_'):  # Для логистической регрессии
            importance = np.abs(model.coef_[0])
        elif hasattr(model, 'feature_importances_'):  # Для случайного леса
            importance = model.feature_importances_
        else:
            print("Model doesn't support feature importance analysis")
            return

        # Сбор данных о важности признаков
        for idx, imp in enumerate(importance):
            if idx < len(self.feature_names):  # TF-IDF признаки
                name = self.feature_names[idx]
                feat_type = 'TF-IDF'
            else:  # Продвинутые признаки
                adv_idx = idx - len(self.feature_names)
                if adv_idx < len(self.advanced_feature_names):
                    original_name = self.advanced_feature_names[adv_idx]
                    name = self.advanced_features_mapping.get(original_name, original_name)
                else:
                    name = f"unknown_feature_{adv_idx}"
                feat_type = 'Advanced'

            importance_dict['feature'].append(name)
            importance_dict['importance'].append(imp)
            importance_dict['type'].append(feat_type)

        imp_df = pd.DataFrame(importance_dict)

        # Вывод анализа
        print("\nFeature Importance Analysis:")
        for feat_type in ['TF-IDF', 'Advanced']:
            type_data = imp_df[imp_df['type'] == feat_type]
            total_importance = type_data['importance'].sum()
            avg_importance = type_data['importance'].mean()
            print(f"\n{feat_type} Features:")
            print(f"Total Importance: {total_importance:.4f}")
            print(f"Average Importance: {avg_importance:.4f}")
            print(f"\nTop 5 Most Important {feat_type} Features:")
            print(type_data.nlargest(5, 'importance')[['feature', 'importance']])

    def _extract_features(self, texts):
        """Извлечение признаков из текстов"""
        print("Extracting features...")

        processed_texts = []
        feature_dicts = []

        for text in tqdm(texts, desc="Processing texts"):
            processed_text, features = self.feature_extractor.process_text_with_features(text)
            processed_texts.append(processed_text)
            feature_dicts.append(features)

        advanced_features = []
        self.advanced_feature_names = []  # Сохраняем порядок признаков

        for feature_dict in feature_dicts:
            flat_features = {}
            # Первый проход - собираем все возможные имена признаков
            if not self.advanced_feature_names:
                for category, features in feature_dict.items():
                    for feature_name in features.keys():
                        full_name = f"{category}_{feature_name}"
                        if full_name in self.advanced_features_mapping:
                            self.advanced_feature_names.append(full_name)

            # Заполняем значения с увеличенным весом
            for category, features in feature_dict.items():
                for feature_name, value in features.items():
                    full_name = f"{category}_{feature_name}"
                    flat_features[f"adv_{full_name}"] = value * 20.0  # Увеличенный вес
            advanced_features.append(flat_features)

        # Создаем DataFrame
        df = pd.DataFrame({
            'text': processed_texts
        })

        # Добавляем продвинутые признаки
        advanced_features_df = pd.DataFrame(advanced_features)
        for col in advanced_features_df.columns:
            df[col] = advanced_features_df[col]

        # Обработка признаков
        if not hasattr(self, 'is_fitted'):
            # Сначала TF-IDF
            tfidf_features = self.vectorizer.fit_transform(df['text'])
            self.feature_names = list(self.vectorizer.get_feature_names_out())

            # Настройка ColumnTransformer
            advanced_columns = [col for col in df.columns if col.startswith('adv_')]
            self.feature_processor = ColumnTransformer([
                ('tfidf', self.vectorizer, 'text'),
                ('advanced', self.scaler, advanced_columns)
            ])

            features = self.feature_processor.fit_transform(df)
            self.is_fitted = True
        else:
            # Если feature_processor уже загружен, используем его
            if hasattr(self, 'feature_processor'):
                features = self.feature_processor.transform(df)
            else:
                raise ValueError("Feature processor not loaded")

        return features

    def train_and_evaluate(self, data_path):
        """Обучение и оценка моделей"""
        if self.model_path == "models_back":
            raise Exception("Переписываешь резервную копию")

        print("Loading data...")
        data = pd.read_csv(data_path, sep=";", encoding="utf-8")
        data['Class'] = data['Class'].map({'propaganda': 1, 'non-propaganda': 0})

        # Используем весь датасет для обучения
        X_train = data["Text"]
        y_train = data["Class"]

        # Создаем тестовую выборку как случайное подмножество
        test_indices = np.random.RandomState(20).choice(
            len(data),
            size=int(len(data) * 0.1),
            replace=False
        )
        X_test = data["Text"].iloc[test_indices]
        y_test = data["Class"].iloc[test_indices]

        print("Extracting features for training set...")
        X_train_features = self._extract_features(X_train)
        print("Extracting features for test set...")
        X_test_features = self._extract_features(X_test)

        best_models = {}

        for model_name, (model, param_grid) in self.models_params.items():
            print(f"\nTraining {model_name}...")

            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=5,
                n_jobs=-1,
                scoring='f1'
            )

            grid_search.fit(X_train_features, y_train)

            best_model = grid_search.best_estimator_
            best_models[model_name] = best_model

            save_model(
                best_model,
                f"{self.model_path}/{model_name}_{self.model_name}"
            )

            y_pred = best_model.predict(X_test_features)
            print(f"\n{model_name.capitalize()} Results:")
            print(classification_report(y_test, y_pred))
            print(f"Best parameters:")
            print(grid_search.best_params_)

            if model_name in ['logistic', 'forest']:
                print(f"\nAnalyzing feature importance for {model_name}...")
                self.analyze_feature_importance(best_model, self.feature_names)

        # Сохранение компонентов
        save_vectorizer(self.vectorizer, f"{self.model_path}/tfidf_{self.model_name}")
        save_vectorizer(self.scaler, f"{self.model_path}/scaler_{self.model_name}")
        save_vectorizer(self.feature_processor, f"{self.model_path}/feature_processor_{self.model_name}")

        # Сохраняем имена признаков для последующего анализа
        feature_names_path = f"{self.model_path}/feature_names_{self.model_name}.npy"
        np.save(feature_names_path, self.feature_names)

        return best_models

    def predict(self, texts, model_name):
        """Предсказание для новых текстов"""
        if not hasattr(self, 'is_fitted'):
            # Загрузка feature processor
            feature_processor_path = f"{self.model_path}/feature_processor_{self.model_name}.joblib"
            if not os.path.exists(feature_processor_path):
                raise ValueError(f"Feature processor not found at {feature_processor_path}")
            print(f"Loading feature processor from {feature_processor_path}")
            self.feature_processor = load_vectorizer(f"{self.model_path}/feature_processor_{self.model_name}")

            # Загрузка имен признаков для анализа важности
            feature_names_path = f"{self.model_path}/feature_names_{self.model_name}.npy"
            if not os.path.exists(feature_names_path):
                raise ValueError(f"Feature names not found at {feature_names_path}")
            self.feature_names = np.load(feature_names_path, allow_pickle=True)

            # Для анализа важности признаков нам нужен отдельный векторайзер
            tfidf_path = f"{self.model_path}/tfidf_{self.model_name}.joblib"
            if not os.path.exists(tfidf_path):
                raise ValueError(f"TF-IDF vectorizer not found at {tfidf_path}")
            print(f"Loading TF-IDF vectorizer from {tfidf_path}")
            self.vectorizer = load_vectorizer(f"{self.model_path}/tfidf_{self.model_name}")

            # Для анализа важности признаков нам нужен отдельный скейлер
            scaler_path = f"{self.model_path}/scaler_{self.model_name}.joblib"
            if not os.path.exists(scaler_path):
                raise ValueError(f"Scaler not found at {scaler_path}")
            print(f"Loading scaler from {scaler_path}")
            self.scaler = load_vectorizer(f"{self.model_path}/scaler_{self.model_name}")

            self.is_fitted = True

        # Extract features using loaded components
        features = self._extract_features(texts)

        # Load model
        model_path = f"{self.model_path}/{model_name}_{self.model_name}"
        if not os.path.exists(f"{model_path}.pkl"):
            raise ValueError(f"Model not found at {model_path}")

        print(f"Loading model from {model_path}")
        model = load_model(model_path)

        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        return predictions, probabilities


# Example usage:
if __name__ == "__main__":
    import time
    from datetime import timedelta

    start_time = time.time()
    print("\nStarting model training... ")

    pipeline = ClassificationPipeline(
        model_path="../models",
        model_name="propaganda_detector_v2"
    )

    best_models = pipeline.train_and_evaluate(
        data_path="../datasets/propaganda_on_sentence_level.csv"
    )

    end_time = time.time()
    training_time = end_time - start_time
    training_duration = str(timedelta(seconds=int(training_time)))
    print(f"\nTotal training time: {training_duration} (HH:MM:SS)")

    # Test predictions
    test_texts = [
        "Orange is orange.",
        "Why We should support our great nation.",
        "Our country stands on the threshold of the greatest achievements in history. While malicious critics try to "
        "undermine our progress, we must remain united and steadfast. Our leader, continuing a long tradition of "
        "outstanding government, strives tirelessly for the benefit of every citizen.",
        "Our leader is the best of the best all over the world.",
        "The Everyday Struggle of Ordinary People Against Feminist Influence.".lower(),
        "In these difficult times, when national and international enemies threaten our way of life, our indomitable "
        "leader stands guard over stability and development. Thanks to his wise and decisive actions, our country's "
        "economy is experiencing unprecedented growth.",
        "We stand on the threshold of the greatest challenges of our time. Our society, our traditions and our "
        "history are under attack from those who want to see us weakened and divided. Do not be deceived: our enemies "
        "are both internal traitors and external aggressors.",
        "Pamela Geller and Robert Spencer co founded anti Muslim group Stop Islamization of America.",
        "Children loves chocolate.",
        "Little Kate always looks forward to the weekend to go for a walk in the nearby park.",
        "Make sure you use the correct instances of models and vectorizers for testing that match the number of "
        "features used in the trained models.",
        "Save the crowbar from the ship to blow up the Japanese.",
        "Destroy this crazy beast. Enroll in the US Army.",
        "Fulfill the five-year plan not in five years, but in four.",
        "Death to capital or death under the heel of capital!",
        "Workers of the world, unite! For a worldwide October!",
        "Why we fight - for our children`s bread."
    ]

    for model_name in ['logistic', 'forest', 'knn']:
        print(f"\nPredictions using {model_name.capitalize()}:")
        predictions, probabilities = pipeline.predict(test_texts, model_name)

        for text, pred, prob in zip(test_texts, predictions, probabilities):
            print(f"\nText: {text}")
            print(f"Prediction: {'Propaganda' if pred == 1 else 'Non-propaganda'}")
            print(f"Probability of non-propaganda: {prob[0]:.2f}")
            print(f"Probability of propaganda: {prob[1]:.2f}")

    end_time = time.time()
    total_time = end_time - start_time
    total_duration = str(timedelta(seconds=int(total_time)))
    print(f"\nTotal execution time: {total_duration} (HH:MM:SS)")
