import logging
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

from pipelines.cascade_classification import CascadePropagandaPipeline
from pipelines.enhanced_cascade import EnhancedCascadePropagandaPipeline
from pipelines.improved_cascade import ImprovedCascadePropagandaPipeline
from utils.translate import translate_corpus

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Класс для оценки производительности пайплайнов пропаганды."""

    def __init__(self, results_dir: str = "../evaluation_results"):
        """
        Инициализация оценщика.

        Args:
            results_dir: Директория для сохранения результатов
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Настройка стилей графиков
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_test_data(self, dataset_path: str, test_size: float = 0.3) -> Tuple[List[str], List[int], List[List[str]]]:
        """
        Загружает и подготавливает тестовые данные из датасета.

        Args:
            dataset_path: Путь к CSV файлу с данными
            test_size: Доля данных для тестирования (0.3 = 30%)

        Returns:
            Tuple из (тексты, бинарные метки, техники)
        """
        logger.info(f"Загрузка данных из {dataset_path}")

        # Загружаем данные
        df = pd.read_csv(dataset_path)
        logger.info(f"Загружено {len(df)} записей")

        # Группируем по предложениям
        grouped_data = df.groupby(['article_id', 'sentence_id', 'sentence']).agg({
            'technique': list,
            'fragment': list
        }).reset_index()

        # Случайная выборка для тестирования
        test_indices = random.sample(range(len(grouped_data)), int(len(grouped_data) * test_size))
        test_data = grouped_data.iloc[test_indices].reset_index(drop=True)

        logger.info(f"Выбрано {len(test_data)} записей для тестирования ({test_size * 100}%)")

        # Подготавливаем данные
        texts = test_data['sentence'].tolist()
        binary_labels = []
        techniques_labels = []

        for _, row in test_data.iterrows():
            techniques = row['technique']
            if isinstance(techniques, list):
                has_propaganda = any(t != 'non-propaganda' for t in techniques)
                binary_labels.append(int(has_propaganda))
                techniques_labels.append([t for t in techniques if t != 'non-propaganda'])
            else:
                binary_labels.append(int(techniques != 'non-propaganda'))
                techniques_labels.append([techniques] if techniques != 'non-propaganda' else [])

        logger.info(f"Распределение классов: {np.bincount(binary_labels)}")

        return texts, binary_labels, techniques_labels

    def evaluate_binary_classification(self,
                                       y_true: List[int],
                                       y_pred: List[int],
                                       y_proba: List[float],
                                       pipeline_name: str) -> Dict:
        """
        Оценка бинарной классификации с визуализацией.

        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            y_proba: Вероятности класса "пропаганда"
            pipeline_name: Название пайплайна

        Returns:
            Словарь с метриками
        """
        logger.info(f"Оценка бинарной классификации для {pipeline_name}")

        # Вычисляем метрики
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # ROC AUC если есть оба класса
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        # Создаем визуализации
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Бинарная классификация - {pipeline_name}', fontsize=16, fontweight='bold')

        # 1. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Не пропаганда', 'Пропаганда'],
                    yticklabels=['Не пропаганда', 'Пропаганда'],
                    ax=axes[0, 0])
        axes[0, 0].set_title('Матрица ошибок')
        axes[0, 0].set_ylabel('Истинные метки')
        axes[0, 0].set_xlabel('Предсказанные метки')

        # 2. ROC Curve
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                            label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC кривая')
            axes[0, 1].legend(loc="lower right")
        else:
            axes[0, 1].text(0.5, 0.5, 'ROC недоступен\n(один класс)',
                            ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('ROC кривая')

        # 3. Распределение вероятностей
        propaganda_proba = [p for i, p in enumerate(y_proba) if y_true[i] == 1]
        non_propaganda_proba = [p for i, p in enumerate(y_proba) if y_true[i] == 0]

        axes[1, 0].hist(non_propaganda_proba, bins=20, alpha=0.7, label='Не пропаганда', color='blue')
        axes[1, 0].hist(propaganda_proba, bins=20, alpha=0.7, label='Пропаганда', color='red')
        axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Порог = 0.5')
        axes[1, 0].set_xlabel('Вероятность')
        axes[1, 0].set_ylabel('Количество')
        axes[1, 0].set_title('Распределение вероятностей')
        axes[1, 0].legend()

        # 4. Метрики в виде барплота
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        bars = axes[1, 1].bar(metric_names, metric_values,
                              color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        axes[1, 1].set_title('Метрики производительности')
        axes[1, 1].set_ylabel('Значение')
        axes[1, 1].set_ylim(0, 1)

        # Добавляем значения на бары
        for bar, value in zip(bars, metric_values):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/binary_{pipeline_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()

        return metrics

    def evaluate_technique_classification(self,
                                          y_true_techniques: List[List[str]],
                                          y_pred_techniques: List[List[str]],
                                          pipeline_name: str) -> Dict:
        """
        Оценка классификации техник с визуализацией.

        Args:
            y_true_techniques: Истинные техники для каждого текста
            y_pred_techniques: Предсказанные техники для каждого текста
            pipeline_name: Название пайплайна

        Returns:
            Словарь с метриками
        """
        logger.info(f"Оценка классификации техник для {pipeline_name}")

        # Собираем все уникальные техники
        all_techniques = set()
        for techniques in y_true_techniques + y_pred_techniques:
            all_techniques.update(techniques)
        all_techniques = sorted(list(all_techniques))

        if not all_techniques:
            logger.warning("Нет техник для оценки")
            return {}

        # Создаем бинарные матрицы для многолейбловой классификации
        y_true_binary = []
        y_pred_binary = []

        for true_techs, pred_techs in zip(y_true_techniques, y_pred_techniques):
            true_vector = [1 if tech in true_techs else 0 for tech in all_techniques]
            pred_vector = [1 if tech in pred_techs else 0 for tech in all_techniques]
            y_true_binary.append(true_vector)
            y_pred_binary.append(pred_vector)

        y_true_binary = np.array(y_true_binary)
        y_pred_binary = np.array(y_pred_binary)

        # Вычисляем метрики
        metrics = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision_macro': precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true_binary, y_pred_binary, average='micro', zero_division=0),
            'recall_micro': recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=0),
            'f1_micro': f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0),
        }

        # Создаем визуализации
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f'Классификация техник - {pipeline_name}', fontsize=16, fontweight='bold')

        # 1. Распределение техник (истинные vs предсказанные)
        true_counts = {}
        pred_counts = {}

        for techniques in y_true_techniques:
            for tech in techniques:
                true_counts[tech] = true_counts.get(tech, 0) + 1

        for techniques in y_pred_techniques:
            for tech in techniques:
                pred_counts[tech] = pred_counts.get(tech, 0) + 1

        techniques_for_plot = sorted(set(list(true_counts.keys()) + list(pred_counts.keys())))
        true_values = [true_counts.get(tech, 0) for tech in techniques_for_plot]
        pred_values = [pred_counts.get(tech, 0) for tech in techniques_for_plot]

        x = np.arange(len(techniques_for_plot))
        width = 0.35

        axes[0, 0].bar(x - width / 2, true_values, width, label='Истинные', alpha=0.8)
        axes[0, 0].bar(x + width / 2, pred_values, width, label='Предсказанные', alpha=0.8)
        axes[0, 0].set_xlabel('Техники')
        axes[0, 0].set_ylabel('Количество')
        axes[0, 0].set_title('Распределение техник')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(techniques_for_plot, rotation=45, ha='right')
        axes[0, 0].legend()

        # 2. F1-score для каждой техники
        f1_per_class = f1_score(y_true_binary, y_pred_binary, average=None, zero_division=0)

        axes[0, 1].bar(range(len(all_techniques)), f1_per_class, color='lightblue')
        axes[0, 1].set_xlabel('Техники')
        axes[0, 1].set_ylabel('F1-score')
        axes[0, 1].set_title('F1-score по техникам')
        axes[0, 1].set_xticks(range(len(all_techniques)))
        axes[0, 1].set_xticklabels(all_techniques, rotation=45, ha='right')
        axes[0, 1].set_ylim(0, 1)

        # 3. Тепловая карта совпадений техник
        if len(all_techniques) <= 20:  # Ограничиваем размер для читаемости
            technique_confusion = np.zeros((len(all_techniques), len(all_techniques)))

            for true_techs, pred_techs in zip(y_true_techniques, y_pred_techniques):
                for true_tech in true_techs:
                    for pred_tech in pred_techs:
                        if true_tech in all_techniques and pred_tech in all_techniques:
                            i = all_techniques.index(true_tech)
                            j = all_techniques.index(pred_tech)
                            technique_confusion[i, j] += 1

            sns.heatmap(technique_confusion, annot=True, fmt='g', cmap='Blues',
                        xticklabels=all_techniques, yticklabels=all_techniques,
                        ax=axes[1, 0])
            axes[1, 0].set_title('Матрица совпадений техник')
            axes[1, 0].set_xlabel('Предсказанные техники')
            axes[1, 0].set_ylabel('Истинные техники')
        else:
            axes[1, 0].text(0.5, 0.5, f'Слишком много техник\nдля визуализации\n({len(all_techniques)})',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Матрица совпадений техник')

        # 4. Общие метрики
        metric_names = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)',
                        'Precision (Micro)', 'Recall (Micro)', 'F1 (Micro)']
        metric_values = [metrics['accuracy'], metrics['precision_macro'], metrics['recall_macro'],
                         metrics['f1_macro'], metrics['precision_micro'], metrics['recall_micro'],
                         metrics['f1_micro']]

        bars = axes[1, 1].bar(metric_names, metric_values,
                              color=['skyblue', 'lightgreen', 'lightcoral', 'gold',
                                     'plum', 'lightpink', 'lightyellow'])
        axes[1, 1].set_title('Метрики классификации техник')
        axes[1, 1].set_ylabel('Значение')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Добавляем значения на бары
        for bar, value in zip(bars, metric_values):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/techniques_{pipeline_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()

        return metrics

    def compare_pipelines(self, results: Dict[str, Dict]) -> None:
        """
        Сравнение производительности нескольких пайплайнов.

        Args:
            results: Словарь с результатами для каждого пайплайна
        """
        logger.info("Создание сравнительного анализа пайплайнов")

        # Сравнение бинарной классификации
        binary_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Сравнение пайплайнов', fontsize=16, fontweight='bold')

        # 1. Сравнение бинарных метрик
        pipeline_names = list(results.keys())
        metric_data = {metric: [] for metric in binary_metrics}

        for pipeline in pipeline_names:
            binary_res = results[pipeline].get('binary', {})
            for metric in binary_metrics:
                metric_data[metric].append(binary_res.get(metric, 0))

        x = np.arange(len(pipeline_names))
        width = 0.15

        for i, metric in enumerate(binary_metrics):
            if any(metric_data[metric]):  # Проверяем, есть ли данные
                axes[0, 0].bar(x + i * width, metric_data[metric], width, label=metric, alpha=0.8)

        axes[0, 0].set_xlabel('Пайплайны')
        axes[0, 0].set_ylabel('Значение метрики')
        axes[0, 0].set_title('Метрики бинарной классификации')
        axes[0, 0].set_xticks(x + width * 2)
        axes[0, 0].set_xticklabels(pipeline_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)

        # 2. Сравнение F1-score для техник
        technique_f1_data = []
        for pipeline in pipeline_names:
            tech_res = results[pipeline].get('techniques', {})
            technique_f1_data.append(tech_res.get('f1_macro', 0))

        bars = axes[0, 1].bar(pipeline_names, technique_f1_data, color='lightcoral', alpha=0.7)
        axes[0, 1].set_xlabel('Пайплайны')
        axes[0, 1].set_ylabel('F1-score (Macro)')
        axes[0, 1].set_title('F1-score классификации техник')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 1)

        # Добавляем значения на бары
        for bar, value in zip(bars, technique_f1_data):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')

        # 3. Время выполнения (если доступно)
        if any('time' in results[p] for p in pipeline_names):
            times = [results[p].get('time', 0) for p in pipeline_names]
            bars = axes[1, 0].bar(pipeline_names, times, color='lightblue', alpha=0.7)
            axes[1, 0].set_xlabel('Пайплайны')
            axes[1, 0].set_ylabel('Время (секунды)')
            axes[1, 0].set_title('Время выполнения')
            axes[1, 0].tick_params(axis='x', rotation=45)

            for bar, value in zip(bars, times):
                axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                f'{value:.2f}s', ha='center', va='bottom')
        else:
            axes[1, 0].text(0.5, 0.5, 'Данные о времени\nвыполнения недоступны',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Время выполнения')

        # 4. Общий рейтинг (комбинированная метрика)
        overall_scores = []
        for pipeline in pipeline_names:
            binary_f1 = results[pipeline].get('binary', {}).get('f1', 0)
            tech_f1 = results[pipeline].get('techniques', {}).get('f1_macro', 0)
            overall_score = (binary_f1 * 0.6 + tech_f1 * 0.4)  # Взвешенная комбинация
            overall_scores.append(overall_score)

        bars = axes[1, 1].bar(pipeline_names, overall_scores, color='gold', alpha=0.7)
        axes[1, 1].set_xlabel('Пайплайны')
        axes[1, 1].set_ylabel('Общий рейтинг')
        axes[1, 1].set_title('Общая производительность\n(60% бинарная + 40% техники)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(0, 1)

        for bar, value in zip(bars, overall_scores):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def evaluate_pipeline(self, pipeline, texts: List[str], binary_labels: List[int],
                          techniques_labels: List[List[str]], pipeline_name: str) -> Dict:
        """
        Полная оценка одного пайплайна.

        Args:
            pipeline: Экземпляр пайплайна
            texts: Тексты для оценки
            binary_labels: Истинные бинарные метки
            techniques_labels: Истинные техники
            pipeline_name: Название пайплайна

        Returns:
            Словарь с результатами оценки
        """
        import time

        logger.info(f"Оценка пайплайна: {pipeline_name}")

        start_time = time.time()

        try:
            # Получаем предсказания
            results, _ = pipeline.predict(texts, print_results=False)

            # Извлекаем данные для оценки
            y_pred_binary = [1 if r['is_propaganda'] else 0 for r in results]
            y_proba = [r['propaganda_probability'] for r in results]
            y_pred_techniques = [[t['name'] for t in r['techniques']] for r in results]

            end_time = time.time()

            # Оценка бинарной классификации
            binary_metrics = self.evaluate_binary_classification(
                binary_labels, y_pred_binary, y_proba, pipeline_name
            )

            # Оценка классификации техник
            technique_metrics = self.evaluate_technique_classification(
                techniques_labels, y_pred_techniques, pipeline_name
            )

            evaluation_results = {
                'binary': binary_metrics,
                'techniques': technique_metrics,
                'time': end_time - start_time
            }

            # Сохраняем детальный отчет
            self.save_detailed_report(evaluation_results, pipeline_name,
                                      binary_labels, y_pred_binary,
                                      techniques_labels, y_pred_techniques)

            return evaluation_results

        except Exception as e:
            logger.error(f"Ошибка при оценке пайплайна {pipeline_name}: {str(e)}")
            return {'error': str(e)}

    def save_detailed_report(self, results: Dict, pipeline_name: str,
                             y_true_binary: List[int], y_pred_binary: List[int],
                             y_true_techniques: List[List[str]], y_pred_techniques: List[List[str]]) -> None:
        """Сохраняет детальный отчет в текстовый файл."""

        report_path = f'{self.results_dir}/report_{pipeline_name.lower()}.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"ДЕТАЛЬНЫЙ ОТЧЕТ ДЛЯ ПАЙПЛАЙНА: {pipeline_name}\n")
            f.write("=" * 60 + "\n\n")

            # Бинарная классификация
            f.write("БИНАРНАЯ КЛАССИФИКАЦИЯ:\n")
            f.write("-" * 30 + "\n")
            for metric, value in results.get('binary', {}).items():
                f.write(f"{metric.upper()}: {value:.4f}\n")

            f.write(f"\nClassification Report:\n")
            f.write(classification_report(y_true_binary, y_pred_binary,
                                          target_names=['Не пропаганда', 'Пропаганда']))

            # Классификация техник
            f.write(f"\n\nКЛАССИФИКАЦИЯ ТЕХНИК:\n")
            f.write("-" * 30 + "\n")
            for metric, value in results.get('techniques', {}).items():
                f.write(f"{metric.upper()}: {value:.4f}\n")

            # Время выполнения
            f.write(f"\n\nВРЕМЯ ВЫПОЛНЕНИЯ: {results.get('time', 0):.2f} секунд\n")

        logger.info(f"Детальный отчет сохранен: {report_path}")


def main():
    """Основная функция для оценки пайплайнов."""

    logger.info("Начало оценки пайплайнов пропаганды")
    logger.info("=" * 60)

    # Инициализация оценщика
    evaluator = PipelineEvaluator()

    # Загрузка тестовых данных
    dataset_path_en = "../datasets/tasks-2-3/combined_dataset.csv"
    dataset_path_ua = "../datasets/tasks-2-3/combined_dataset_ua.csv"

    texts_en, binary_labels_en, techniques_labels_en = None, None, None
    texts_ua, binary_labels_ua, techniques_labels_ua = None, None, None

    # Загружаем английский датасет
    try:
        texts_en, binary_labels_en, techniques_labels_en = evaluator.load_test_data(dataset_path_en, test_size=1)
        logger.info(f"✓ Загружен английский датасет: {len(texts_en)} текстов")
        logger.info(f"  Распределение классов: {np.bincount(binary_labels_en)}")
    except FileNotFoundError:
        logger.warning(f"✗ Английский датасет не найден: {dataset_path_en}")

    # Загружаем украинский датасет
    try:
        texts_ua, binary_labels_ua, techniques_labels_ua = evaluator.load_test_data(dataset_path_ua, test_size=1)
        logger.info(f"✓ Загружен украинский датасет: {len(texts_ua)} текстов")
        logger.info(f"  Распределение классов: {np.bincount(binary_labels_ua)}")
    except FileNotFoundError:
        logger.warning(f"✗ Украинский датасет не найден: {dataset_path_ua}")

    # Если нет ни одного датасета, создаем тестовые данные
    if texts_en is None and texts_ua is None:
        logger.warning("Оба датасета недоступны. Создание тестовых данных...")
        texts_en = [
            "The weather is sunny today with temperature reaching 25 degrees.",
            "Our country is the only force of good in the world that opposes the forces of chaos and evil!",
            "The national bank published a report on inflation for the last quarter.",
            "Only the current government is capable of bringing our country out of the long-term crisis!",
            "Scientists from the university published research on water quality.",
        ]
        binary_labels_en = [0, 1, 0, 1, 0]
        techniques_labels_en = [[], ['Flag-waving'], [], ['Loaded Language'], []]
        logger.info(f"Создано {len(texts_en)} тестовых английских текстов")

    # Конфигурации пайплайнов для тестирования
    pipelines_config = []

    # Украинские модели (используют украинские тексты)
    if texts_ua is not None:
        ukrainian_models = [
            {
                'class': EnhancedCascadePropagandaPipeline,
                'name': 'Enhanced_Cascade_UA_v1',
                'model_file': 'ecpm_ua_v1.pkl',
                'params': {
                    'model_path': "../models",
                    'model_name': "ecpm_ua_v1",
                    'use_extra_features': True,
                    'use_ukrainian': True,
                },
                'texts': texts_ua,
                'binary_labels': binary_labels_ua,
                'techniques_labels': techniques_labels_ua
            },
            {
                'class': EnhancedCascadePropagandaPipeline,
                'name': 'Enhanced_Cascade_UA_v2',
                'model_file': 'ecpm_ua_v2.pkl',
                'params': {
                    'model_path': "../models",
                    'model_name': "ecpm_ua_v2",
                    'use_extra_features': True,
                    'use_ukrainian': True,
                },
                'texts': texts_ua,
                'binary_labels': binary_labels_ua,
                'techniques_labels': techniques_labels_ua
            },
            {
                'class': EnhancedCascadePropagandaPipeline,
                'name': 'Enhanced_Cascade_UA_v3',
                'model_file': 'ecpm_ua_v3.pkl',
                'params': {
                    'model_path': "../models",
                    'model_name': "ecpm_ua_v3",
                    'use_extra_features': True,
                    'use_ukrainian': True,
                },
                'texts': texts_ua,
                'binary_labels': binary_labels_ua,
                'techniques_labels': techniques_labels_ua
            },
            {
                'class': EnhancedCascadePropagandaPipeline,
                'name': 'Enhanced_Cascade_UA_v4',
                'model_file': 'ecpm_ua_v4.pkl',
                'params': {
                    'model_path': "../models",
                    'model_name': "ecpm_ua_v4",
                    'use_extra_features': True,
                    'use_ukrainian': True,
                },
                'texts': texts_ua,
                'binary_labels': binary_labels_ua,
                'techniques_labels': techniques_labels_ua
            },
        ]

        # Проверяем доступность украинских моделей
        for config in ukrainian_models:
            model_path = f"../models/{config['model_file']}"
            if os.path.exists(model_path):
                pipelines_config.append(config)
                logger.info(f"✓ Найдена украинская модель: {config['name']}")
            else:
                logger.warning(f"✗ Украинская модель не найдена: {config['name']}")

    # Английские модели (используют английские тексты)
    # if texts_en is not None:
    #     english_models = [
    #         # Basic Cascade модели
    #         {
    #             'class': CascadePropagandaPipeline,
    #             'name': 'Basic_Cascade_v2',
    #             'model_file': 'cpm_v2.pkl',
    #             'params': {
    #                 'model_path': "../models",
    #                 'model_name': "cpm_v2",
    #             },
    #             'texts': texts_en,
    #             'binary_labels': binary_labels_en,
    #             'techniques_labels': techniques_labels_en
    #         },
    #         {
    #             'class': CascadePropagandaPipeline,
    #             'name': 'Basic_Cascade_v3',
    #             'model_file': 'cpm_v3.pkl',
    #             'params': {
    #                 'model_path': "../models",
    #                 'model_name': "cpm_v3",
    #             },
    #             'texts': texts_en,
    #             'binary_labels': binary_labels_en,
    #             'techniques_labels': techniques_labels_en
    #         },
    #         {
    #             'class': CascadePropagandaPipeline,
    #             'name': 'Basic_Cascade_v4',
    #             'model_file': 'cpm_v4.pkl',
    #             'params': {
    #                 'model_path': "../models",
    #                 'model_name': "cpm_v4",
    #             },
    #             'texts': texts_en,
    #             'binary_labels': binary_labels_en,
    #             'techniques_labels': techniques_labels_en
    #         },
    #         {
    #             'class': CascadePropagandaPipeline,
    #             'name': 'Basic_Cascade_v5',
    #             'model_file': 'cpm_v5.pkl',
    #             'params': {
    #                 'model_path': "../models",
    #                 'model_name': "cpm_v5",
    #             },
    #             'texts': texts_en,
    #             'binary_labels': binary_labels_en,
    #             'techniques_labels': techniques_labels_en
    #         },
    #         # Improved Cascade модели
    #         {
    #             'class': ImprovedCascadePropagandaPipeline,
    #             'name': 'Improved_Cascade_v1',
    #             'model_file': 'icpm_v1.pkl',
    #             'params': {
    #                 'model_path': "../models",
    #                 'model_name': "icpm_v1",
    #             },
    #             'texts': texts_en,
    #             'binary_labels': binary_labels_en,
    #             'techniques_labels': techniques_labels_en
    #         },
    #         # Enhanced Cascade модели
    #         {
    #             'class': EnhancedCascadePropagandaPipeline,
    #             'name': 'Enhanced_Cascade_v1',
    #             'model_file': 'ecpm_v1.pkl',
    #             'params': {
    #                 'model_path': "../models",
    #                 'model_name': "ecpm_v1",
    #                 'use_extra_features': True,
    #                 'use_ukrainian': False,
    #             },
    #             'texts': texts_en,
    #             'binary_labels': binary_labels_en,
    #             'techniques_labels': techniques_labels_en
    #         },
    #         {
    #             'class': EnhancedCascadePropagandaPipeline,
    #             'name': 'Enhanced_Cascade_v2',
    #             'model_file': 'ecpm_v2.pkl',
    #             'params': {
    #                 'model_path': "../models",
    #                 'model_name': "ecpm_v2",
    #                 'use_extra_features': True,
    #                 'use_ukrainian': False,
    #             },
    #             'texts': texts_en,
    #             'binary_labels': binary_labels_en,
    #             'techniques_labels': techniques_labels_en
    #         },
    #         {
    #             'class': EnhancedCascadePropagandaPipeline,
    #             'name': 'Enhanced_Cascade_v3',
    #             'model_file': 'ecpm_v3.pkl',
    #             'params': {
    #                 'model_path': "../models",
    #                 'model_name': "ecpm_v3",
    #                 'use_extra_features': True,
    #                 'use_ukrainian': False,
    #             },
    #             'texts': texts_en,
    #             'binary_labels': binary_labels_en,
    #             'techniques_labels': techniques_labels_en
    #         },
    #     ]
    #
    #     # Проверяем доступность английских моделей
    #     for config in english_models:
    #         model_path = f"../models/{config['model_file']}"
    #         if os.path.exists(model_path):
    #             pipelines_config.append(config)
    #             logger.info(f"✓ Найдена английская модель: {config['name']}")
    #         else:
    #             logger.warning(f"✗ Английская модель не найдена: {config['name']}")

    if not pipelines_config:
        logger.error("Не найдено ни одной модели для оценки!")
        logger.info("Проверьте наличие файлов моделей в папке ../models/")
        return

    logger.info(f"Будет оценено {len(pipelines_config)} пайплайнов")
    logger.info("-" * 60)

    # Результаты оценки
    all_results = {}
    successful_evaluations = 0
    failed_evaluations = 0

    # Оценка каждого пайплайна
    for i, config in enumerate(pipelines_config, 1):
        logger.info(f"\n[{i}/{len(pipelines_config)}] Оценка пайплайна: {config['name']}")
        logger.info("=" * 40)

        try:
            # Создаем экземпляр пайплайна
            logger.info("Инициализация пайплайна...")
            pipeline = config['class'](**config['params'])

            # Оценка пайплайна
            logger.info("Запуск предсказаний и оценки...")
            results = evaluator.evaluate_pipeline(
                pipeline,
                config['texts'],
                config['binary_labels'],
                config['techniques_labels'],
                config['name']
            )

            all_results[config['name']] = results

            # Выводим краткие результаты
            if 'error' not in results:
                successful_evaluations += 1
                logger.info("✓ Оценка завершена успешно!")
                logger.info(f"  Бинарная классификация:")
                logger.info(f"    Accuracy: {results['binary'].get('accuracy', 0):.3f}")
                logger.info(f"    Precision: {results['binary'].get('precision', 0):.3f}")
                logger.info(f"    Recall: {results['binary'].get('recall', 0):.3f}")
                logger.info(f"    F1-score: {results['binary'].get('f1', 0):.3f}")
                if 'roc_auc' in results['binary']:
                    logger.info(f"    ROC-AUC: {results['binary'].get('roc_auc', 0):.3f}")

                logger.info(f"  Классификация техник:")
                logger.info(f"    F1-macro: {results['techniques'].get('f1_macro', 0):.3f}")
                logger.info(f"    F1-micro: {results['techniques'].get('f1_micro', 0):.3f}")
                logger.info(f"    Precision-macro: {results['techniques'].get('precision_macro', 0):.3f}")
                logger.info(f"    Recall-macro: {results['techniques'].get('recall_macro', 0):.3f}")

                logger.info(f"  Время выполнения: {results.get('time', 0):.2f} секунд")

                # Вычисляем общий рейтинг
                overall_score = (results['binary'].get('f1', 0) * 0.6 +
                                 results['techniques'].get('f1_macro', 0) * 0.4)
                logger.info(f"  Общий рейтинг: {overall_score:.3f}/1.000")

            else:
                failed_evaluations += 1
                logger.error(f"✗ Ошибка при оценке: {results['error']}")

        except Exception as e:
            failed_evaluations += 1
            logger.error(f"✗ Критическая ошибка при оценке пайплайна {config['name']}: {str(e)}")
            all_results[config['name']] = {'error': str(e)}

    # Подводим итоги
    logger.info("\n" + "=" * 60)
    logger.info("ИТОГИ ОЦЕНКИ")
    logger.info("=" * 60)
    logger.info(f"Всего пайплайнов: {len(pipelines_config)}")
    logger.info(f"Успешно оценено: {successful_evaluations}")
    logger.info(f"Ошибок: {failed_evaluations}")

    # Создаем сравнительный анализ если есть успешные результаты
    successful_results = {name: result for name, result in all_results.items() if 'error' not in result}

    if len(successful_results) > 1:
        logger.info("\nСоздание сравнительного анализа...")
        evaluator.compare_pipelines(successful_results)
        logger.info("✓ Сравнительный анализ сохранен")
    elif len(successful_results) == 1:
        logger.info("Только один пайплайн оценен успешно - сравнительный анализ недоступен")
    else:
        logger.warning("Ни один пайплайн не был оценен успешно!")

    # Сохраняем сводный отчет
    logger.info("Сохранение сводного отчета...")
    save_summary_report(all_results, evaluator.results_dir)
    logger.info("✓ Сводный отчет сохранен")

    # Показываем топ-3 лучших пайплайна
    if successful_results:
        logger.info("\nТОП-3 ЛУЧШИХ ПАЙПЛАЙНА:")
        logger.info("-" * 40)

        # Сортируем по общему рейтингу
        ranked_results = []
        for name, result in successful_results.items():
            overall_score = (result['binary'].get('f1', 0) * 0.6 +
                             result['techniques'].get('f1_macro', 0) * 0.4)
            ranked_results.append((name, overall_score, result))

        ranked_results.sort(key=lambda x: x[1], reverse=True)

        for i, (name, score, result) in enumerate(ranked_results[:3], 1):
            logger.info(f"{i}. {name}")
            logger.info(f"   Общий рейтинг: {score:.3f}")
            logger.info(f"   Бинарная F1: {result['binary'].get('f1', 0):.3f}")
            logger.info(f"   Техniki F1: {result['techniques'].get('f1_macro', 0):.3f}")
            logger.info(f"   Время: {result.get('time', 0):.2f}s")
            if i < 3:
                logger.info("")

    # Финальная информация
    logger.info("\n" + "=" * 60)
    logger.info("ОЦЕНКА ЗАВЕРШЕНА")
    logger.info("=" * 60)
    logger.info(f"Результаты сохранены в: {evaluator.results_dir}")
    logger.info("Файлы результатов:")
    logger.info("  - summary_report.txt - сводный отчет")
    logger.info("  - comparison.png - сравнительные графики")
    logger.info("  - binary_*.png - результаты бинарной классификации")
    logger.info("  - techniques_*.png - результаты классификации техник")
    logger.info("  - report_*.txt - детальные отчеты по каждому пайплайну")

    if failed_evaluations > 0:
        logger.warning(f"\nВНИМАНИЕ: {failed_evaluations} пайплайн(ов) не удалось оценить.")
        logger.info("Проверьте логи выше для деталей ошибок.")

    logger.info("\nСпасибо за использование системы оценки пайплайнов!")

    return all_results

    # Добавляем украинский пайплайн если модель существует
    ua_model_path = "../models/ecpm_ua_v1.pkl"
    if os.path.exists(ua_model_path):
        pipelines_config.append({
            'class': EnhancedCascadePropagandaPipeline,
            'name': 'Enhanced_Cascade_UA_v1',
            'params': {
                'model_path': "../models",
                'model_name': "ecpm_ua_v1",
                'use_extra_features': True,
                'use_ukrainian': True,
            },
            'texts': texts  # Используем оригинальные тексты для украинской модели
        })

    # Результаты оценки
    all_results = {}
    successful_evaluations = 0
    failed_evaluations = 0

    # Оценка каждого пайплайна
    for i, config in enumerate(pipelines_config, 1):
        logger.info(f"\n[{i}/{len(pipelines_config)}] Оценка пайплайна: {config['name']}")
        logger.info("=" * 40)

        try:
            # Создаем экземпляр пайплайна
            logger.info("Инициализация пайплайна...")
            pipeline = config['class'](**config['params'])

            # Оценка пайплайна
            logger.info("Запуск предсказаний и оценки...")
            results = evaluator.evaluate_pipeline(
                pipeline,
                config['texts'],
                binary_labels,
                techniques_labels,
                config['name']
            )

            all_results[config['name']] = results

            # Выводим краткие результаты
            if 'error' not in results:
                successful_evaluations += 1
                logger.info("✓ Оценка завершена успешно!")
                logger.info(f"  Бинарная классификация:")
                logger.info(f"    Accuracy: {results['binary'].get('accuracy', 0):.3f}")
                logger.info(f"    Precision: {results['binary'].get('precision', 0):.3f}")
                logger.info(f"    Recall: {results['binary'].get('recall', 0):.3f}")
                logger.info(f"    F1-score: {results['binary'].get('f1', 0):.3f}")
                if 'roc_auc' in results['binary']:
                    logger.info(f"    ROC-AUC: {results['binary'].get('roc_auc', 0):.3f}")

                logger.info(f"  Классификация техник:")
                logger.info(f"    F1-macro: {results['techniques'].get('f1_macro', 0):.3f}")
                logger.info(f"    F1-micro: {results['techniques'].get('f1_micro', 0):.3f}")
                logger.info(f"    Precision-macro: {results['techniques'].get('precision_macro', 0):.3f}")
                logger.info(f"    Recall-macro: {results['techniques'].get('recall_macro', 0):.3f}")

                logger.info(f"  Время выполнения: {results.get('time', 0):.2f} секунд")

                # Вычисляем общий рейтинг
                overall_score = (results['binary'].get('f1', 0) * 0.6 +
                                 results['techniques'].get('f1_macro', 0) * 0.4)
                logger.info(f"  Общий рейтинг: {overall_score:.3f}/1.000")

            else:
                failed_evaluations += 1
                logger.error(f"✗ Ошибка при оценке: {results['error']}")

        except Exception as e:
            failed_evaluations += 1
            logger.error(f"✗ Критическая ошибка при оценке пайплайна {config['name']}: {str(e)}")
            all_results[config['name']] = {'error': str(e)}

    # Подводим итоги
    logger.info("\n" + "=" * 60)
    logger.info("ИТОГИ ОЦЕНКИ")
    logger.info("=" * 60)
    logger.info(f"Всего пайплайнов: {len(pipelines_config)}")
    logger.info(f"Успешно оценено: {successful_evaluations}")
    logger.info(f"Ошибок: {failed_evaluations}")

    # Создаем сравнительный анализ если есть успешные результаты
    successful_results = {name: result for name, result in all_results.items() if 'error' not in result}

    if len(successful_results) > 1:
        logger.info("\nСоздание сравнительного анализа...")
        evaluator.compare_pipelines(successful_results)
        logger.info("✓ Сравнительный анализ сохранен")
    elif len(successful_results) == 1:
        logger.info("Только один пайплайн оценен успешно - сравнительный анализ недоступен")
    else:
        logger.warning("Ни один пайплайн не был оценен успешно!")

    # Сохраняем сводный отчет
    logger.info("Сохранение сводного отчета...")
    save_summary_report(all_results, evaluator.results_dir)
    logger.info("✓ Сводный отчет сохранен")

    # Показываем топ-3 лучших пайплайна
    if successful_results:
        logger.info("\nТОП-3 ЛУЧШИХ ПАЙПЛАЙНА:")
        logger.info("-" * 40)

        # Сортируем по общему рейтингу
        ranked_results = []
        for name, result in successful_results.items():
            overall_score = (result['binary'].get('f1', 0) * 0.6 +
                             result['techniques'].get('f1_macro', 0) * 0.4)
            ranked_results.append((name, overall_score, result))

        ranked_results.sort(key=lambda x: x[1], reverse=True)

        for i, (name, score, result) in enumerate(ranked_results[:3], 1):
            logger.info(f"{i}. {name}")
            logger.info(f"   Общий рейтинг: {score:.3f}")
            logger.info(f"   Бинарная F1: {result['binary'].get('f1', 0):.3f}")
            logger.info(f"   Техники F1: {result['techniques'].get('f1_macro', 0):.3f}")
            logger.info(f"   Время: {result.get('time', 0):.2f}s")
            if i < 3:
                logger.info("")

    # Финальная информация
    logger.info("\n" + "=" * 60)
    logger.info("ОЦЕНКА ЗАВЕРШЕНА")
    logger.info("=" * 60)
    logger.info(f"Результаты сохранены в: {evaluator.results_dir}")
    logger.info("Файлы результатов:")
    logger.info("  - summary_report.txt - сводный отчет")
    logger.info("  - comparison.png - сравнительные графики")
    logger.info("  - binary_*.png - результаты бинарной классификации")
    logger.info("  - techniques_*.png - результаты классификации техник")
    logger.info("  - report_*.txt - детальные отчеты по каждому пайплайну")

    if failed_evaluations > 0:
        logger.warning(f"\nВНИМАНИЕ: {failed_evaluations} пайплайн(ов) не удалось оценить.")
        logger.info("Проверьте логи выше для деталей ошибок.")

    logger.info("\nСпасибо за использование системы оценки пайплайнов!")

    return all_results


def save_summary_report(results: Dict[str, Dict], results_dir: str) -> None:
    """Сохраняет сводный отчет по всем пайплайнам."""

    summary_path = f'{results_dir}/summary_report.txt'

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("СВОДНЫЙ ОТЧЕТ ПО ОЦЕНКЕ ПАЙПЛАЙНОВ\n")
        f.write("=" * 50 + "\n\n")

        # Таблица результатов
        f.write("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Пайплайн':<25} {'Bin F1':<8} {'Tech F1':<8} {'Время':<8}\n")
        f.write("-" * 50 + "\n")

        for pipeline_name, result in results.items():
            if 'error' not in result:
                bin_f1 = result.get('binary', {}).get('f1', 0)
                tech_f1 = result.get('techniques', {}).get('f1_macro', 0)
                time_taken = result.get('time', 0)
                f.write(f"{pipeline_name:<25} {bin_f1:<8.3f} {tech_f1:<8.3f} {time_taken:<8.2f}\n")
            else:
                f.write(f"{pipeline_name:<25} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8}\n")

        f.write("\n\nДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:\n")
        f.write("-" * 50 + "\n")

        for pipeline_name, result in results.items():
            f.write(f"\n{pipeline_name}:\n")
            if 'error' not in result:
                # Бинарные метрики
                f.write("  Бинарная классификация:\n")
                for metric, value in result.get('binary', {}).items():
                    f.write(f"    {metric}: {value:.4f}\n")

                # Метрики техник
                f.write("  Классификация техник:\n")
                for metric, value in result.get('techniques', {}).items():
                    f.write(f"    {metric}: {value:.4f}\n")

                f.write(f"  Время выполнения: {result.get('time', 0):.2f}s\n")
            else:
                f.write(f"  Ошибка: {result['error']}\n")

    logger.info(f"Сводный отчет сохранен: {summary_path}")


if __name__ == "__main__":
    # Устанавливаем случайное зерно для воспроизводимости
    random.seed(42)
    np.random.seed(42)

    main()
