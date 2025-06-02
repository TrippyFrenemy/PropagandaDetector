import copy
import logging
from typing import List

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from data_manipulating.feature_extractor import TextFeatureExtractor
from data_manipulating.feature_extractor_ua import TextFeatureExtractorUA
from data_manipulating.manipulate_models import load_model, save_model
from pipelines.cascade_classification import BinaryPropagandaModel, PropagandaDataset, CascadePropagandaPipeline, \
    FocalLoss, AttentionLayer, TechniquePropagandaModel
from utils.translate import check_lang_corpus, translate_corpus


# Настройка глобального логгера (root logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.getLogger("pymorphy3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class EnhancedBinaryPropagandaModel(nn.Module):
    """Enhanced binary classification model with balanced architecture."""

    def __init__(self, vocab_size, embedding_dim, num_filters, lstm_hidden, k_range, use_extra_features=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.lstm_hidden = lstm_hidden
        self.use_extra_features = use_extra_features
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Основной embedding слой
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(0.2)

        # CNN с разными размерами ядер для выявления ключевых паттернов
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    embedding_dim,
                    num_filters,
                    kernel_size=k,
                    padding='same'
                ),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            for k in k_range
        ])

        self.cnn_output_size = num_filters * len(k_range)

        # BiLSTM для анализа последовательности
        self.bilstm = nn.LSTM(
            self.cnn_output_size,
            lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )

        # Механизм внимания
        self.attention = AttentionLayer(lstm_hidden * 2, num_heads=8)

        if use_extra_features:
            self.extra_features_size = 17

            # Feature encoder с акцентом на важные признаки
            self.feature_encoder = nn.Sequential(
                nn.Linear(self.extra_features_size, lstm_hidden),
                nn.LayerNorm(lstm_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(lstm_hidden, lstm_hidden * 2)
            )

            # Комбинированный классификатор
            combined_size = lstm_hidden * 4 + lstm_hidden * 2
            self.classifier = nn.Sequential(
                nn.Linear(combined_size, lstm_hidden * 2),
                nn.LayerNorm(lstm_hidden * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(lstm_hidden * 2, lstm_hidden),
                nn.LayerNorm(lstm_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(lstm_hidden, 2)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(lstm_hidden * 4, lstm_hidden * 2),
                nn.LayerNorm(lstm_hidden * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(lstm_hidden * 2, lstm_hidden),
                nn.LayerNorm(lstm_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(lstm_hidden, 2)
            )

        # Инициализация весов
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def process_text(self, x, attention_mask=None):
        # Embedding
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        # CNN обработка
        x_cnn = x.transpose(1, 2)
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(x_cnn)
            conv_outputs.append(conv_out)
        x = torch.cat(conv_outputs, dim=1)
        x = x.transpose(1, 2)

        # BiLSTM с packed sequence
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.bilstm(packed_x)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=attention_mask.size(1)
            )
        else:
            x, _ = self.bilstm(x)

        # Attention
        if attention_mask is not None:
            x = self.attention(x, x, x, attention_mask)

        # Пулинг с разными стратегиями для лучшего захвата информации
        mean_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)

        # Взвешенная комбинация с акцентом на максимальные значения
        weighted_pool = 0.6 * max_pool + 0.4 * mean_pool
        final_features = torch.cat([weighted_pool, max_pool], dim=1)

        return final_features

    def process_extra_features(self, features_tensor):
        return self.feature_encoder(features_tensor)

    def forward(self, x, attention_mask=None, extra_features=None):
        text_features = self.process_text(x, attention_mask)

        if self.use_extra_features and extra_features is not None:
            encoded_features = self.process_extra_features(extra_features)
            combined_features = torch.cat([text_features, encoded_features], dim=1)
            return self.classifier(combined_features)

        return self.classifier(text_features)


class EnhancedPropagandaDataset(PropagandaDataset):
    """Enhanced dataset that includes additional text features."""

    def __init__(self, data, tokenizer, max_length=512, binary_only=False, use_extra_features=True, use_ukrainian=False, nlp=None):
        super().__init__(data, tokenizer, max_length, binary_only)
        self.use_extra_features = use_extra_features
        if use_extra_features:
            if use_ukrainian:
                self.feature_extractor = TextFeatureExtractorUA(nlp)
            else:
                self.feature_extractor = TextFeatureExtractor()
            self._extract_features()

    def _extract_features(self):
        """Extract additional features for all texts."""
        self.extra_features = []
        for idx in range(len(self.data)):
            text = self.data.iloc[idx]['sentence']
            _, features = self.feature_extractor.process_text_with_features(text)

            # Convert dictionary features to a flat tensor
            feature_vector = []

            # Linguistic features
            ling = features['linguistic']
            feature_vector.extend([
                ling['avg_sentence_length'],
                ling['sentence_length_std'],
                ling['avg_word_length'],
                ling['avg_syllables_per_word'],
                ling['unique_words_ratio']
            ])

            # Emotional features
            emot = features['emotional']
            feature_vector.extend([
                emot['polarity'],
                emot['subjectivity'],
                emot['exclamation_ratio'],
                emot['question_ratio'],
                emot['ellipsis_ratio']
            ])

            # Rhetorical features
            rhet = features['rhetorical']
            feature_vector.extend([
                rhet['repetition_ratio'],
                rhet['alliteration_ratio'],
                rhet['avg_sentence_similarity']
            ])

            # Statistical features
            stat = features['statistical']
            feature_vector.extend([
                stat['type_token_ratio'],
                stat['hapax_legomena_ratio'],
                stat['avg_word_frequency'],
                stat['frequency_std']
            ])

            # Convert to tensor and add to list
            self.extra_features.append(torch.tensor(feature_vector, dtype=torch.float32))

    def __getitem__(self, idx):
        # Get base components
        result = super().__getitem__(idx)

        # Add extra features if needed
        if self.use_extra_features:
            result['extra_features'] = self.extra_features[idx]  # Now returning tensor

        return result


class EnhancedCascadePropagandaPipeline(CascadePropagandaPipeline):
    """Enhanced pipeline that supports both old and new models."""

    def __init__(self, *args, use_extra_features=True, use_ukrainian=False, **kwargs):
        super().__init__(*args, use_extra_features=use_extra_features, use_ukrainian=use_ukrainian, **kwargs)
        self.use_extra_features = use_extra_features
        self.use_ukrainian = use_ukrainian
        if self.use_ukrainian:
            self.nlp = spacy.load("uk_core_news_sm")

    def create_datasets(self, data: pd.DataFrame):
        """Create datasets with support for additional features."""
        # Создаем датасеты с поддержкой дополнительных признаков
        binary_dataset = EnhancedPropagandaDataset(
            data,
            self.tokenizer,
            max_length=self.max_length,
            binary_only=True,
            use_extra_features=self.use_extra_features,
            use_ukrainian=self.use_ukrainian,
            nlp=self.nlp if hasattr(self, "nlp") else None,
        )

        propaganda_mask = data['technique'].apply(
            lambda x: isinstance(x, list) and any(t != 'non-propaganda' for t in x)
            if isinstance(x, list)
            else x != 'non-propaganda'
        )

        technique_data = data[propaganda_mask].copy()
        technique_dataset = EnhancedPropagandaDataset(
            technique_data,
            self.tokenizer,
            max_length=self.max_length,
            binary_only=False,
            use_extra_features=self.use_extra_features,
            use_ukrainian=self.use_ukrainian,
            nlp=self.nlp if hasattr(self, "nlp") else None,
        )

        self.technique_encoder = technique_dataset.technique_encoder
        return binary_dataset, technique_dataset

    def train_binary_model(self, train_loader, val_loader):
        """Train binary model with support for additional features."""
        model = EnhancedBinaryPropagandaModel(
            self.vocab_size,
            self.embedding_dim,
            self.num_filters,
            self.lstm_hidden,
            self.binary_k_range,
            use_extra_features=self.use_extra_features,
        ).to(self.device)

        # Создаем оптимизатор и планировщик
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps
        )

        # Создаем функцию потерь с весами классов
        if self.class_weights:
            all_labels = []
            for batch in train_loader:
                all_labels.extend(batch['binary_label'].cpu().numpy())
            all_labels = np.array(all_labels)

            class_weights = self._calculate_class_weights(all_labels)
            criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        else:
            criterion = FocalLoss()

        criterion = criterion.to(self.device)

        # Обучение
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0

        for epoch in range(self.num_epochs_binary):
            # Обучение
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []

            for batch in tqdm(train_loader, desc=f"Training binary model - Epoch {epoch + 1}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()
                # Добавляем extra_features в forward pass
                extra_features = batch.get('extra_features')
                outputs = model(batch['input_ids'], batch['attention_mask'], extra_features)
                loss = criterion(outputs, batch['binary_label'])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                train_labels.extend(batch['binary_label'].cpu().numpy())

            # Валидация
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # Добавляем extra_features в forward pass
                    extra_features = batch.get('extra_features')
                    outputs = model(batch['input_ids'], batch['attention_mask'], extra_features)
                    loss = criterion(outputs, batch['binary_label'])

                    val_loss += loss.item()
                    val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    val_labels.extend(batch['binary_label'].cpu().numpy())

            # Вычисляем метрики
            train_metrics = {
                'loss': train_loss / len(train_loader),
                'f1': f1_score(train_labels, train_preds, average='weighted')
            }

            val_metrics = {
                'loss': val_loss / len(val_loader),
                'f1': f1_score(val_labels, val_preds, average='weighted')
            }

            # Логируем результаты
            logger.info(f"\nEpoch {epoch + 1}")
            logger.info("Training metrics:")
            for k, v in train_metrics.items():
                logger.info(f"{k}: {v:.4f}")

            logger.info("\nValidation metrics:")
            for k, v in val_metrics.items():
                logger.info(f"{k}: {v:.4f}")

            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model = copy.deepcopy(model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping triggered")
                    break

        return best_model

    def predict(self, texts: List[str], print_results: bool = True, use_translate: bool = False):
        """Predict with support for both old and enhanced models."""
        try:
            # Загружаем модели если их нет
            if not hasattr(self, 'binary_model') or not hasattr(self, 'technique_model'):
                self._load_models()

            self.binary_model.eval()
            self.technique_model.eval()

            text_base = texts

            if not self.use_ukrainian and use_translate:
                texts = translate_corpus(texts)

            # Проверяем тип модели и соответственно создаем датасет
            is_enhanced = isinstance(self.binary_model, EnhancedBinaryPropagandaModel)

            dataset = EnhancedPropagandaDataset(
                pd.DataFrame({
                    'sentence': texts,
                    'technique': ['non-propaganda'] * len(texts),
                    'fragment': [None] * len(texts)
                }),
                self.tokenizer,
                max_length=self.max_length,
                binary_only=False,
                use_extra_features=is_enhanced,
                use_ukrainian=self.use_ukrainian,
                nlp=self.nlp if hasattr(self, "nlp") else None,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True
            )

            results = []

            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # Бинарная классификация с учетом extra_features
                    extra_features = batch.get('extra_features') if is_enhanced else None
                    binary_outputs = self.binary_model(
                        batch['input_ids'],
                        batch['attention_mask'],
                        extra_features
                    )
                    binary_probs = torch.softmax(binary_outputs, dim=1)

                    # Пороги уверенности
                    high_confidence = binary_probs[:, 1] > 0.7
                    medium_confidence = (binary_probs[:, 1] > 0.6) & (~high_confidence)
                    low_confidence = (binary_probs[:, 1] > 0.5) & (~high_confidence) & (~medium_confidence)

                    is_propaganda = high_confidence | medium_confidence | low_confidence

                    # Остальной код предсказания как в оригинальном методе
                    propaganda_indices = is_propaganda.nonzero().squeeze().cpu()
                    if propaganda_indices.dim() == 0 and propaganda_indices.nelement() > 0:
                        propaganda_indices = [propaganda_indices.item()]
                    elif propaganda_indices.dim() == 1:
                        propaganda_indices = propaganda_indices.tolist()
                    else:
                        propaganda_indices = []

                    batch_results = []
                    for idx, prob in enumerate(binary_probs[:, 1]):
                        result = {
                            'is_propaganda': False,
                            'propaganda_probability': prob.item(),
                            'confidence_level': 'none',
                            'techniques': []
                        }

                        if high_confidence[idx]:
                            result['confidence_level'] = 'high'
                            result['is_propaganda'] = True
                        elif medium_confidence[idx]:
                            result['confidence_level'] = 'medium'
                            result['is_propaganda'] = True
                        elif low_confidence[idx]:
                            result['confidence_level'] = 'low'
                            result['is_propaganda'] = True

                        batch_results.append(result)

                    if propaganda_indices:
                        fragment_weights = self._calculate_fragment_weights(
                            batch['input_ids'],
                            batch['attention_mask']
                        ).to(self.device)

                        technique_outputs = self.technique_model(
                            batch['input_ids'],
                            batch['attention_mask'],
                            fragment_weights
                        )
                        technique_probs = torch.sigmoid(technique_outputs)

                        for idx in propaganda_indices:
                            probs = technique_probs[idx]
                            techniques = []

                            for technique_idx, prob in enumerate(probs):
                                if prob > 0.1:
                                    technique = self.technique_encoder.inverse_transform([technique_idx])[0]
                                    if technique != 'non-propaganda':
                                        confidence = (
                                            'high' if prob > 0.6
                                            else 'medium' if prob > 0.4
                                            else 'low'
                                        )
                                        techniques.append({
                                            'name': technique,
                                            'probability': prob.item(),
                                            'confidence': confidence
                                        })

                            techniques.sort(key=lambda x: x['probability'], reverse=True)
                            batch_results[idx]['techniques'] = techniques
                            batch_results[idx]['num_techniques'] = len(techniques)

                    results.extend(batch_results)

            formatted_output = self._print_prediction_results(text_base, results) if print_results else ""
            return results, formatted_output

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _save_models(self):
        """Save models and all parameters."""
        state = {
            # Model states
            'binary_model_state': self.binary_model.state_dict(),
            'technique_model_state': self.technique_model.state_dict(),
            'technique_encoder': self.technique_encoder,
            'tokenizer': self.tokenizer,

            # Model architecture parameters
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_filters': self.num_filters,
            'lstm_hidden': self.lstm_hidden,
            'max_length': self.max_length,
            'binary_k_range': self.binary_k_range,
            'technique_k_range': self.technique_k_range,
            'use_extra_features': self.use_extra_features,
            'use_ukrainian': self.use_ukrainian,

            # Tokenizer info
            'tokenizer_name': 'bert-base-uncased' if not self.use_ukrainian else 'google-bert/bert-base-multilingual-cased',

            # Training parameters
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'class_weights': self.class_weights,
            'model_type': 'enhanced'  # Add model type identifier
        }
        save_model(state, f"{self.model_path}/{self.model_name}")

    def _load_models(self):
        """Load models and restore parameters."""
        checkpoint = load_model(f"{self.model_path}/{self.model_name}")

        # Restore parameters
        self.vocab_size = checkpoint.get('vocab_size', self.vocab_size)
        self.embedding_dim = checkpoint.get('embedding_dim', self.embedding_dim)
        self.num_filters = checkpoint.get('num_filters', self.num_filters)
        self.lstm_hidden = checkpoint.get('lstm_hidden', self.lstm_hidden)
        self.max_length = checkpoint.get('max_length', self.max_length)
        self.batch_size = checkpoint.get('batch_size', self.batch_size)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.warmup_steps = checkpoint.get('warmup_steps', self.warmup_steps)
        self.class_weights = checkpoint.get('class_weights', self.class_weights)
        self.use_extra_features = checkpoint.get('use_extra_features', True)
        self.use_ukrainian = checkpoint.get('use_ukrainian', False)


        self.binary_k_range = checkpoint.get('binary_k_range', [3, 4, 5])
        self.technique_k_range = checkpoint.get('technique_k_range', [2, 3, 4, 5])

        self.tokenizer = checkpoint.get('tokenizer')

        # Determine model type and initialize appropriate model
        model_type = checkpoint.get('model_type', 'standard')

        if model_type == 'enhanced':
            # Load enhanced binary model
            self.binary_model = EnhancedBinaryPropagandaModel(
                self.vocab_size,
                self.embedding_dim,
                self.num_filters,
                self.lstm_hidden,
                self.binary_k_range,
                use_extra_features=self.use_extra_features
            ).to(self.device)
        else:
            # Load standard binary model
            self.binary_model = BinaryPropagandaModel(
                self.vocab_size,
                self.embedding_dim,
                self.num_filters,
                self.lstm_hidden,
                self.binary_k_range
            ).to(self.device)

        # Load technique model
        num_classes = len(checkpoint['technique_encoder'].classes_)
        self.technique_model = TechniquePropagandaModel(
            self.vocab_size,
            self.embedding_dim,
            self.num_filters,
            self.lstm_hidden,
            num_classes,
            self.technique_k_range
        ).to(self.device)

        # Load state dictionaries
        try:
            self.binary_model.load_state_dict(checkpoint['binary_model_state'])
            self.technique_model.load_state_dict(checkpoint['technique_model_state'])
        except Exception as e:
            raise RuntimeError(f"Error loading model state dict: {str(e)}")

        # Load encoder
        self.technique_encoder = checkpoint['technique_encoder']

        # Set models to eval mode
        self.binary_model.eval()
        self.technique_model.eval()


if __name__ == "__main__":
    # # Инициализация пайплайна
    # pipeline = EnhancedCascadePropagandaPipeline(
    #     model_path="../models",
    #     model_name="ecpm_ua_v1",
    #     batch_size=32,
    #     num_epochs_binary=10,
    #     num_epochs_technique=10,
    #     learning_rate=2e-5,
    #     warmup_steps=1000,
    #     max_length=512,
    #     class_weights=True,
    #     binary_k_range=[2, 3, 4, 5, 6, 7],
    #     technique_k_range=[3, 4, 5],
    #     dataset_distribution=0.9,
    #     use_ukrainian=True
    # )
    #
    # # Обучение и оценка
    # try:
    #     logger.info("Starting training and evaluation...")
    #     metrics = pipeline.train_and_evaluate(
    #         data_path="../datasets/tasks-2-3/combined_dataset_ua.csv"
    #     )
    #
    #     logger.info("\nFinal Evaluation Results:")
    #     logger.info("\nBinary Classification Metrics:")
    #     for metric, value in metrics['binary_metrics'].items():
    #         logger.info(f"{metric}: {value:.4f}")
    #
    #     logger.info("\nTechnique Classification Metrics:")
    #     for metric, value in metrics['technique_metrics'].items():
    #         logger.info(f"{metric}: {value:.4f}")
    #
    # except Exception as e:
    #     logger.error(f"Pipeline execution failed: {str(e)}")

    # del pipeline, metrics
    #
    # pipeline = EnhancedCascadePropagandaPipeline(
    #     model_path="../models",
    #     model_name="ecpm_ua_v2",
    #     batch_size=32,
    #     num_epochs_binary=12,
    #     num_epochs_technique=12,
    #     learning_rate=2e-7,
    #     warmup_steps=4000,
    #     max_length=2048,
    #     class_weights=True,
    #     binary_k_range=[3, 5, 7, 9],
    #     technique_k_range=[2, 3, 4, 5, 6, 7],
    #     dataset_distribution=0.95,
    #     use_ukrainian=True
    # )
    #
    # # Обучение и оценка
    # try:
    #     logger.info("Starting training and evaluation...")
    #     metrics = pipeline.train_and_evaluate(
    #         data_path="../datasets/tasks-2-3/combined_dataset_ua.csv"
    #     )
    #
    #     logger.info("\nFinal Evaluation Results:")
    #     logger.info("\nBinary Classification Metrics:")
    #     for metric, value in metrics['binary_metrics'].items():
    #         logger.info(f"{metric}: {value:.4f}")
    #
    #     logger.info("\nTechnique Classification Metrics:")
    #     for metric, value in metrics['technique_metrics'].items():
    #         logger.info(f"{metric}: {value:.4f}")
    #
    # except Exception as e:
    #     logger.error(f"Pipeline execution failed: {str(e)}")
    #
    # from evaluate_pipeline import main as main_evaluate_pipeline
    # main_evaluate_pipeline()
    #

    # pipeline = EnhancedCascadePropagandaPipeline(
    #     model_path="../models",
    #     model_name="ecpm_ua_v3",
    #     batch_size=64,
    #     num_epochs_binary=15,
    #     num_epochs_technique=10,
    #     learning_rate=2e-7,
    #     warmup_steps=5000,
    #     max_length=1024,
    #     class_weights=True,
    #     binary_k_range=[3, 5, 7, 9, 11],
    #     technique_k_range=[2, 3, 4, 5],
    #     dataset_distribution=0.99,
    #     use_ukrainian=True
    # )
    #
    # # Обучение и оценка
    # try:
    #     logger.info("Starting training and evaluation...")
    #     metrics = pipeline.train_and_evaluate(
    #         data_path="../datasets/tasks-2-3/combined_dataset_ua.csv"
    #     )
    #
    #     logger.info("\nFinal Evaluation Results:")
    #     logger.info("\nBinary Classification Metrics:")
    #     for metric, value in metrics['binary_metrics'].items():
    #         logger.info(f"{metric}: {value:.4f}")
    #
    #     logger.info("\nTechnique Classification Metrics:")
    #     for metric, value in metrics['technique_metrics'].items():
    #         logger.info(f"{metric}: {value:.4f}")
    #
    # except Exception as e:
    #     logger.error(f"Pipeline execution failed: {str(e)}")

    # pipeline = EnhancedCascadePropagandaPipeline(
    #     model_path="../models",
    #     model_name="ecpm_ua_v3",
    #     batch_size=32,
    #     num_epochs_binary=40,
    #     num_epochs_technique=10,
    #     learning_rate=2e-5,
    #     warmup_steps=1000,
    #     max_length=512,
    #     class_weights=True,
    #     binary_k_range=[2, 3, 4, 5, 6],
    #     technique_k_range=[3, 5, 7],
    #     dataset_distribution=0.95,
    #     use_ukrainian=True
    # )
    #
    # # Обучение и оценка
    # try:
    #     logger.info("Starting training and evaluation...")
    #     metrics = pipeline.train_and_evaluate(
    #         data_path="../datasets/tasks-2-3/combined_dataset_ua.csv"
    #     )
    #
    #     logger.info("\nFinal Evaluation Results:")
    #     logger.info("\nBinary Classification Metrics:")
    #     for metric, value in metrics['binary_metrics'].items():
    #         logger.info(f"{metric}: {value:.4f}")
    #
    #     logger.info("\nTechnique Classification Metrics:")
    #     for metric, value in metrics['technique_metrics'].items():
    #         logger.info(f"{metric}: {value:.4f}")
    #
    # except Exception as e:
    #     logger.error(f"Pipeline execution failed: {str(e)}")

    pipeline = EnhancedCascadePropagandaPipeline(
        model_path="../models",
        model_name="ecpm_ua_v4",
        batch_size=32,
        num_epochs_binary=100,
        num_epochs_technique=10,
        learning_rate=2e-5,
        warmup_steps=1000,
        max_length=512,
        class_weights=True,
        binary_k_range=[1, 3, 5, 7, 9],
        technique_k_range=[3, 5, 7],
        dataset_distribution=0.95,
        use_ukrainian=True,
    )

    try:
        logger.info("Starting training and evaluation...")
        metrics = pipeline.train_and_evaluate(
            data_path="../datasets/tasks-2-3/combined_dataset_ua.csv"
        )

        logger.info("\nFinal Evaluation Results:")
        logger.info("\nBinary Classification Metrics:")
        for metric, value in metrics['binary_metrics'].items():
            logger.info(f"{metric}: {value:.4f}")

        logger.info("\nTechnique Classification Metrics:")
        for metric, value in metrics['technique_metrics'].items():
            logger.info(f"{metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")

    from evaluate_pipeline import main as main_evaluate_pipeline
    main_evaluate_pipeline()
