import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report, confusion_matrix
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_manipulating.metrics import calculate_metrics
from data_manipulating.manipulate_models import save_model, load_model
from pipelines.classification import ClassificationPipeline
from utils.google_translate import check_lang_corpus, translate_corpus

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss для работы с несбалансированными классами."""

    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

        # Если alpha передан как число, преобразуем его в тензор
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, alpha])

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Предсказания модели [batch_size, num_classes]
            targets: Целевые метки [batch_size]

        Returns:
            Значение функции потерь
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Обработка alpha
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            # Если alpha - тензор для разных классов
            if self.alpha.size(0) == inputs.size(1):
                batch_alpha = self.alpha.gather(0, targets)
            else:
                batch_alpha = self.alpha[0]  # используем одинаковый вес
        else:
            batch_alpha = 1.0

        # Вычисляем focal loss
        focal_loss = batch_alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()

class AttentionLayer(nn.Module):
    """Слой многоголового внимания."""

    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, query, key, value, mask=None):
        # Преобразуем входные тензоры в нужный формат
        # MultiheadAttention ожидает формат [seq_len, batch, hidden]
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        # Преобразуем маску в правильный формат и тип
        if mask is not None:
            # Инвертируем маску, так как MultiheadAttention ожидает True для игнорируемых позиций
            mask = ~mask.bool()

        # Применяем внимание
        output, _ = self.attention(
            query,
            key,
            value,
            key_padding_mask=mask
        )

        # Возвращаем в исходный формат [batch, seq_len, hidden]
        return output.transpose(0, 1)


class HierarchicalModel(nn.Module):
    """Иерархическая нейронная модель с CNN и BiLSTM."""

    def __init__(self, vocab_size, embedding_dim, num_filters, lstm_hidden, num_classes):
        super().__init__()

        # Сохраняем размерности
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.lstm_hidden = lstm_hidden

        # Слой эмбеддингов
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # CNN слои для извлечения локальных признаков
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                embedding_dim,
                num_filters,
                kernel_size=k,
                padding='same'
            )
            for k in [3, 4, 5]
        ])

        # Вычисляем размер входа для BiLSTM
        cnn_output_size = num_filters * len(self.conv_layers)

        # BiLSTM для обработки последовательности
        self.bilstm = nn.LSTM(
            cnn_output_size,
            lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )

        # Механизмы внимания
        self.word_attention = AttentionLayer(lstm_hidden * 2)
        self.sentence_attention = AttentionLayer(lstm_hidden * 2)

        # Размер контекстного вектора
        # word_context (lstm_hidden * 2) + fragment_context (lstm_hidden * 2) +
        # sentence_context (lstm_hidden * 2) + max_pooling (lstm_hidden * 2)
        context_size = lstm_hidden * 8

        # Головы классификации с исправленными размерностями
        self.classifier = nn.Sequential(
            nn.Linear(context_size, context_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),  # Увеличим dropout
            nn.BatchNorm1d(context_size // 2),  # Добавим batch normalization
            nn.Linear(context_size // 2, context_size // 4),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        classifier_output_size = context_size // 4
        self.binary_head = nn.Linear(classifier_output_size, 2)
        self.technique_head = nn.Linear(classifier_output_size, num_classes)

        # Нормализация слоев
        self.layer_norm1 = nn.LayerNorm(cnn_output_size)
        self.layer_norm2 = nn.LayerNorm(lstm_hidden * 2)

        # Регуляризация
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, attention_mask=None, fragment_weights=None):
        """
        Прямой проход с учетом фрагментов.

        Args:
            x: Входной тензор [batch, seq_len]
            attention_mask: Маска для паддинга [batch, seq_len]
            fragment_weights: Веса для фрагментов пропаганды [batch, seq_len]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Эмбеддинги [batch, seq_len, embedding_dim]
        x = self.embedding(x)

        # Подготовка для CNN [batch, embedding_dim, seq_len]
        x = x.transpose(1, 2)

        # CNN обработка с сохранением размерности
        conv_outputs = []
        for conv in self.conv_layers:
            # Применяем свертку и активацию
            conv_out = F.relu(conv(x))  # [batch, num_filters, seq_len]
            conv_outputs.append(conv_out)

        # Конкатенация выходов CNN [batch, num_filters * num_conv_layers, seq_len]
        x = torch.cat(conv_outputs, dim=1)

        # Возвращаем к формату sequence-first [batch, seq_len, features]
        x = x.transpose(1, 2)

        x = self.layer_norm1(x)
        x = self.dropout(x)

        # BiLSTM обработка
        if attention_mask is not None:
            # Получаем реальные длины последовательностей
            lengths = attention_mask.sum(dim=1).cpu()

            # Пакуем последовательность
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )

            # Пропускаем через LSTM
            packed_output, _ = self.bilstm(packed_x)

            # Распаковываем обратно
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=seq_len
            )
        else:
            x, _ = self.bilstm(x)

        x = self.layer_norm2(x)
        x = self.dropout(x)

        # Применение внимания
        word_context = self.word_attention(x, x, x, attention_mask)  # [batch, seq_len, lstm_hidden * 2]

        if fragment_weights is not None:
            # Применяем веса фрагментов [batch, seq_len, 1]
            fragment_weights = fragment_weights.unsqueeze(-1)
            weighted_x = x * fragment_weights
            fragment_context = weighted_x.sum(dim=1) / (fragment_weights.sum(dim=1) + 1e-10)
        else:
            fragment_context = torch.zeros(batch_size, self.lstm_hidden * 2).to(x.device)

        sentence_context = self.sentence_attention(x, x, x, attention_mask)  # [batch, seq_len, lstm_hidden * 2]

        # Объединение контекстов [batch, lstm_hidden * 8]
        combined_context = torch.cat([
            word_context.mean(dim=1),  # [batch, lstm_hidden * 2]
            fragment_context,  # [batch, lstm_hidden * 2]
            sentence_context.mean(dim=1),  # [batch, lstm_hidden * 2]
            torch.max(x, dim=1)[0]  # [batch, lstm_hidden * 2]
        ], dim=1)

        # Классификация
        features = self.classifier(combined_context)  # [batch, context_size // 4]
        binary_out = self.binary_head(features)  # [batch, 2]
        technique_out = self.technique_head(features)  # [batch, num_classes]

        return binary_out, technique_out


class PropagandaDataset(Dataset):
    """Датасет для обнаружения пропаганды с учетом фрагментов."""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Извлечение уникальных техник из списков
        all_techniques = set()
        for techniques_list in data['technique']:
            if isinstance(techniques_list, list):
                all_techniques.update(techniques_list)
            else:
                all_techniques.add(techniques_list)

        # Создаем энкодер для техник
        self.technique_encoder = LabelEncoder()
        techniques = sorted(list(all_techniques - {'non-propaganda'}))
        self.technique_encoder.fit(['non-propaganda'] + techniques)

        self.prepare_labels()

    def prepare_labels(self):
        """Подготовка меток и токенизация фрагментов."""
        self.technique_labels = []
        self.binary_labels = []

        for techniques in self.data['technique']:
            if isinstance(techniques, list):
                # Если есть хоть одна техника кроме non-propaganda
                has_propaganda = any(t != 'non-propaganda' for t in techniques)
                self.binary_labels.append(int(has_propaganda))

                # Берем первую технику не non-propaganda или non-propaganda если таких нет
                primary_technique = next(
                    (t for t in techniques if t != 'non-propaganda'),
                    'non-propaganda'
                )
                self.technique_labels.append(
                    self.technique_encoder.transform([primary_technique])[0]
                )
            else:
                # Одиночная техника
                self.binary_labels.append(
                    int(techniques != 'non-propaganda')
                )
                self.technique_labels.append(
                    self.technique_encoder.transform([techniques])[0]
                )

        # Конвертируем метки в numpy массивы
        self.binary_labels = np.array(self.binary_labels, dtype=np.int64)
        self.technique_labels = np.array(self.technique_labels, dtype=np.int64)

        # Токенизация фрагментов
        self.fragment_tokens = []
        self.fragment_masks = []

        for _, row in self.data.iterrows():
            fragments = row['fragment']

            # Обработка различных типов фрагментов
            if isinstance(fragments, list):
                fragment = fragments[0] if fragments else None
            else:
                fragment = fragments

            # Проверка на валидность фрагмента
            if pd.isna(fragment) or (isinstance(row['technique'], list) and
                                     all(t == 'non-propaganda' for t in row['technique'])):
                # Создаем пустой фрагмент
                self.fragment_tokens.append(None)
                self.fragment_masks.append(None)
            else:
                try:
                    fragment_encoding = self.tokenizer(
                        str(fragment),
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.max_length,
                        padding='max_length',
                        return_tensors='pt'
                    )

                    # Сохраняем токены и маску
                    self.fragment_tokens.append(fragment_encoding['input_ids'].squeeze())
                    self.fragment_masks.append(fragment_encoding['attention_mask'].squeeze())
                except Exception as e:
                    logger.warning(f"Error encoding fragment: {str(e)}")
                    self.fragment_tokens.append(None)
                    self.fragment_masks.append(None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentence']

        # Токенизация основного текста
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Получаем базовые тензоры
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Инициализируем веса фрагментов
        fragment_weights = torch.ones(self.max_length)

        # Обрабатываем фрагмент, если он есть
        if self.fragment_tokens[idx] is not None:
            fragment_ids = self.fragment_tokens[idx]
            fragment_mask = self.fragment_masks[idx]

            # Получаем фактическую длину фрагмента
            frag_length = fragment_mask.sum().item()

            # Ищем фрагмент в тексте только в пределах реальных токенов
            text_length = attention_mask.sum().item()

            for i in range(min(text_length - frag_length + 1, self.max_length - frag_length + 1)):
                if torch.equal(
                        input_ids[i:i + frag_length],
                        fragment_ids[:frag_length]
                ):
                    fragment_weights[i:i + frag_length] = 2.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'fragment_weights': fragment_weights,
            'binary_label': torch.tensor(self.binary_labels[idx], dtype=torch.long),
            'technique_label': torch.tensor(self.technique_labels[idx], dtype=torch.long)
        }


class HierarchicalClassificationPipeline(ClassificationPipeline):
    def __init__(
            self,
            model_path: str = "models",
            model_name: str = "neural_propaganda_model",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            batch_size: int = 32,
            num_epochs: int = 10,
            learning_rate: float = 2e-5,
            warmup_steps: int = 1000,
            max_length: int = 512,
            class_weights: bool = True,
            **kwargs
    ):
        super().__init__(model_path, model_name)
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_length = max_length
        self.class_weights = class_weights

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self._initialize_components()

        logger.info(f"Initialized pipeline on device: {device}")

    def _initialize_components(self):
        """Инициализация компонентов модели."""
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding_dim = 768
        self.num_filters = 256
        self.lstm_hidden = 256

        self.patience = 3
        self.max_steps = self.num_epochs * 1000

    def _load_and_prepare_data(self, data_path: str) -> pd.DataFrame:
        """Загрузка и подготовка данных."""
        try:
            data = pd.read_csv(data_path)
            required_columns = ['sentence', 'technique', 'fragment']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Группировка по предложениям
            grouped_data = data.groupby(['article_id', 'sentence_id', 'sentence']).agg({
                'technique': list,
                'fragment': list
            }).reset_index()

            return grouped_data

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def _create_model(self, num_classes):
        """Создание модели."""
        return HierarchicalModel(
            self.vocab_size,
            self.embedding_dim,
            self.num_filters,
            self.lstm_hidden,
            num_classes
        ).to(self.device)

    def _create_dataloaders(self, data: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        try:
            # Убедимся, что у нас есть примеры всех классов
            all_techniques = data['technique'].explode().unique()
            print(f"Unique techniques: {all_techniques}")

            # Стратифицированное разделение
            train_size = int(0.9 * len(data))
            indices = list(range(len(data)))
            train_indices, val_indices = train_test_split(
                indices,
                train_size=train_size,
                stratify=data['technique'].apply(lambda x: x[0] if isinstance(x, list) else x)
            )

            # Создаем датасет
            dataset = PropagandaDataset(
                data,
                self.tokenizer,
                max_length=self.max_length
            )

            # Создаем сабсеты с учетом стратификации
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True
            )

            return train_loader, val_loader

        except Exception as e:
            logger.error(f"Failed to create dataloaders: {str(e)}")
            raise

    def train_and_evaluate(self, data_path: str) -> Dict[str, Any]:
        """Обучение и оценка модели."""
        try:
            logger.info("Loading and preparing dataset...")
            data = self._load_and_prepare_data(data_path)

            logger.info("Creating dataloaders...")
            # Create temporary dataset to get encoder
            temp_dataset = PropagandaDataset(
                data,
                self.tokenizer,
                max_length=self.max_length
            )
            # Store the encoder from dataset
            self.technique_encoder = temp_dataset.technique_encoder

            # Now create the actual dataloaders
            self.train_loader, self.val_loader = self._create_dataloaders(data)

            logger.info("Initializing model...")
            num_classes = len(self.technique_encoder.classes_)
            model = self._create_model(num_classes)

            # Create loss functions with proper weights
            if self.class_weights:
                # Вычисляем веса для binary classification
                binary_counts = np.bincount(temp_dataset.binary_labels)
                binary_weights = torch.FloatTensor(
                    [1.0 / (c if c > 0 else 1.0) for c in binary_counts]
                ).to(self.device)

                # Вычисляем веса для technique classification
                technique_counts = np.bincount(temp_dataset.technique_labels)
                technique_weights = torch.FloatTensor(
                    [1.0 / (c if c > 0 else 1.0) for c in technique_counts]
                ).to(self.device)

                binary_criterion = FocalLoss(alpha=binary_weights, gamma=2.0)
                technique_criterion = FocalLoss(alpha=technique_weights, gamma=2.0)
            else:
                binary_criterion = FocalLoss()
                technique_criterion = FocalLoss()

            # Create optimizer and scheduler
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

            # Move model and loss functions to device
            model = model.to(self.device)
            binary_criterion = binary_criterion.to(self.device)
            technique_criterion = technique_criterion.to(self.device)

            logger.info("Starting training...")
            best_val_metrics = None
            best_model = None

            for epoch in range(self.num_epochs):
                train_metrics = self._train_epoch(
                    model,
                    self.train_loader,
                    optimizer,
                    scheduler,
                    binary_criterion,
                    technique_criterion
                )

                val_metrics = self._validate_epoch(
                    model,
                    self.val_loader,
                    binary_criterion,
                    technique_criterion
                )

                # Логирование метрик
                logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
                logger.info("Training metrics:")
                for k, v in train_metrics.items():
                    logger.info(f"{k}: {v:.4f}")

                logger.info("\nValidation metrics:")
                for k, v in val_metrics.items():
                    logger.info(f"{k}: {v:.4f}")

                # Early stopping и сохранение лучшей модели
                if best_val_metrics is None or val_metrics['loss'] < best_val_metrics['loss']:
                    best_val_metrics = val_metrics
                    self.patience_counter = 0
                    best_model = copy.deepcopy(model)
                    self._save_model(model, val_metrics)
                    logger.info("Saved new best model")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logger.info("Early stopping triggered")
                        break

            logger.info("Training completed. Evaluating final model...")
            # Используем лучшую модель для финальной оценки
            if best_model is not None:
                model = best_model

            # Проводим полную оценку модели
            final_metrics = self._evaluate_model(model, self.val_loader)

            logger.info("\nFinal Evaluation Results:")
            logger.info("\nBinary Classification Metrics:")
            for metric, value in final_metrics['binary_metrics'].items():
                logger.info(f"{metric}: {value:.4f}")

            logger.info("\nTechnique Classification Metrics:")
            for metric, value in final_metrics['technique_metrics'].items():
                logger.info(f"{metric}: {value:.4f}")

            return final_metrics

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def _train_epoch(self, model, train_loader, optimizer, scheduler, binary_criterion, technique_criterion):
        """Обучение одной эпохи."""
        model.train()
        total_loss = 0
        binary_losses = 0
        technique_losses = 0

        binary_preds = []
        binary_labels = []
        technique_preds = []
        technique_labels = []

        for batch in tqdm(train_loader, desc="Training"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            binary_out, technique_out = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['fragment_weights']
            )

            binary_loss = binary_criterion(binary_out, batch['binary_label'])
            technique_loss = technique_criterion(technique_out, batch['technique_label'])

            # Динамическое взвешивание потерь
            total_binary_weight = torch.exp(-binary_loss.detach())
            total_technique_weight = torch.exp(-technique_loss.detach())

            loss = (binary_loss * total_binary_weight +
                    technique_loss * total_technique_weight) / (
                           total_binary_weight + total_technique_weight
                   )

            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            binary_losses += binary_loss.item()
            technique_losses += technique_loss.item()

            # Collect predictions
            binary_preds.extend(binary_out.argmax(dim=1).cpu().numpy())
            binary_labels.extend(batch['binary_label'].cpu().numpy())
            technique_preds.extend(technique_out.argmax(dim=1).cpu().numpy())
            technique_labels.extend(batch['technique_label'].cpu().numpy())

        # Calculate metrics
        metrics = {
            'loss': total_loss / len(train_loader),
            'binary_loss': binary_losses / len(train_loader),
            'technique_loss': technique_losses / len(train_loader),
            'binary_f1': f1_score(binary_labels, binary_preds, average='weighted'),
            'technique_f1': f1_score(technique_labels, technique_preds, average='weighted')
        }

        return metrics

    def _validate_epoch(self, model, val_loader, binary_criterion, technique_criterion):
        """Валидация модели."""
        model.eval()
        total_loss = 0
        binary_losses = 0
        technique_losses = 0

        binary_preds = []
        binary_labels = []
        technique_preds = []
        technique_labels = []
        technique_probs = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                binary_out, technique_out = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['fragment_weights']
                )

                binary_loss = binary_criterion(binary_out, batch['binary_label'])
                technique_loss = technique_criterion(technique_out, batch['technique_label'])
                loss = 0.4 * binary_loss + 0.6 * technique_loss

                total_loss += loss.item()
                binary_losses += binary_loss.item()
                technique_losses += technique_loss.item()

                binary_preds.extend(binary_out.argmax(dim=1).cpu().numpy())
                binary_labels.extend(batch['binary_label'].cpu().numpy())
                technique_preds.extend(technique_out.argmax(dim=1).cpu().numpy())
                technique_labels.extend(batch['technique_label'].cpu().numpy())
                technique_probs.extend(torch.softmax(technique_out, dim=1).cpu().numpy())

        # Преобразуем в numpy массивы
        binary_preds = np.array(binary_preds)
        binary_labels = np.array(binary_labels)
        technique_preds = np.array(technique_preds)
        technique_labels = np.array(technique_labels)
        technique_probs = np.array(technique_probs)

        # Безопасное вычисление ROC AUC
        try:
            # Проверяем, есть ли хотя бы два класса
            unique_classes = np.unique(technique_labels)
            if len(unique_classes) >= 2:
                # Бинаризуем метки для мультикласса
                binary_labels_one_hot = label_binarize(
                    technique_labels,
                    classes=range(len(self.technique_encoder.classes_))
                )
                technique_roc_auc = roc_auc_score(
                    binary_labels_one_hot,
                    technique_probs,
                    average='weighted',
                    multi_class='ovr'
                )
            else:
                logger.warning("Only one class present in validation set. ROC AUC set to 0.0")
                technique_roc_auc = 0.0
        except Exception as e:
            logger.warning(f"Error calculating ROC AUC: {str(e)}. Setting to 0.0")
            technique_roc_auc = 0.0

        metrics = {
            'loss': total_loss / len(val_loader),
            'binary_loss': binary_losses / len(val_loader),
            'technique_loss': technique_losses / len(val_loader),
            'binary_f1': f1_score(binary_labels, binary_preds, average='weighted'),
            'technique_f1': f1_score(technique_labels, technique_preds, average='weighted'),
            'technique_roc_auc': technique_roc_auc
        }

        return metrics

    def _evaluate_model(self, model, data_loader):
        """
        Полная оценка модели на данных.

        Args:
            model: Обученная модель
            data_loader: DataLoader с данными для оценки

        Returns:
            Dict с метриками
        """
        model.eval()
        all_binary_preds = []
        all_binary_labels = []
        all_technique_preds = []
        all_technique_labels = []
        all_technique_probs = []

        # Собираем все предсказания
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                binary_out, technique_out = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['fragment_weights']
                )

                # Сохраняем предсказания
                all_binary_preds.extend(binary_out.argmax(dim=1).cpu().numpy())
                all_binary_labels.extend(batch['binary_label'].cpu().numpy())
                all_technique_preds.extend(technique_out.argmax(dim=1).cpu().numpy())
                all_technique_labels.extend(batch['technique_label'].cpu().numpy())
                all_technique_probs.extend(torch.softmax(technique_out, dim=1).cpu().numpy())

        # Преобразуем в numpy массивы
        all_binary_preds = np.array(all_binary_preds)
        all_binary_labels = np.array(all_binary_labels)
        all_technique_preds = np.array(all_technique_preds)
        all_technique_labels = np.array(all_technique_labels)
        all_technique_probs = np.array(all_technique_probs)

        # Вычисляем бинарные метрики
        binary_metrics = {
            'accuracy': accuracy_score(all_binary_labels, all_binary_preds),
            'precision': precision_score(all_binary_labels, all_binary_preds, average='weighted'),
            'recall': recall_score(all_binary_labels, all_binary_preds, average='weighted'),
            'f1': f1_score(all_binary_labels, all_binary_preds, average='weighted')
        }

        # Вычисляем метрики для техник
        try:
            # Проверяем наличие множественных классов
            unique_classes = np.unique(all_technique_labels)
            if len(unique_classes) >= 2:
                # Бинаризуем метки для мультикласса
                technique_labels_one_hot = label_binarize(
                    all_technique_labels,
                    classes=range(len(self.technique_encoder.classes_))
                )
                technique_roc_auc = roc_auc_score(
                    technique_labels_one_hot,
                    all_technique_probs,
                    average='weighted',
                    multi_class='ovr'
                )
            else:
                logger.warning("Only one class present in evaluation set. ROC AUC set to 0.0")
                technique_roc_auc = 0.0
        except Exception as e:
            logger.warning(f"Error calculating ROC AUC: {str(e)}. Setting to 0.0")
            technique_roc_auc = 0.0

        technique_metrics = {
            'accuracy': accuracy_score(all_technique_labels, all_technique_preds),
            'precision': precision_score(all_technique_labels, all_technique_preds, average='weighted'),
            'recall': recall_score(all_technique_labels, all_technique_preds, average='weighted'),
            'f1': f1_score(all_technique_labels, all_technique_preds, average='weighted'),
            'roc_auc': technique_roc_auc
        }

        # Формируем подробный отчет по классам для техник
        class_report = classification_report(
            all_technique_labels,
            all_technique_preds,
            target_names=self.technique_encoder.classes_,
            output_dict=True
        )

        # Собираем все метрики в один словарь
        return {
            'binary_metrics': binary_metrics,
            'technique_metrics': technique_metrics,
            'class_report': class_report,
            'confusion_matrix': confusion_matrix(
                all_technique_labels,
                all_technique_preds
            ).tolist()
        }

    def predict(self, texts: List[str]) -> Dict[str, Any]:
        """Предсказание для новых текстов."""
        try:
            if not check_lang_corpus(texts, "en"):
                texts = translate_corpus(texts)
            model = self._load_model()
            model.eval()

            # Подготовка данных
            dataset = PropagandaDataset(
                pd.DataFrame({'sentence': texts, 'technique': ['non-propaganda'] * len(texts),
                              'fragment': [None] * len(texts)}),
                self.tokenizer,
                max_length=self.max_length
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True
            )

            predictions = []
            probabilities = []

            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    binary_out, technique_out = model(
                        batch['input_ids'],
                        batch['attention_mask'],
                        batch['fragment_weights']
                    )

                    binary_probs = torch.softmax(binary_out, dim=1)
                    technique_probs = torch.softmax(technique_out, dim=1)

                    for b_prob, t_prob in zip(binary_probs, technique_probs):
                        if b_prob[1] > 0.3:  # Снизим порог с 0.5 до 0.3
                            text_preds = []
                            text_probs = {}

                            for idx, prob in enumerate(t_prob):
                                if prob > 0.3:  # Также снизим порог здесь
                                    technique = self.technique_encoder.inverse_transform([idx])[0]
                                    if technique != 'non-propaganda':
                                        text_preds.append(technique)
                                        text_probs[technique] = prob.item()

                            if not text_preds:
                                predictions.append(['non-propaganda'])
                                probabilities.append({'non-propaganda': b_prob[0].item()})
                            else:
                                predictions.append(text_preds)
                                probabilities.append(text_probs)
                        else:
                            predictions.append(['non-propaganda'])
                            probabilities.append({'non-propaganda': b_prob[0].item()})

            return {
                'predictions': predictions,
                'probabilities': probabilities
            }

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _save_model(self, model, metrics=None):
        """Сохранение модели."""
        try:
            state = {
                'model_state': model.state_dict(),
                'technique_encoder': self.technique_encoder
            }
            if metrics:
                state['metrics'] = metrics

            save_model(state, f"{self.model_path}/{self.model_name}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    def _load_model(self):
        """Загрузка модели."""
        try:
            checkpoint = load_model(f"{self.model_path}/{self.model_name}")
            num_classes = len(checkpoint['technique_encoder'].classes_)
            model = self._create_model(num_classes)
            model.load_state_dict(checkpoint['model_state'])
            self.technique_encoder = checkpoint['technique_encoder']
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize pipeline with configuration
    pipeline = HierarchicalClassificationPipeline(
        model_path="../models",
        model_name="hpm_v1",
        batch_size=32,
        num_epochs=10,
        learning_rate=2e-5,
        warmup_steps=1000,
        max_length=512,
        class_weights=True
    )

    # Training and evaluation
    try:
        logger.info("Starting training and evaluation...")
        metrics = pipeline.train_and_evaluate(
            data_path="../datasets/tasks-2-3/combined_dataset.csv"
        )

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

    # Test predictions
    test_texts = [
        "Orange is orange.",
        "Why We should support our great nation.",
        "Our country stands on the threshold of the greatest achievements in history. While malicious critics try to undermine our progress, we must remain united and steadfast. Our leader, continuing a long tradition of outstanding government, strives tirelessly for the benefit of every citizen.",
        "Our leader is the best of the best all over the world.",
        "The Everyday Struggle of Ordinary People Against Feminist Influence.",
        "In these difficult times, when national and international enemies threaten our way of life, our indomitable leader stands guard over stability and development. Thanks to his wise and decisive actions, our country's economy is experiencing unprecedented growth.",
        "We stand on the threshold of the greatest challenges of our time. Our society, our traditions and our history are under attack from those who want to see us weakened and divided. Do not be deceived: our enemies are both internal traitors and external aggressors.",
        "Pamela Geller and Robert Spencer co founded anti Muslim group Stop Islamization of America.",
        "Children loves chocolate.",
        "Little Kate always looks forward to the weekend to go for a walk in the nearby park.",
        "Make sure you use the correct instances of models and vectorizers for testing that match the number of features used in the trained models.",
        "Save the crowbar from the ship to blow up the Japanese.",
        "Destroy this crazy beast. Enroll in the US Army.",
        "Fulfill the five-year plan not in five years, but in four.",
        "Death to capital or death under the heel of capital!",
        "Workers of the world, unite! For a worldwide October!",
        "Why we fight - for our children`s bread."
    ]

    try:
        logger.info("Running predictions on test texts...")
        results = pipeline.predict(test_texts)

        print("\nPrediction Results:")
        print("-" * 50)
        for text, preds, probs in zip(
                test_texts,
                results['predictions'],
                results['probabilities']
        ):
            print(f"\nText: {text}")
            print("Detected Propaganda Techniques:")
            if 'non-propaganda' in preds:
                print("No propaganda detected")
            else:
                for technique in preds:
                    prob = probs.get(technique, 0.0)
                    print(f"  - {technique}: {prob:.3f}")
            print("-" * 30)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise
