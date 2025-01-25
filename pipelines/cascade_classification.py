import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import logging
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_manipulating.manipulate_models import save_model, load_model
from pipelines.classification import ClassificationPipeline
from utils.google_translate import check_lang_corpus, translate_corpus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


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
    """Revised multi-head attention layer with proper dimension handling."""

    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def forward(self, query, key, value, mask=None):
        # Reshape inputs to match MultiheadAttention expectations
        # Input shape: [batch_size, seq_len, hidden_size]
        # Required shape: [seq_len, batch_size, hidden_size]
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        # Process attention mask to match expected shape
        if mask is not None:
            # MultiheadAttention expects mask of shape [batch_size, seq_len]
            # and True values are locations to be masked
            mask = mask.bool()
            mask = ~mask  # Invert mask as per PyTorch requirements

        # Apply attention
        # Note: key_padding_mask should be [batch_size, seq_len]
        output, attention_weights = self.attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=mask,
            need_weights=True
        )

        # Restore original shape [batch_size, seq_len, hidden_size]
        output = output.transpose(0, 1)

        return output


class BinaryPropagandaModel(nn.Module):
    """Revised binary classification model with proper attention handling."""

    def __init__(self, vocab_size, embedding_dim, num_filters, lstm_hidden):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.lstm_hidden = lstm_hidden

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # CNN layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                embedding_dim,
                num_filters,
                kernel_size=k,
                padding='same'
            )
            for k in [3, 5, 7]
        ])

        # BiLSTM
        cnn_output_size = num_filters * len(self.conv_layers)
        self.bilstm = nn.LSTM(
            cnn_output_size,
            lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )

        # Attention
        self.attention = AttentionLayer(lstm_hidden * 2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(lstm_hidden),
            nn.Linear(lstm_hidden, 2),
        )

    def forward(self, x, attention_mask=None):
        # Embeddings [batch, seq_len, emb_dim]
        x = self.embedding(x)
        batch_size, seq_len, _ = x.size()

        # CNN [batch, num_filters, seq_len]
        x = x.transpose(1, 2)
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(x))
            conv_outputs.append(conv_out)
        x = torch.cat(conv_outputs, dim=1)
        x = x.transpose(1, 2)

        # BiLSTM with packed sequence if mask is provided
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.bilstm(packed_x)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=seq_len
            )
        else:
            x, _ = self.bilstm(x)

        # Apply attention
        if attention_mask is not None:
            x = self.attention(x, x, x, attention_mask)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Classification
        x = self.classifier(x)

        return x


class TechniquePropagandaModel(nn.Module):
    """Model for propaganda technique classification with proper attention handling."""

    def __init__(self, vocab_size, embedding_dim, num_filters, lstm_hidden, num_classes):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.lstm_hidden = lstm_hidden

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # CNN layers with different kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                embedding_dim,
                num_filters,
                kernel_size=k,
                padding='same'
            )
            for k in [3, 5]
        ])

        # BiLSTM
        cnn_output_size = num_filters * len(self.conv_layers)
        self.bilstm = nn.LSTM(
            cnn_output_size,
            lstm_hidden,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        # Attention layers
        self.word_attention = AttentionLayer(lstm_hidden * 2)
        self.sent_attention = AttentionLayer(lstm_hidden * 2)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(cnn_output_size)
        self.layer_norm2 = nn.LayerNorm(lstm_hidden * 2)

        # Вычисляем размер входа для классификатора
        # lstm_hidden * 2 (bidirectional) * 4 (конкатенация контекстов)
        input_size = lstm_hidden * 8

        self.classifier = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(input_size // 2),
            nn.Linear(input_size // 2, num_classes),
            nn.Dropout(0.1)
        )

    def forward(self, x, attention_mask=None, fragment_weights=None):
        # Get dimensions
        batch_size, seq_len = x.size()

        # Embeddings [batch, seq_len, emb_dim]
        x = self.embedding(x)

        # CNN [batch, num_filters * num_kernels, seq_len]
        x = x.transpose(1, 2)
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(x))
            conv_outputs.append(conv_out)
        x = torch.cat(conv_outputs, dim=1)
        x = x.transpose(1, 2)

        # Layer normalization
        x = self.layer_norm1(x)

        # BiLSTM with packed sequence if mask is provided
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.bilstm(packed_x)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=seq_len
            )
        else:
            x, _ = self.bilstm(x)

        x = self.layer_norm2(x)

        # Apply multi-level attention
        if attention_mask is not None:
            # Получаем контексты на уровне слов и предложений
            word_context = self.word_attention(x, x, x, attention_mask)
            sent_context = self.sent_attention(x, x, x, attention_mask)
        else:
            word_context = self.word_attention(x, x, x)
            sent_context = self.sent_attention(x, x, x)

        # Combine contexts
        if fragment_weights is not None:
            # Применяем веса к фрагментам
            fragment_weights = fragment_weights.unsqueeze(-1)
            weighted_x = x * fragment_weights

            # Объединяем все контексты
            x = torch.cat([
                word_context.mean(dim=1),  # Усредненный контекст слов
                sent_context.mean(dim=1),  # Усредненный контекст предложений
                weighted_x.sum(dim=1) / (fragment_weights.sum(dim=1) + 1e-10),  # Взвешенная сумма
                torch.max(x, dim=1)[0]  # Максимальные значения признаков
            ], dim=1)
        else:
            # Без весов фрагментов
            x = torch.cat([
                word_context.mean(dim=1),
                sent_context.mean(dim=1),
                torch.mean(x, dim=1),
                torch.max(x, dim=1)[0]
            ], dim=1)

        # Classification
        x = self.classifier(x)

        return x


class PropagandaDataset(Dataset):
    """Датасет для каскадной модели."""

    def __init__(self, data, tokenizer, max_length=512, binary_only=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.binary_only = binary_only

        if not binary_only:
            # Создаем энкодер для техник только если нужно
            all_techniques = set()
            for techniques in data['technique']:
                if isinstance(techniques, list):
                    all_techniques.update(techniques)
                else:
                    all_techniques.add(techniques)

            self.technique_encoder = LabelEncoder()
            techniques = sorted(list(all_techniques - {'non-propaganda'}))
            self.technique_encoder.fit(['non-propaganda'] + techniques)

        self.prepare_labels()

    def prepare_labels(self):
        """Подготовка меток."""
        self.binary_labels = []
        if not self.binary_only:
            self.technique_labels = []
            self.fragment_tokens = []
            self.fragment_masks = []

        for idx, row in self.data.iterrows():
            # Подготовка бинарных меток
            techniques = row['technique']
            if isinstance(techniques, list):
                has_propaganda = any(t != 'non-propaganda' for t in techniques)
                self.binary_labels.append(int(has_propaganda))

                if not self.binary_only:
                    # Создаем multi-hot encoding для всех техник
                    labels = torch.zeros(len(self.technique_encoder.classes_))
                    for tech in techniques:
                        if tech != 'non-propaganda':
                            idx = self.technique_encoder.transform([tech])[0]
                            labels[idx] = 1
                    self.technique_labels.append(labels)
            else:
                self.binary_labels.append(int(techniques != 'non-propaganda'))
                if not self.binary_only:
                    labels = torch.zeros(len(self.technique_encoder.classes_))
                    if techniques != 'non-propaganda':
                        idx = self.technique_encoder.transform([techniques])[0]
                        labels[idx] = 1
                    self.technique_labels.append(labels)

            # Обработка фрагментов остается той же
            if not self.binary_only:
                fragments = row['fragment']
                if isinstance(fragments, list):
                    fragment = fragments[0] if fragments else None
                else:
                    fragment = fragments

                if pd.isna(fragment):
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

                        self.fragment_tokens.append(fragment_encoding['input_ids'].squeeze())
                        self.fragment_masks.append(fragment_encoding['attention_mask'].squeeze())
                    except Exception as e:
                        logger.warning(f"Error encoding fragment: {str(e)}")
                        self.fragment_tokens.append(None)
                        self.fragment_masks.append(None)

        # Преобразуем в numpy массивы
        self.binary_labels = np.array(self.binary_labels, dtype=np.int64)
        if not self.binary_only:
            self.technique_labels = torch.stack(self.technique_labels)  # Изменено для multi-hot encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentence']

        # Токенизация текста
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'binary_label': torch.tensor(self.binary_labels[idx], dtype=torch.long)
        }

        # Добавляем метки техник и фрагменты только если нужно
        if not self.binary_only:
            result['technique_label'] = torch.tensor(
                self.technique_labels[idx],
                dtype=torch.long
            )

            # Добавляем веса фрагментов
            fragment_weights = torch.ones(self.max_length)
            if self.fragment_tokens[idx] is not None:
                fragment_ids = self.fragment_tokens[idx]
                fragment_mask = self.fragment_masks[idx]

                frag_length = fragment_mask.sum().item()
                text_length = encoding['attention_mask'].squeeze().sum().item()

                for i in range(min(text_length - frag_length + 1, self.max_length - frag_length + 1)):
                    if torch.equal(
                            result['input_ids'][i:i + frag_length],
                            fragment_ids[:frag_length]
                    ):
                        fragment_weights[i:i + frag_length] = 2.0

            result['fragment_weights'] = fragment_weights

        return result


class CascadePropagandaPipeline(ClassificationPipeline):
    def __init__(
            self,
            model_path: str = "models",
            model_name: str = "cascade_propaganda_model",
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

    def get_params(self, detailed=True):
        """
        Получение параметров моделей и конфигурации.

        Args:
            detailed (bool): Если True, возвращает детальную информацию о слоях

        Returns:
            dict: Словарь с параметрами
        """
        # Загружаем модели если нужно
        if not hasattr(self, 'binary_model') or not hasattr(self, 'technique_model'):
            self._load_models()

        # Определяем класс модели
        binary_model_class = self.binary_model.__class__.__name__
        technique_model_class = self.technique_model.__class__.__name__

        # Базовые параметры
        params = {
            'model_version': {
                'binary_model': binary_model_class,
                'technique_model': technique_model_class
            },
            'general_config': {
                'device': self.device,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate,
                'warmup_steps': self.warmup_steps,
                'max_length': self.max_length,
                'class_weights': self.class_weights
            },
            'model_architecture': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'num_filters': self.num_filters,
                'lstm_hidden': self.lstm_hidden
            }
        }

        if detailed:
            params.update(self._get_detailed_params())
            if hasattr(self, 'technique_encoder'):
                params.update({
                    'tokenizer': {'name': self.tokenizer.name_or_path},
                    'num_technique_classes': len(self.technique_encoder.classes_),
                    'technique_classes': self.technique_encoder.classes_.tolist()
                })

        return params

    def _get_detailed_params(self):
        """
        Получение детальной информации о слоях модели.
        """
        return {
            'binary_model_layers': self._get_binary_model_layers(),
            'technique_model_layers': self._get_technique_model_layers()
        }

    def _get_binary_model_layers(self):
        """
        Базовая информация о слоях бинарной модели.
        """
        return {
            'embedding': {
                'type': 'Embedding',
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim
            },
            'conv_layers': [{
                'type': 'Conv1d',
                'in_channels': self.embedding_dim,
                'out_channels': self.num_filters,
                'kernel_size': k,
                'padding': 'same'
            } for k in [3, 5, 7]],
            'bilstm': {
                'type': 'LSTM',
                'input_size': self.num_filters * 3,
                'hidden_size': self.lstm_hidden,
                'num_layers': 2,
                'bidirectional': True
            },
            'attention': {
                'type': 'AttentionLayer',
                'hidden_size': self.lstm_hidden * 2,
                'num_heads': 8
            },
            'classifier': [
                {'type': 'Linear', 'in': self.lstm_hidden * 2, 'out': self.lstm_hidden},
                {'type': 'ReLU'},
                {'type': 'Dropout', 'p': 0.5},
                {'type': 'Linear', 'in': self.lstm_hidden, 'out': 2}
            ]
        }

    def _get_technique_model_layers(self):
        """
        Базовая информация о слоях модели техник.
        """
        if not hasattr(self, 'technique_model'):
            return {}

        return {
            'embedding': {
                'type': 'Embedding',
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim
            },
            'conv_layers': [{
                'type': 'Conv1d',
                'in_channels': self.embedding_dim,
                'out_channels': self.num_filters,
                'kernel_size': k,
                'padding': 'same'
            } for k in [3, 5]],
            'bilstm': {
                'type': 'LSTM',
                'input_size': self.num_filters * 4,
                'hidden_size': self.lstm_hidden,
                'num_layers': 3,
                'bidirectional': True,
                'dropout': 0.3
            },
            'attention': {
                'word_attention': {
                    'type': 'AttentionLayer',
                    'hidden_size': self.lstm_hidden * 2,
                    'num_heads': 8
                },
                'sent_attention': {
                    'type': 'AttentionLayer',
                    'hidden_size': self.lstm_hidden * 2,
                    'num_heads': 8
                }
            },
            'classifier': [
                {'type': 'Linear', 'in': self.lstm_hidden * 8, 'out': self.lstm_hidden * 4},
                {'type': 'ReLU'},
                {'type': 'Dropout', 'p': 0.3},
                {'type': 'BatchNorm1d', 'size': self.lstm_hidden * 4},
                {'type': 'Linear', 'in': self.lstm_hidden * 4, 'out': len(self.technique_encoder.classes_)}
            ]
        }

    def print_params(self, detailed=True):
        """
        Печать параметров в удобочитаемом формате.
        """
        params = self.get_params(detailed)
        print("\n=== Model Parameters ===")

        print("\nModel Versions:")
        for model, version in params['model_version'].items():
            if version:  # Проверяем, что версия существует
                print(f"  {model}: {version}")

        print("\nGeneral Configuration:")
        for param, value in params['general_config'].items():
            print(f"  {param}: {value}")

        print("\nModel Architecture:")
        for param, value in params['model_architecture'].items():
            print(f"  {param}: {value}")

        if detailed:
            print("\nBinary Model Layers:")
            self._print_layers(params['binary_model_layers'])

            if 'technique_model_layers' in params and params['technique_model_layers']:
                print("\nTechnique Model Layers:")
                self._print_layers(params['technique_model_layers'])

            if 'technique_classes' in params:
                print("\nTechnique Classes:")
                print(f"  Number of classes: {params['num_technique_classes']}")
                print("  Classes:", ", ".join(params['technique_classes']))

    def _print_dict(self, d, indent=0):
        """
        Рекурсивная печать словаря с отступами.
        """
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                self._print_dict(value, indent + 2)
            else:
                print(" " * indent + f"{key}: {value}")

    def _print_layers(self, layers, indent=2):
        """
        Печать информации о слоях.
        """
        for layer_name, layer_info in layers.items():
            if isinstance(layer_info, list):
                print(" " * indent + f"{layer_name}:")
                for i, layer in enumerate(layer_info):
                    print(" " * (indent + 2) + f"[{i}] {layer['type']}:")
                    for param, value in layer.items():
                        if param != 'type':
                            print(" " * (indent + 4) + f"{param}: {value}")
            else:
                print(" " * indent + f"{layer_name}:")
                for param, value in layer_info.items():
                    if isinstance(value, dict):
                        print(" " * (indent + 2) + f"{param}:")
                        for sub_param, sub_value in value.items():
                            print(" " * (indent + 4) + f"{sub_param}: {sub_value}")
                    else:
                        print(" " * (indent + 2) + f"{param}: {value}")

    def create_datasets(self, data: pd.DataFrame):
        """Создание датасетов для обеих моделей."""
        # Датасет для бинарной классификации (все примеры)
        binary_dataset = PropagandaDataset(
            data,
            self.tokenizer,
            max_length=self.max_length,
            binary_only=True
        )

        # Создаем маску для выделения пропаганды
        propaganda_mask = data['technique'].apply(
            lambda x: isinstance(x, list) and any(t != 'non-propaganda' for t in x)
            if isinstance(x, list)
            else x != 'non-propaganda'
        )

        # Получаем данные с пропагандой для техник
        technique_data = data[propaganda_mask].copy()

        # Создаем датасет техник
        technique_dataset = PropagandaDataset(
            technique_data,
            self.tokenizer,
            max_length=self.max_length,
            binary_only=False
        )

        # Сохраняем энкодер техник
        self.technique_encoder = technique_dataset.technique_encoder

        # Логируем статистику
        logger.info(f"Binary dataset size: {len(binary_dataset)}")
        logger.info(f"Technique dataset size: {len(technique_dataset)}")

        if hasattr(technique_dataset, 'technique_labels'):
            class_counts = technique_dataset.technique_labels.sum(dim=0)
            for i, count in enumerate(class_counts):
                technique = self.technique_encoder.inverse_transform([i])[0]
                logger.info(f"{technique}: {count.item()}")

        return binary_dataset, technique_dataset

    def train_binary_model(self, train_loader, val_loader):
        """Обучение бинарной модели."""
        model = BinaryPropagandaModel(
            self.vocab_size,
            self.embedding_dim,
            self.num_filters,
            self.lstm_hidden
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
            # Собираем все метки из train_loader
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

        for epoch in range(self.num_epochs):
            # Обучение
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []

            for batch in tqdm(train_loader, desc=f"Training binary model - Epoch {epoch + 1}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()
                outputs = model(batch['input_ids'], batch['attention_mask'])
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

                    outputs = model(batch['input_ids'], batch['attention_mask'])
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
        return best_model

    def train_technique_model(self, train_loader, val_loader, num_classes):
        """Обучение модели для классификации техник."""
        model = TechniquePropagandaModel(
            self.vocab_size,
            self.embedding_dim,
            self.num_filters,
            self.lstm_hidden,
            num_classes
        ).to(self.device)

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

        # Изменяем функцию потерь для multi-label классификации
        criterion = nn.BCEWithLogitsLoss()
        criterion = criterion.to(self.device)

        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0

        for epoch in range(self.num_epochs*2):
            # Обучение
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []

            for batch in tqdm(train_loader, desc=f"Training technique model - Epoch {epoch + 1}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()
                outputs = model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['fragment_weights']
                )
                loss = criterion(outputs, batch['technique_label'].float())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_preds.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy())
                train_labels.extend(batch['technique_label'].cpu().numpy())

            # Валидация
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            val_probs = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = model(
                        batch['input_ids'],
                        batch['attention_mask'],
                        batch['fragment_weights']
                    )
                    loss = criterion(outputs, batch['technique_label'].float())

                    val_loss += loss.item()
                    val_preds.extend((torch.sigmoid(outputs) > 0.5).float().cpu().numpy())
                    val_labels.extend(batch['technique_label'].cpu().numpy())
                    val_probs.extend(torch.sigmoid(outputs).cpu().numpy())

            # Вычисляем метрики
            train_metrics = {
                'loss': train_loss / len(train_loader),
                'f1': f1_score(
                    np.array(train_labels),
                    np.array(train_preds),
                    average='weighted',
                    zero_division=0
                )
            }

            val_metrics = {
                'loss': val_loss / len(val_loader),
                'f1': f1_score(
                    np.array(val_labels),
                    np.array(val_preds),
                    average='weighted',
                    zero_division=0
                )
            }

            # Пробуем вычислить ROC AUC
            try:
                val_metrics['roc_auc'] = roc_auc_score(
                    np.array(val_labels),
                    np.array(val_probs),
                    average='weighted',
                    multi_class='ovr'
                )
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {str(e)}")
                val_metrics['roc_auc'] = 0.0

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

    def train_and_evaluate(self, data_path: str) -> Dict[str, Any]:
        """Обучение и оценка каскадной модели."""
        try:
            path = os.path.join(self.model_path, f"{self.model_name}.pkl")

            if os.path.exists(path):
                raise FileExistsError(
                    f"Model file already exists at {path}. Please use a different model name or remove the existing file.")

            logger.info("Loading and preparing dataset...")
            data = self._load_and_prepare_data(data_path)

            # Создаем датасеты
            binary_dataset, technique_dataset = self.create_datasets(data)

            # Разделяем на train/val для бинарной модели
            train_size = int(0.9 * len(binary_dataset))
            val_size = len(binary_dataset) - train_size

            binary_train_dataset, binary_val_dataset = torch.utils.data.random_split(
                binary_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            # Для модели техник используем обычное разделение, как для бинарной модели
            technique_train_size = int(0.9 * len(technique_dataset))
            technique_val_size = len(technique_dataset) - technique_train_size

            technique_train_dataset, technique_val_dataset = torch.utils.data.random_split(
                technique_dataset,
                [technique_train_size, technique_val_size],
                generator=torch.Generator().manual_seed(42)
            )

            # Создаем DataLoader'ы
            binary_train_loader = DataLoader(
                binary_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            binary_val_loader = DataLoader(
                binary_val_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True
            )

            technique_train_loader = DataLoader(
                technique_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            technique_val_loader = DataLoader(
                technique_val_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True
            )

            # Обучаем бинарную модель
            logger.info("Training binary model...")
            self.binary_model = self.train_binary_model(
                binary_train_loader,
                binary_val_loader
            )

            # Обучаем модель техник
            logger.info("Training technique model...")
            num_classes = len(technique_dataset.technique_encoder.classes_)
            self.technique_model = self.train_technique_model(
                technique_train_loader,
                technique_val_loader,
                num_classes
            )

            # Сохраняем состояние
            self._save_models()

            # Возвращаем итоговые метрики
            return self._evaluate_models(binary_val_loader, technique_val_loader)

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def _evaluate_models(self, binary_val_loader, technique_val_loader):
        """Evaluate both models with proper handling of different label types."""
        self.binary_model.eval()
        self.technique_model.eval()

        # Binary model evaluation
        binary_preds = []
        binary_labels = []
        binary_probs = []

        with torch.no_grad():
            for batch in binary_val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.binary_model(batch['input_ids'], batch['attention_mask'])
                probs = torch.softmax(outputs, dim=1)

                binary_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                binary_labels.extend(batch['binary_label'].cpu().numpy())
                binary_probs.extend(probs[:, 1].cpu().numpy())

        # Technique model evaluation
        technique_preds = []
        technique_labels = []
        technique_probs = []

        with torch.no_grad():
            for batch in technique_val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.technique_model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['fragment_weights']
                )
                probs = torch.sigmoid(outputs)  # Use sigmoid for multilabel
                preds = (probs > 0.5).float()  # Threshold predictions

                technique_preds.append(preds.cpu().numpy())
                technique_labels.append(batch['technique_label'].cpu().numpy())
                technique_probs.append(probs.cpu().numpy())

        # Convert lists to arrays
        technique_preds = np.vstack(technique_preds)
        technique_labels = np.vstack(technique_labels)
        technique_probs = np.vstack(technique_probs)

        # Binary metrics
        binary_metrics = {
            'accuracy': accuracy_score(binary_labels, binary_preds),
            'precision': precision_score(binary_labels, binary_preds, average='weighted'),
            'recall': recall_score(binary_labels, binary_preds, average='weighted'),
            'f1': f1_score(binary_labels, binary_preds, average='weighted')
        }

        # Add ROC AUC for binary model if both classes present
        unique_classes = np.unique(binary_labels)
        if len(unique_classes) > 1:
            binary_metrics['roc_auc'] = roc_auc_score(binary_labels, binary_probs)

        # Technique metrics - using sample-averaged metrics for multilabel
        technique_metrics = {
            'accuracy': accuracy_score(technique_labels, technique_preds),
            'precision': precision_score(technique_labels, technique_preds, average='samples'),
            'recall': recall_score(technique_labels, technique_preds, average='samples'),
            'f1': f1_score(technique_labels, technique_preds, average='samples')
        }

        # Per-class report for techniques
        class_report = classification_report(
            technique_labels,
            technique_preds,
            target_names=self.technique_encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        return {
            'binary_metrics': binary_metrics,
            'technique_metrics': technique_metrics,
            'class_report': class_report
        }

    def _load_and_prepare_data(self, data_path: str) -> pd.DataFrame:
        """Загрузка и подготовка данных."""
        try:
            data = pd.read_csv(data_path)
            required_columns = ['article_id', 'sentence_id', 'sentence', 'technique', 'fragment']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Группировка по предложениям
            grouped_data = data.groupby(['article_id', 'sentence_id', 'sentence']).agg({
                'technique': list,
                'fragment': list
            }).reset_index()

            logger.info(f"Loaded {len(grouped_data)} sentences")
            return grouped_data

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def _save_models(self):
        """Сохранение моделей и всех необходимых параметров."""
        state = {
            # Модели
            'binary_model_state': self.binary_model.state_dict(),
            'technique_model_state': self.technique_model.state_dict(),
            'technique_encoder': self.technique_encoder,

            # Параметры для инициализации
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_filters': self.num_filters,
            'lstm_hidden': self.lstm_hidden,
            'max_length': self.max_length,

            # Параметры токенизатора
            'tokenizer_name': 'bert-base-uncased',  # или другое имя, если используется другой токенизатор

            # Дополнительные параметры
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'class_weights': self.class_weights
        }
        save_model(state, f"{self.model_path}/{self.model_name}")

    def _load_models(self):
        """Загрузка моделей и восстановление всех параметров."""
        checkpoint = load_model(f"{self.model_path}/{self.model_name}")

        # Восстанавливаем параметры
        self.vocab_size = checkpoint['vocab_size']
        self.embedding_dim = checkpoint['embedding_dim']
        self.num_filters = checkpoint['num_filters']
        self.lstm_hidden = checkpoint['lstm_hidden']
        self.max_length = checkpoint['max_length']
        self.batch_size = checkpoint['batch_size']
        self.learning_rate = checkpoint['learning_rate']
        self.warmup_steps = checkpoint['warmup_steps']
        self.class_weights = checkpoint['class_weights']

        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])

        # Загружаем бинарную модель
        self.binary_model = BinaryPropagandaModel(
            self.vocab_size,
            self.embedding_dim,
            self.num_filters,
            self.lstm_hidden
        ).to(self.device)
        self.binary_model.load_state_dict(checkpoint['binary_model_state'])

        # Загружаем модель техник
        num_classes = len(checkpoint['technique_encoder'].classes_)
        self.technique_model = TechniquePropagandaModel(
            self.vocab_size,
            self.embedding_dim,
            self.num_filters,
            self.lstm_hidden,
            num_classes
        ).to(self.device)
        self.technique_model.load_state_dict(checkpoint['technique_model_state'])

        # Загружаем энкодер
        self.technique_encoder = checkpoint['technique_encoder']

        # Переводим модели в режим оценки
        self.binary_model.eval()
        self.technique_model.eval()

    def _print_prediction_results(self, texts: List[str], results: List[Dict[str, Any]]) -> str:
        """
        Format prediction results into a string.

        Args:
            texts: List of input texts
            results: List of prediction results dictionaries

        Returns:
            str: Formatted string containing prediction results
        """
        output = ["Prediction Results:", "-" * 50]

        for text, result in zip(texts, results):
            # Add text analysis header
            output.append(f"\nText: {text}")

            if result['is_propaganda']:
                # Add propaganda detection results
                output.append(f"Propaganda detected (probability: {result['propaganda_probability']:.3f})")
                output.append(f"Confidence level: {result['confidence_level']}")

                if result['techniques']:
                    # Log all technique probabilities above threshold
                    output.append("\nAll detected techniques (probability > 0.1):")
                    for technique in result['techniques']:
                        output.append(f"  - {technique['name']}:")
                        output.append(f"    Probability: {technique['probability']:.3f}")
                        output.append(f"    Confidence: {technique['confidence']}")
                else:
                    output.append("No specific techniques detected with high confidence")
            else:
                output.append("No propaganda detected")

            # Add separator
            output.append("-" * 30)

        # Join all lines with newlines
        formatted_output = "\n".join(output)

        # Print to console
        # print(formatted_output)

        return formatted_output

    def predict(self, texts: List[str], print_results: bool = True) -> Tuple[List[Dict[str, Any]], str]:
        """
        Predict propaganda in texts with improved technique detection.

        Args:
            texts: List of texts to analyze
            print_results: Whether to print results to console

        Returns:
            Tuple containing:
            - List of prediction result dictionaries
            - Formatted string of prediction results
        """
        try:
            # Загружаем модели если нужно
            if not hasattr(self, 'binary_model') or not hasattr(self, 'technique_model'):
                self._load_models()

            self.binary_model.eval()
            self.technique_model.eval()

            text_base = texts

            if not check_lang_corpus(texts, "en"):
                texts = translate_corpus(texts)

            # Подготавливаем данные
            dataset = PropagandaDataset(
                pd.DataFrame({
                    'sentence': texts,
                    'technique': ['non-propaganda'] * len(texts),
                    'fragment': [None] * len(texts)
                }),
                self.tokenizer,
                max_length=self.max_length,
                binary_only=False
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

                    # Бинарная классификация
                    binary_outputs = self.binary_model(batch['input_ids'], batch['attention_mask'])
                    binary_probs = torch.softmax(binary_outputs, dim=1)

                    # Пороги уверенности
                    high_confidence = binary_probs[:, 1] > 0.7
                    medium_confidence = (binary_probs[:, 1] > 0.6) & (~high_confidence)
                    low_confidence = (binary_probs[:, 1] > 0.5) & (~high_confidence) & (~medium_confidence)

                    is_propaganda = high_confidence | medium_confidence | low_confidence

                    # Индексы пропагандистских текстов
                    propaganda_indices = is_propaganda.nonzero().squeeze().cpu()
                    if propaganda_indices.dim() == 0 and propaganda_indices.nelement() > 0:
                        propaganda_indices = [propaganda_indices.item()]
                    elif propaganda_indices.dim() == 1:
                        propaganda_indices = propaganda_indices.tolist()
                    else:
                        propaganda_indices = []

                    # Подготавливаем базовые результаты
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

                    # Анализ техник для пропагандистских текстов
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
                        technique_probs = torch.sigmoid(technique_outputs)  # Changed from softmax to sigmoid

                        for idx in propaganda_indices:
                            probs = technique_probs[idx]
                            techniques = []

                            # Анализ техник с вероятностью выше порога
                            for technique_idx, prob in enumerate(probs):
                                if prob > 0.1:  # Порог обнаружения техники
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

                            # Сортируем техники по вероятности
                            techniques.sort(key=lambda x: x['probability'], reverse=True)
                            batch_results[idx]['techniques'] = techniques
                            batch_results[idx]['num_techniques'] = len(techniques)

                            if techniques:
                                batch_results[idx]['primary_technique'] = techniques[0]['name']
                                batch_results[idx]['technique_confidence'] = techniques[0]['confidence']
                            else:
                                batch_results[idx]['primary_technique'] = 'unknown'
                                batch_results[idx]['technique_confidence'] = 'low'

                    results.extend(batch_results)

            # Форматируем и опционально печатаем результаты
            formatted_output = self._print_prediction_results(text_base, results) if print_results else ""

            return results, formatted_output

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _calculate_fragment_weights(self, input_ids, attention_mask):
        """Вычисление весов фрагментов на основе внимания и позиции."""
        attention_weights = attention_mask.float()
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_weights = torch.exp(-0.1 * position_ids)
        return attention_weights * position_weights.unsqueeze(0)

    def _calculate_class_weights(self, labels):
        """Вычисление весов классов."""
        if isinstance(labels, torch.Tensor):
            # Если labels это тензор
            unique_labels, counts = labels.unique(return_counts=True)
            weights = torch.FloatTensor([
                1.0 / (count.item() if count.item() > 0 else 1.0)
                for count in counts
            ])
        else:
            # Если labels это numpy массив
            counts = np.bincount(labels)
            weights = torch.FloatTensor([
                1.0 / (c if c > 0 else 1.0) for c in counts
            ])

        # Нормализация весов
        weights = weights / weights.sum()
        return weights.to(self.device)


if __name__ == "__main__":
    # Инициализация пайплайна
    pipeline = CascadePropagandaPipeline(
        model_path="../models",
        model_name="cpm_v5",
        batch_size=32,
        num_epochs=10,
        learning_rate=2e-5,
        warmup_steps=1000,
        max_length=512,
        class_weights=True,
        # device="cpu"
    )

    # Обучение и оценка
    try:
        logger.info("Starting training and evaluation...")
        metrics = pipeline.train_and_evaluate(
            data_path="../datasets/tasks-2-3/combined_dataset.csv"
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
        raise

    # Тестовые предсказания
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
        results, formatted = pipeline.predict(test_texts, True)
        print(formatted)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise
