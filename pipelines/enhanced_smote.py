import copy
import logging
from typing import List, Optional, Tuple, Dict
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import TfidfVectorizer

from data_manipulating.feature_extractor import TextFeatureExtractor
from data_manipulating.manipulate_models import save_model, load_model
from pipelines.cascade_classification import (
    CascadePropagandaPipeline,
    BinaryPropagandaModel,
    TechniquePropagandaModel,
    FocalLoss,
)

from utils.translate import check_lang_corpus, translate_corpus

logger = logging.getLogger(__name__)


class SMOTEResampledDataset(Dataset):
    """
    Dataset для данных, пересемплированных SMOTE.
    Поддерживает как бинарную задачу, так и 'technique' задачу.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 is_binary: bool = True,
                 technique_encoder=None):
        super().__init__()
        self.X = X
        self.y = y
        self.is_binary = is_binary
        self.technique_encoder = technique_encoder

        # Для мультиклассовой (мульти-меточной) задачи
        if not is_binary and technique_encoder is not None:
            self.num_classes = len(technique_encoder.classes_)
            self.technique_labels = []
            # Преобразуем single-label индексы в multi-hot вектор
            for label_idx in self.y:
                label_vec = torch.zeros(self.num_classes)
                label_vec[label_idx] = 1.0
                self.technique_labels.append(label_vec)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feature_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.is_binary:
            return {
                "feature_vector": feature_tensor,
                "binary_label": torch.tensor(self.y[idx], dtype=torch.long),
                "is_smote_sample": True
            }
        else:
            return {
                "feature_vector": feature_tensor,
                "technique_label": self.technique_labels[idx],
                "is_smote_sample": True
            }


class EnhancedSmotePropagandaPipeline(CascadePropagandaPipeline):
    """
    Pipeline, расширяющий CascadePropagandaPipeline и использующий SMOTE для
    балансировки классов. Ключевое отличие — мы отдельно сохраняем/загружаем
    признаки/векторизатор для бинарной модели и для техники.
    """

    def __init__(
            self,
            *args,
            use_extra_features: bool = True,
            use_smote: bool = True,
            smote_ratio: float = 0.8,
            smote_k_neighbors: int = 5,
            random_state: int = 42,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_extra_features = use_extra_features
        self.use_smote = use_smote
        self.smote_ratio = smote_ratio
        self.smote_k_neighbors = smote_k_neighbors
        self.random_state = random_state

        # Храним векторизаторы/скейлеры отдельно для двух задач:
        self.smote_vectorizer_binary = None
        self.smote_vectorizer_technique = None
        # Аналогично — размерности:
        self.smote_feature_dim_binary = None
        self.smote_feature_dim_technique = None

    def create_datasets(self, data: pd.DataFrame):
        if self.use_smote:
            return self._create_balanced_datasets(data)
        else:
            return super().create_datasets(data)

    def _create_balanced_datasets(self, data: pd.DataFrame):
        logger.info("Creating balanced datasets with SMOTE")

        # 1) Бинарный датасет
        binary_dataset = self._prepare_binary_dataset_with_smote(data)

        # 2) Датасет для техник
        propaganda_mask = data['technique'].apply(
            lambda x: isinstance(x, list) and any(t != 'non-propaganda' for t in x)
            if isinstance(x, list)
            else x != 'non-propaganda'
        )
        technique_data = data[propaganda_mask].copy()
        technique_dataset = self._prepare_technique_dataset_with_smote(technique_data)

        # Сохраняем encoder техник
        self.technique_encoder = technique_dataset.technique_encoder

        logger.info(f"Binary dataset size: {len(binary_dataset)}")
        logger.info(f"Technique dataset size: {len(technique_dataset)}")

        return binary_dataset, technique_dataset

    def _prepare_binary_dataset_with_smote(self, data: pd.DataFrame):
        texts = data['sentence'].tolist()
        binary_labels = []
        for techniques in data['technique']:
            if isinstance(techniques, list):
                is_prop = any(t != 'non-propaganda' for t in techniques)
            else:
                is_prop = (techniques != 'non-propaganda')
            binary_labels.append(int(is_prop))

        logger.info(f"Original binary distribution: {Counter(binary_labels)}")

        # Извлекаем TF-IDF и доп. фичи, сохраняем в self.smote_vectorizer_binary
        X_text, X_extra = self._extract_features_for_smote_binary(texts)
        y = np.array(binary_labels)

        # SMOTE-ресемплинг
        X_res, y_res = self._apply_smote(X_text, y, X_extra)

        logger.info(f"After SMOTE binary distribution: {Counter(y_res)}")

        # Создаём датасет
        dataset = SMOTEResampledDataset(X_res, y_res, is_binary=True)
        return dataset

    def _prepare_technique_dataset_with_smote(self, data: pd.DataFrame):
        texts = data['sentence'].tolist()

        # Создаём LabelEncoder для техник
        from sklearn.preprocessing import LabelEncoder
        all_techniques = set()
        for t in data['technique']:
            if isinstance(t, list):
                all_techniques.update(t)
            else:
                all_techniques.add(t)
        technique_encoder = LabelEncoder()
        # удаляем 'non-propaganda'
        techniques_list = sorted(all_techniques - {'non-propaganda'})
        technique_encoder.fit(['non-propaganda'] + techniques_list)

        # Определяем "основную" технику
        primary_techniques = []
        for t in data['technique']:
            if isinstance(t, list):
                props = [x for x in t if x != 'non-propaganda']
                primary_techniques.append(props[0] if props else 'non-propaganda')
            else:
                primary_techniques.append(t)
        technique_indices = technique_encoder.transform(primary_techniques)
        logger.info(f"Original technique distribution: {Counter(technique_indices)}")

        # Извлекаем TF-IDF и доп. фичи, сохраняем в self.smote_vectorizer_technique
        X_text, X_extra = self._extract_features_for_smote_technique(texts)
        y = np.array(technique_indices)

        # SMOTE
        X_res, y_res = self._apply_smote(
            X_text, y, X_extra,
            sampling_strategy='minority',
            k_neighbors=min(3, min(Counter(y).values()) - 1)
        )
        logger.info(f"After SMOTE technique distribution: {Counter(y_res)}")

        dataset = SMOTEResampledDataset(
            X_res, y_res,
            is_binary=False,
            technique_encoder=technique_encoder
        )
        dataset.technique_encoder = technique_encoder
        return dataset

    def _extract_features_for_smote_binary(self, texts: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Тут создаём TfidfVectorizer и делаем fit_transform для бинарной задачи.
        Сохраняем результат в self.smote_vectorizer_binary.
        """
        logger.info("Extracting SMOTE features for BINARY classification")
        max_features = 500  # Пример
        vect = TfidfVectorizer(max_features=max_features, stop_words='english')
        X_text = vect.fit_transform(texts).toarray()
        self.smote_vectorizer_binary = vect

        # Доп. фичи (при необходимости)
        X_extra = None
        if self.use_extra_features:
            X_extra = self._compute_extra_features(texts)
        return X_text, X_extra

    def _extract_features_for_smote_technique(self, texts: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Аналогично, но для техники.
        Создаём отдельный векторизатор self.smote_vectorizer_technique.
        """
        logger.info("Extracting SMOTE features for TECHNIQUE classification")
        max_features = 700  # Пример, может отличаться
        vect = TfidfVectorizer(max_features=max_features, stop_words='english')
        X_text = vect.fit_transform(texts).toarray()
        self.smote_vectorizer_technique = vect

        X_extra = None
        if self.use_extra_features:
            X_extra = self._compute_extra_features(texts)
        return X_text, X_extra

    def _compute_extra_features(self, texts: List[str]) -> np.ndarray:
        extractor = TextFeatureExtractor()
        feats_list = []
        for t in tqdm(texts, desc="Extra features"):
            _, f = extractor.process_text_with_features(t)
            # Пример: собираем 7-10 нужных фич
            ling = f['linguistic']
            emot = f['emotional']
            stat = f['statistical']
            # и т.д.
            vector = [
                ling.get('avg_sentence_length', 0.0),
                ling.get('sentence_length_std', 0.0),
                ling.get('avg_word_length', 0.0),
                ling.get('avg_syllables_per_word', 0.0),
                ling.get('unique_words_ratio', 0.0),
                emot.get('polarity', 0.0),
                emot.get('subjectivity', 0.0),
                stat.get('type_token_ratio', 0.0),
            ]
            feats_list.append(vector)

        X_extra = np.array(feats_list, dtype=np.float32)
        # Можно дополнительно scaler.fit_transform, но тогда нужно сохранить scaler
        # и восстанавливать при инференсе. Ниже — упрощённый вариант без нормализации.
        return X_extra

    def _apply_smote(self, X: np.ndarray, y: np.ndarray,
                     extra: Optional[np.ndarray] = None,
                     sampling_strategy='auto', k_neighbors=None):
        if extra is not None:
            X_combined = np.hstack([X, extra])
        else:
            X_combined = X

        min_samples = min(Counter(y).values())
        if k_neighbors is None:
            k_neighbors = min(self.smote_k_neighbors, min_samples - 1)
        k_neighbors = max(1, k_neighbors)

        sm = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=self.random_state
        )
        try:
            X_res, y_res = sm.fit_resample(X_combined, y)
            return X_res, y_res
        except Exception as e:
            logger.error(f"SMOTE failed: {e}")
            return X_combined, y

    def _evaluate_models(self, binary_val_loader, technique_val_loader):
        """
        Evaluate both models (binary and technique) with support for SMOTE-based
        (feature_vector) and standard (input_ids/attention_mask) batches.
        """
        import numpy as np
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
            classification_report
        )

        self.binary_model.eval()
        self.technique_model.eval()

        binary_preds = []
        binary_labels = []
        binary_probs = []

        with torch.no_grad():
            for batch in binary_val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Если SMOTE-датасет - используем 'feature_vector'
                if 'feature_vector' in batch:
                    outputs = self.binary_model(batch['feature_vector'])
                else:
                    # Обычный датасет — берем input_ids, attention_mask, опционально extra_features
                    extra_features = batch.get('extra_features')
                    outputs = self.binary_model(batch['input_ids'], batch['attention_mask'], extra_features)

                probs = torch.softmax(outputs, dim=1)

                binary_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                binary_labels.extend(batch['binary_label'].cpu().numpy())
                binary_probs.extend(probs[:, 1].cpu().numpy())

        # Метрики для бинарной классификации
        binary_metrics = {
            'accuracy': accuracy_score(binary_labels, binary_preds),
            'precision': precision_score(binary_labels, binary_preds, average='weighted', zero_division=0),
            'recall': recall_score(binary_labels, binary_preds, average='weighted', zero_division=0),
            'f1': f1_score(binary_labels, binary_preds, average='weighted', zero_division=0)
        }
        # Добавляем ROC AUC, если есть более одного класса
        unique_classes = np.unique(binary_labels)
        if len(unique_classes) > 1:
            binary_metrics['roc_auc'] = roc_auc_score(binary_labels, binary_probs)

        technique_preds = []
        technique_labels = []
        technique_probs = []

        with torch.no_grad():
            for batch in technique_val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Аналогичная логика для SMOTE
                if 'feature_vector' in batch:
                    outputs = self.technique_model(batch['feature_vector'])
                else:
                    # Обычный датасет — берем input_ids, attention_mask и fragment_weights
                    outputs = self.technique_model(
                        batch['input_ids'],
                        batch['attention_mask'],
                        batch['fragment_weights']
                    )
                probs = torch.sigmoid(outputs)  # Мульти-лейбл → сигмоида
                preds = (probs > 0.5).float()

                technique_preds.append(preds.cpu().numpy())
                technique_labels.append(batch['technique_label'].cpu().numpy())
                technique_probs.append(probs.cpu().numpy())

        # Приводим к массивам
        technique_preds = np.vstack(technique_preds)
        technique_labels = np.vstack(technique_labels)
        technique_probs = np.vstack(technique_probs)

        # Метрики техник: sample-based (подходит для многометочной классификации)
        technique_metrics = {
            'accuracy': accuracy_score(technique_labels, technique_preds),
            'precision': precision_score(technique_labels, technique_preds, average='samples', zero_division=0),
            'recall': recall_score(technique_labels, technique_preds, average='samples', zero_division=0),
            'f1': f1_score(technique_labels, technique_preds, average='samples', zero_division=0)
        }

        # Detailed per-class report
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

    def train_binary_model(self, train_loader, val_loader):
        """
        Если датасет SMOTE, строим Sequential модель по размеру feature_dim,
        иначе — BinaryPropagandaModel.
        """
        sample_batch = next(iter(train_loader))
        using_smote = 'is_smote_sample' in sample_batch
        logger.info(f"Train binary model with SMOTE={using_smote}")

        if using_smote:
            feature_dim = sample_batch['feature_vector'].shape[1]
            # Собираем модель
            model = nn.Sequential(
                nn.Linear(feature_dim, self.lstm_hidden * 2),
                nn.LayerNorm(self.lstm_hidden * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.lstm_hidden * 2, self.lstm_hidden),
                nn.LayerNorm(self.lstm_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.lstm_hidden, 2)
            ).to(self.device)
            self.smote_feature_dim_binary = feature_dim
        else:
            model = BinaryPropagandaModel(
                self.vocab_size,
                self.embedding_dim,
                self.num_filters,
                self.lstm_hidden,
                self.binary_k_range
            ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_epochs_binary * len(train_loader)
        )
        # FocalLoss
        all_labels = []
        for b in train_loader:
            all_labels.extend(b['binary_label'].cpu().numpy())
        class_weights = self._calculate_class_weights(np.array(all_labels)) if self.class_weights else None
        criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(self.device)

        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0

        for epoch in range(self.num_epochs_binary):
            model.train()
            train_loss = 0.0
            train_preds, train_labels = [], []

            for batch in tqdm(train_loader, desc=f"[Binary] Epoch {epoch + 1}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()

                if using_smote:
                    outputs = model(batch['feature_vector'])
                else:
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
            val_loss = 0.0
            val_preds, val_labels_ = [], []

            with torch.no_grad():
                for vb in val_loader:
                    vb = {k: v.to(self.device) for k, v in vb.items()}
                    if using_smote:
                        vouts = model(vb['feature_vector'])
                    else:
                        vouts = model(vb['input_ids'], vb['attention_mask'])

                    vloss = criterion(vouts, vb['binary_label'])
                    val_loss += vloss.item()
                    val_preds.extend(vouts.argmax(dim=1).cpu().numpy())
                    val_labels_.extend(vb['binary_label'].cpu().numpy())

            # Метрики
            epoch_val_loss = val_loss / len(val_loader)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model = copy.deepcopy(model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping (binary)")
                    break

        return best_model

    def train_technique_model(self, train_loader, val_loader, num_classes):
        """
        Аналогично для техник. Отличие — выводим multi-label.
        """
        sample_batch = next(iter(train_loader))
        using_smote = 'is_smote_sample' in sample_batch
        logger.info(f"Train technique model with SMOTE={using_smote}")

        if using_smote:
            feature_dim = sample_batch['feature_vector'].shape[1]
            model = nn.Sequential(
                nn.Linear(feature_dim, self.lstm_hidden * 2),
                nn.LayerNorm(self.lstm_hidden * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.lstm_hidden * 2, self.lstm_hidden),
                nn.LayerNorm(self.lstm_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.lstm_hidden, num_classes)
            ).to(self.device)
            self.smote_feature_dim_technique = feature_dim
        else:
            model = TechniquePropagandaModel(
                self.vocab_size,
                self.embedding_dim,
                self.num_filters,
                self.lstm_hidden,
                num_classes,
                self.technique_k_range
            ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_epochs_technique * len(train_loader)
        )
        criterion = nn.BCEWithLogitsLoss().to(self.device)

        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0

        for epoch in range(self.num_epochs_technique):
            model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"[Technique] Epoch {epoch + 1}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()

                if using_smote:
                    outputs = model(batch['feature_vector'])
                else:
                    outputs = model(batch['input_ids'],
                                    batch['attention_mask'],
                                    batch['fragment_weights'])
                loss = criterion(outputs, batch['technique_label'].float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            # Валидация
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vb in val_loader:
                    vb = {k: v.to(self.device) for k, v in vb.items()}
                    if using_smote:
                        vouts = model(vb['feature_vector'])
                    else:
                        vouts = model(vb['input_ids'],
                                      vb['attention_mask'],
                                      vb['fragment_weights'])
                    loss_tech = criterion(vouts, vb['technique_label'].float())
                    val_loss += loss_tech.item()

            epoch_val_loss = val_loss / len(val_loader)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model = copy.deepcopy(model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping (technique)")
                    break

        return best_model

    def _save_models(self):
        import pickle
        state = {
            'binary_model_state': self.binary_model.state_dict(),
            'technique_model_state': self.technique_model.state_dict(),
            'technique_encoder': self.technique_encoder,

            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_filters': self.num_filters,
            'lstm_hidden': self.lstm_hidden,
            'max_length': self.max_length,
            'binary_k_range': self.binary_k_range,
            'technique_k_range': self.technique_k_range,
            'tokenizer_name': 'bert-base-uncased',

            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'class_weights': self.class_weights,

            'use_extra_features': self.use_extra_features,
            'use_smote': self.use_smote,
            'smote_ratio': self.smote_ratio,
            'smote_k_neighbors': self.smote_k_neighbors,
            'random_state': self.random_state,

            'model_type': 'enhanced_smote',
            # Размерности для binary/technique
            'smote_feature_dim_binary': self.smote_feature_dim_binary,
            'smote_feature_dim_technique': self.smote_feature_dim_technique,
            # Сохраняем векторизаторы
            'smote_vectorizer_binary': pickle.dumps(
                self.smote_vectorizer_binary) if self.smote_vectorizer_binary else None,
            'smote_vectorizer_technique': pickle.dumps(
                self.smote_vectorizer_technique) if self.smote_vectorizer_technique else None,
        }
        save_model(state, f"{self.model_path}/{self.model_name}")
        logger.info(f"Model saved to {self.model_path}/{self.model_name}")

    def _load_models(self):
        import pickle
        checkpoint = load_model(f"{self.model_path}/{self.model_name}")

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
        self.use_smote = checkpoint.get('use_smote', True)
        self.smote_ratio = checkpoint.get('smote_ratio', 0.8)
        self.smote_k_neighbors = checkpoint.get('smote_k_neighbors', 5)
        self.random_state = checkpoint.get('random_state', 42)

        self.binary_k_range = checkpoint.get('binary_k_range', [3, 4, 5])
        self.technique_k_range = checkpoint.get('technique_k_range', [2, 3, 4, 5])
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint.get('tokenizer_name', 'bert-base-uncased'))

        model_type = checkpoint.get('model_type', 'standard')

        # Восстанавливаем раздельные feature_dim
        self.smote_feature_dim_binary = checkpoint.get('smote_feature_dim_binary', None)
        self.smote_feature_dim_technique = checkpoint.get('smote_feature_dim_technique', None)

        # Восстанавливаем векторизаторы
        if checkpoint.get('smote_vectorizer_binary'):
            self.smote_vectorizer_binary = pickle.loads(checkpoint['smote_vectorizer_binary'])
        if checkpoint.get('smote_vectorizer_technique'):
            self.smote_vectorizer_technique = pickle.loads(checkpoint['smote_vectorizer_technique'])

        # Сборка двух моделей (binary / technique)
        if model_type == 'enhanced_smote' and (self.smote_feature_dim_binary is not None):
            # 1) Бинарная
            bf = self.smote_feature_dim_binary
            self.binary_model = nn.Sequential(
                nn.Linear(bf, self.lstm_hidden * 2),
                nn.LayerNorm(self.lstm_hidden * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.lstm_hidden * 2, self.lstm_hidden),
                nn.LayerNorm(self.lstm_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.lstm_hidden, 2)
            ).to(self.device)
        else:
            # Обычная бинарная
            self.binary_model = BinaryPropagandaModel(
                self.vocab_size,
                self.embedding_dim,
                self.num_filters,
                self.lstm_hidden,
                self.binary_k_range
            ).to(self.device)

        # Модель для техник
        if model_type == 'enhanced_smote' and (self.smote_feature_dim_technique is not None):
            tf = self.smote_feature_dim_technique
            num_classes = len(checkpoint['technique_encoder'].classes_)
            self.technique_model = nn.Sequential(
                nn.Linear(tf, self.lstm_hidden * 2),
                nn.LayerNorm(self.lstm_hidden * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.lstm_hidden * 2, self.lstm_hidden),
                nn.LayerNorm(self.lstm_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.lstm_hidden, num_classes)
            ).to(self.device)
        else:
            num_classes = 2
            if 'technique_encoder' in checkpoint:
                num_classes = len(checkpoint['technique_encoder'].classes_)
            self.technique_model = TechniquePropagandaModel(
                self.vocab_size,
                self.embedding_dim,
                self.num_filters,
                self.lstm_hidden,
                num_classes,
                self.technique_k_range
            ).to(self.device)

        # Загружаем веса
        try:
            self.binary_model.load_state_dict(checkpoint['binary_model_state'])
            self.technique_model.load_state_dict(checkpoint['technique_model_state'])
        except Exception as e:
            raise RuntimeError(f"Error loading model state dict: {e}")

        self.technique_encoder = checkpoint['technique_encoder']
        self.binary_model.eval()
        self.technique_model.eval()
        logger.info(f"Loaded model: {model_type} from {self.model_path}/{self.model_name}")

    def predict(self, texts: List[str], print_results: bool = True):
        """
        При SMOTE-моделях нужно сформировать фичи точно так же,
        используя соответствующие векторизаторы.
        """
        # Если не загружены, загрузим
        if not hasattr(self, 'binary_model') or not hasattr(self, 'technique_model'):
            self._load_models()

        # Переводим язык при необходимости
        original_texts = texts
        if not check_lang_corpus(texts, "en"):
            texts = translate_corpus(texts)

        is_smote_model = (self.smote_feature_dim_binary is not None) and (self.smote_feature_dim_technique is not None)
        logger.info(f"Predict with SMOTE model = {is_smote_model}")

        # --- Бинарные предсказания ---
        if is_smote_model:
            # Используем сохранённый векторизатор для binary
            if not self.smote_vectorizer_binary:
                raise RuntimeError("No smote_vectorizer_binary found in checkpoint")

            X_text_binary = self.smote_vectorizer_binary.transform(texts).toarray()
            X_extra_binary = self._compute_extra_features(texts) if self.use_extra_features else None
            if X_extra_binary is not None:
                X_combined_binary = np.hstack([X_text_binary, X_extra_binary])
            else:
                X_combined_binary = X_text_binary

            # Приводим размер к self.smote_feature_dim_binary (если нужно)
            bdim = self.smote_feature_dim_binary
            if X_combined_binary.shape[1] != bdim:
                logger.warning(f"Binary feats shape {X_combined_binary.shape[1]} != expected {bdim}. Trunc/pad.")
                if X_combined_binary.shape[1] > bdim:
                    X_combined_binary = X_combined_binary[:, :bdim]
                else:
                    pad = np.zeros((X_combined_binary.shape[0], bdim - X_combined_binary.shape[1]))
                    X_combined_binary = np.hstack([X_combined_binary, pad])

            features_tensor_bin = torch.tensor(X_combined_binary, dtype=torch.float32)
            # батчами
            results = []
            bs = self.batch_size
            self.binary_model.eval()

            with torch.no_grad():
                for i in range(0, len(features_tensor_bin), bs):
                    batch_feats = features_tensor_bin[i: i + bs].to(self.device)
                    outputs = self.binary_model(batch_feats)
                    probs = torch.softmax(outputs, dim=1)

                    for j in range(batch_feats.size(0)):
                        prob_propag = probs[j, 1].item()
                        results.append({
                            'propaganda_probability': prob_propag,
                            'is_propaganda': (prob_propag > 0.5),
                            'confidence_level': (
                                'high' if prob_propag > 0.7
                                else 'medium' if prob_propag > 0.6
                                else 'low' if prob_propag > 0.5
                                else 'none'
                            ),
                            'techniques': []
                        })

            indices_prop = [i for i, r in enumerate(results) if r['is_propaganda']]
            if indices_prop:
                # Извлекаем признаки для техники
                if not self.smote_vectorizer_technique:
                    raise RuntimeError("No smote_vectorizer_technique found in checkpoint")

                X_text_tech = self.smote_vectorizer_technique.transform(texts).toarray()
                X_extra_tech = self._compute_extra_features(texts) if self.use_extra_features else None
                if X_extra_tech is not None:
                    X_combined_tech = np.hstack([X_text_tech, X_extra_tech])
                else:
                    X_combined_tech = X_text_tech

                # Приводим к нужной размерности
                tdim = self.smote_feature_dim_technique
                if X_combined_tech.shape[1] != tdim:
                    logger.warning(f"Technique feats shape {X_combined_tech.shape[1]} != expected {tdim}. Trunc/pad.")
                    if X_combined_tech.shape[1] > tdim:
                        X_combined_tech = X_combined_tech[:, :tdim]
                    else:
                        pad = np.zeros((X_combined_tech.shape[0], tdim - X_combined_tech.shape[1]))
                        X_combined_tech = np.hstack([X_combined_tech, pad])

                feats_tech = torch.tensor(X_combined_tech, dtype=torch.float32).to(self.device)
                self.technique_model.eval()

                with torch.no_grad():
                    # Прогоняем только индексы, где is_propaganda = True
                    # но поскольку у нас нет выбора отдельных примеров, придётся предсказать всё,
                    # а потом взять из outputs нужные
                    outputs_tech = self.technique_model(feats_tech)
                    sigm = torch.sigmoid(outputs_tech)

                # Для каждого текста, где пропаганда, получаем вероятности техник
                for idx in indices_prop:
                    probs_ = sigm[idx].cpu().numpy()  # shape: [num_classes]
                    technique_list = []
                    for c_i, pr in enumerate(probs_):
                        if pr > 0.1:  # порог
                            tech_name = self.technique_encoder.inverse_transform([c_i])[0]
                            if tech_name != 'non-propaganda':
                                conf = (
                                    'high' if pr > 0.6
                                    else 'medium' if pr > 0.4
                                    else 'low'
                                )
                                technique_list.append({
                                    'name': tech_name,
                                    'probability': float(pr),
                                    'confidence': conf
                                })
                    technique_list.sort(key=lambda x: x['probability'], reverse=True)
                    results[idx]['techniques'] = technique_list

            formatted_str = ""

            # Форматируем вывод
            if print_results:
                formatted_str = self._print_prediction_results(original_texts, results)
            return results, formatted_str
        else:
            # Если модель не SMOTE, вызываем стандартный predict родительского класса
            return super().predict(texts, print_results)


if __name__ == "__main__":
    # Initialize enhanced pipeline with SMOTE
    pipeline = EnhancedSmotePropagandaPipeline(
        model_path="../models",
        model_name="ecpm_smote_v1",
        batch_size=32,
        num_epochs_binary=15,
        num_epochs_technique=10,
        learning_rate=2e-5,
        warmup_steps=1000,
        max_length=512,
        class_weights=True,

        # SMOTE-specific parameters
        use_smote=True,
        smote_ratio=0.8,
        smote_k_neighbors=5,
        random_state=42,

        # Model architecture parameters
        use_extra_features=True,
        binary_k_range=[2, 3, 4, 5],
        technique_k_range=[3, 4, 5],
    )

    # Train and evaluate
    try:
        logger.info("Starting training and evaluation with SMOTE...")
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
        "Why we fight - for our children's bread."
    ]

    try:
        logger.info("Running predictions on test texts...")
        results, formatted = pipeline.predict(test_texts, True)
        print(formatted)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise
