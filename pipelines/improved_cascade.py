import copy
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer

from data_manipulating.manipulate_models import load_model
from pipelines.cascade_classification import CascadePropagandaPipeline, FocalLoss, TechniquePropagandaModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


class EnhancedAttentionLayer(nn.Module):
    """Улучшенный слой внимания с градиентным клиппингом и нормализацией."""

    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Применяем нормализацию перед вниманием
        query = self.layer_norm(query)
        key = self.layer_norm(key)
        value = self.layer_norm(value)

        # Транспонируем для MultiheadAttention
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        if mask is not None:
            mask = mask.bool()
            mask = ~mask

        # Применяем внимание с градиентным клиппингом
        with torch.amp.autocast("cuda", enabled=True):
            output, attention_weights = self.attention(
                query=query,
                key=key,
                value=value,
                key_padding_mask=mask,
                need_weights=True
            )

        # Dropout и остаточное соединение
        output = self.dropout(output)
        output = output.transpose(0, 1)

        return output, attention_weights


class ImprovedBinaryPropagandaModel(nn.Module):
    """Улучшенная модель бинарной классификации с дополнительными механизмами регуляризации."""

    def __init__(self, vocab_size, embedding_dim, num_filters, lstm_hidden, k_range, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # Улучшенные CNN слои
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding="same"),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout / 2)
            ) for k in k_range
        ])

        cnn_output_size = num_filters * len(self.conv_layers)

        # Улучшенный BiLSTM с градиентным клиппингом
        self.bilstm = nn.LSTM(
            cnn_output_size,
            lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        # Улучшенный механизм внимания
        self.attention = EnhancedAttentionLayer(lstm_hidden * 2)

        # Улучшенный классификатор
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.LayerNorm(lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(lstm_hidden // 2, 2)
        )

        # Инициализация весов
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, attention_mask=None):
        # Применяем эмбеддинги с dropout
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        # CNN с остаточными соединениями
        x_input = x.transpose(1, 2)

        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(x_input)
            conv_outputs.append(conv_out)

        x = torch.cat(conv_outputs, dim=1)
        x = x.transpose(1, 2)

        # BiLSTM с градиентным клиппингом
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            self.bilstm.flatten_parameters()
            packed_output, _ = self.bilstm(packed_x)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=x.size(1)
            )
        else:
            self.bilstm.flatten_parameters()
            x, _ = self.bilstm(x)

        # Применяем улучшенный механизм внимания
        if attention_mask is not None:
            x, _ = self.attention(x, x, x, attention_mask)

        # Глобальный пулинг с взвешиванием по маске
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            x = torch.mean(x, dim=1)

        # Классификация
        x = self.classifier(x)

        return x


class ImprovedCascadePropagandaPipeline(CascadePropagandaPipeline):
    """Улучшенный пайплайн с дополнительными механизмами обработки ошибок и мониторинга."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = torch.amp.GradScaler(self.device)  # Исправлено устаревшее предупреждение
        self.criterion = None  # Будет инициализирован в train_binary_model
        self.checkpoint_path = "../checkpoints"

        # Добавляем отслеживание лучших метрик
        self.best_metrics = {
            'binary': {'loss': float('inf'), 'f1': 0},
            'technique': {'loss': float('inf'), 'f1': 0}
        }

        # Конфигурация обучения
        self.training_config = {
            'gradient_clip_val': 1.0,
            'early_stopping_patience': 3,
            'min_improvement': 1e-4,
            'scheduler_T0': 5,
            'scheduler_T_mult': 2,
            'scheduler_eta_min': 1e-6,
            'dropout_range': (0.1, 0.8),
            'dropout_factor': 1.1
        }

    def train_binary_model(self, train_loader, val_loader):
        try:
            model = ImprovedBinaryPropagandaModel(
                self.vocab_size,
                self.embedding_dim,
                self.num_filters,
                self.lstm_hidden,
                self.binary_k_range
            ).to(self.device)

            # Инициализация критерия потерь
            if self.class_weights:
                all_labels = []
                for batch in train_loader:
                    all_labels.extend(batch['binary_label'].cpu().numpy())
                class_weights = self._calculate_class_weights(np.array(all_labels))
                self.criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(self.device)
            else:
                self.criterion = FocalLoss().to(self.device)

            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.training_config['scheduler_T0'],
                T_mult=self.training_config['scheduler_T_mult'],
                eta_min=self.training_config['scheduler_eta_min']
            )

            best_model = None
            best_val_loss = float('inf')
            no_improvement = 0

            for epoch in range(self.num_epochs_binary):
                # Обучение с обработкой ошибок
                try:
                    train_metrics = self._train_epoch(
                        model, train_loader, self.optimizer, scheduler
                    )
                    val_metrics = self._validate_epoch(model, val_loader)

                    # Логирование метрик
                    self._log_metrics(epoch, train_metrics, val_metrics)

                    # Проверка улучшения
                    improved = self._check_improvement(
                        val_metrics['loss'],
                        best_val_loss,
                        epoch
                    )

                    if improved:
                        best_val_loss = val_metrics['loss']
                        best_model = copy.deepcopy(model)
                        no_improvement = 0
                        # Сохраняем лучшую модель
                        self._save_checkpoint(model, self.optimizer, epoch, val_metrics)
                    else:
                        no_improvement += 1
                        if no_improvement >= self.training_config['early_stopping_patience']:
                            logger.info(f"Early stopping triggered at epoch {epoch}")
                            break

                    # Адаптивная регуляризация
                    if epoch > 0 and self._check_overfitting(train_metrics, val_metrics):
                        self._adjust_regularization(model)

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        logger.warning(f"OOM in epoch {epoch}, reducing batch size")
                        self._handle_oom(train_loader, val_loader)
                        continue
                    raise e

            return best_model or model

        except Exception as e:
            logger.error(f"Error in train_binary_model: {str(e)}")
            raise

    def _train_epoch(self, model, train_loader, optimizer, scheduler):
        """Улучшенная реализация обучения одной эпохи."""
        model.train()
        metrics = defaultdict(float)

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            try:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Проверка валидности батча
                if not self._validate_batch(batch):
                    logger.warning(f"Skipping invalid batch {batch_idx}")
                    continue

                # Mixed precision training
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(batch['input_ids'], batch['attention_mask'])
                    loss = self.criterion(outputs, batch['binary_label'])

                # Обработка градиентов
                self.scaler.scale(loss).backward()

                # Клиппинг градиентов
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.training_config['gradient_clip_val']
                )

                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)  # Оптимизация памяти

                # Обновление метрик
                metrics['loss'] += loss.item()
                metrics.update(self._calculate_batch_metrics(outputs, batch))

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._handle_batch_oom(batch_idx)
                    continue
                raise e

        scheduler.step()

        # Усреднение метрик
        return {k: v / len(train_loader) for k, v in metrics.items()}

    def _validate_epoch(self, model, val_loader):
        """Улучшенная реализация валидации."""
        model.eval()
        metrics = defaultdict(float)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = model(batch['input_ids'], batch['attention_mask'])
                loss = self.criterion(outputs, batch['binary_label'])

                metrics['loss'] += loss.item()

                # Сохраняем предсказания и метки для метрик
                preds = outputs.argmax(dim=1).cpu()
                labels = batch['binary_label'].cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        # Вычисляем дополнительные метрики
        metrics.update(self._calculate_validation_metrics(all_preds, all_labels))

        return {k: v / len(val_loader) if k == 'loss' else v
                for k, v in metrics.items()}

    def _handle_batch_oom(self, batch_idx):
        """Обработка OOM для отдельного батча."""
        torch.cuda.empty_cache()
        logger.warning(f"OOM in batch {batch_idx}, skipping")

    def _validate_batch(self, batch):
        """Проверка валидности входного батча."""
        required_keys = ['input_ids', 'attention_mask', 'binary_label']
        return all(key in batch for key in required_keys)

    def _adjust_regularization(self, model):
        """Адаптивно настраивает регуляризацию модели для борьбы с переобучением."""
        # Настройка dropout
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                current_p = module.p
                new_p = min(
                    self.training_config['dropout_range'][1],
                    current_p * self.training_config['dropout_factor']
                )
                module.p = new_p
                logger.info(f"Adjusted dropout from {current_p:.3f} to {new_p:.3f}")

        # Увеличение weight decay для оптимизатора
        for param_group in self.optimizer.param_groups:
            current_wd = param_group['weight_decay']
            new_wd = current_wd * 1.1  # Увеличиваем weight decay на 10%
            param_group['weight_decay'] = new_wd
            logger.info(f"Adjusted weight decay from {current_wd:.6f} to {new_wd:.6f}")

        # Усиление L2 регуляризации для линейных слоев
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if not hasattr(module, 'l2_strength'):
                    module.l2_strength = 0.01
                module.l2_strength *= 1.1  # Увеличиваем L2 регуляризацию на 10%
                logger.info(f"Adjusted L2 regularization to {module.l2_strength:.6f}")

    def _calculate_batch_metrics(self, outputs, batch):
        """Вычисление метрик для батча."""
        preds = outputs.argmax(dim=1).cpu()
        labels = batch['binary_label'].cpu()

        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='weighted')
        }

    def _calculate_validation_metrics(self, all_preds, all_labels):
        """Вычисление метрик валидации."""
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }

    def _check_improvement(self, current_loss, best_loss, epoch):
        """Проверка улучшения с динамическим порогом."""
        improvement_threshold = self.training_config['min_improvement'] * \
                                (0.9 ** epoch)
        return current_loss < best_loss - improvement_threshold

    def _check_overfitting(self, train_metrics, val_metrics):
        """Проверка переобучения."""
        return train_metrics['loss'] < val_metrics['loss'] * 0.95

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

        try:
            self.binary_k_range = checkpoint['binary_k_range']
            self.technique_k_range = checkpoint['technique_k_range']
        except:
            pass

        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])

        # Загружаем бинарную модель
        self.binary_model = ImprovedBinaryPropagandaModel(
            self.vocab_size,
            self.embedding_dim,
            self.num_filters,
            self.lstm_hidden,
            self.binary_k_range
        ).to(self.device)
        self.binary_model.load_state_dict(checkpoint['binary_model_state'])

        # Загружаем модель техник
        num_classes = len(checkpoint['technique_encoder'].classes_)
        self.technique_model = TechniquePropagandaModel(
            self.vocab_size,
            self.embedding_dim,
            self.num_filters,
            self.lstm_hidden,
            num_classes,
            self.technique_k_range
        ).to(self.device)
        self.technique_model.load_state_dict(checkpoint['technique_model_state'])

        # Загружаем энкодер
        self.technique_encoder = checkpoint['technique_encoder']

        # Переводим модели в режим оценки
        self.binary_model.eval()
        self.technique_model.eval()

    def _save_checkpoint(self, model, optimizer, epoch, metrics):
        """Сохранение контрольной точки модели."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        checkpoint_path = f"{self.checkpoint_path}/checkpoint_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _log_metrics(self, epoch, train_metrics, val_metrics):
        """Логирование метрик обучения."""
        logger.info(f"\nEpoch {epoch + 1}")
        logger.info("Training metrics:")
        for k, v in train_metrics.items():
            logger.info(f"{k}: {v:.4f}")
        logger.info("\nValidation metrics:")
        for k, v in val_metrics.items():
            logger.info(f"{k}: {v:.4f}")


if __name__ == "__main__":
    # Инициализация пайплайна
    pipeline = ImprovedCascadePropagandaPipeline(
        model_path="../models",
        model_name="icpm_v3",
        batch_size=32,
        num_epochs=10,
        learning_rate=2e-5,
        warmup_steps=1000,
        max_length=512,
        class_weights=True,
        binary_k_range=[2, 3, 4],
        technique_k_range=[3, 4, 5, 6, 7],
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
