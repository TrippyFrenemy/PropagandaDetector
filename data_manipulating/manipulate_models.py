import pickle
from typing import List, Tuple, Any

import joblib
import numpy as np
import xgboost as xgb
from imblearn.ensemble import (
    BalancedRandomForestClassifier, RUSBoostClassifier,
    EasyEnsembleClassifier, BalancedBaggingClassifier
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def print_model_info(model: Any) -> None:
    """
    Print information about a model.

    Args:
        model: Trained model to inspect
    """
    print(f"\nModel type: {type(model).__name__}")

    if hasattr(model, 'get_params'):
        print("\nModel parameters:")
        for param, value in model.get_params().items():
            print(f"  {param}: {value}")

    if hasattr(model, 'feature_importances_'):
        print("\nFeature importances:")
        for i, importance in enumerate(model.feature_importances_):
            print(f"  Feature {i}: {importance:.4f}")


def get_best_threshold(y_true: Any, y_proba: Any) -> float:
    """
    Find the best probability threshold using F1 score.

    Args:
        y_true: True labels
        y_proba: Prediction probabilities

    Returns:
        Optimal threshold value
    """
    thresholds = np.linspace(0, 1, 100)
    best_f1 = -1
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def create_base_models() -> List[Tuple[str, Any]]:
    """Create optimized base models for the stacking ensemble."""
    return [
        # XGBoost with optimized parameters
        ('xgb', xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=2,
            tree_method='hist',  # More stable than 'gpu_hist'
            random_state=42,
            n_jobs=-1
        )),

        # Balanced Random Forest
        ('balanced_rf', BalancedRandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            max_features='sqrt',
            min_samples_split=5,
            min_samples_leaf=2,
            sampling_strategy='auto',
            random_state=42,
            n_jobs=-1
        )),

        # Easy Ensemble with optimized base estimator
        ('easy_ensemble', EasyEnsembleClassifier(
            n_estimators=100,
            estimator=DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=2
            ),
            random_state=42,
            n_jobs=-1
        )),

        # Balanced Bagging with optimized parameters
        ('balanced_bagging', BalancedBaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=2
            ),
            n_estimators=100,
            sampling_strategy='auto',
            replacement=True,
            random_state=42,
            n_jobs=-1
        )),

        # Neural Network with reduced complexity
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )),

        # Gradient Boosting with conservative parameters
        ('gradient_boosting', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            max_features=0.8,
            random_state=42
        )),

        # Extra Trees with balanced weights
        ('extra_trees', ExtraTreesClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ]


def create_meta_classifier() -> Any:
    """Create meta-classifier with conservative parameters."""
    return LogisticRegression(
        C=0.1,
        penalty='l2',
        solver='saga',
        max_iter=15000,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )


def save_model(model, file_name="model"):
    with open(f'{file_name}.pkl', 'wb') as file:
        pickle.dump(model, file)


def load_model(file_name="model"):
    with open(f'{file_name}.pkl', 'rb') as file:
        model = pickle.load(file)
        return model


def save_vectorizer(model, file_name="model"):
    joblib.dump(model, f'{file_name}.joblib')


def load_vectorizer(file_name="model"):
    return joblib.load(f'{file_name}.joblib')
