from typing import Dict, Tuple, Any
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

from .metrics import calculate_metrics


def create_samplers() -> Dict[str, Any]:
    """Create dictionary of sampling strategies."""
    return {
        "smote": SMOTE(random_state=42),
        "adasyn": ADASYN(random_state=42),
        "borderline": BorderlineSMOTE(random_state=42),
        "random_over": RandomOverSampler(random_state=42),
        "random_under": RandomUnderSampler(random_state=42),
        "nearmiss": NearMiss(version=3),
        "tomek": TomekLinks(),
        "smoteenn": SMOTEENN(random_state=42),
        "smotetomek": SMOTETomek(random_state=42)
    }


def create_advanced_samplers() -> Dict[str, Any]:
    """Create dictionary of advanced sampling strategies including combinations."""
    basic_samplers = {
        "smote": SMOTE(random_state=42),
        "adasyn": ADASYN(random_state=42),
        "borderline": BorderlineSMOTE(random_state=42),
        "random_over": RandomOverSampler(random_state=42),
        "random_under": RandomUnderSampler(random_state=42),
        "nearmiss": NearMiss(version=3),
        "tomek": TomekLinks(),
        "enn": EditedNearestNeighbours(),
        "smoteenn": SMOTEENN(random_state=42),
        "smotetomek": SMOTETomek(random_state=42)
    }

    # Add combined strategies
    combined_samplers = {
        # SMOTE followed by Random Undersampling
        "smote_random_under": ImbPipeline([
            ('oversample', SMOTE(random_state=42)),
            ('undersample', RandomUnderSampler(random_state=42))
        ]),

        # SMOTE followed by NearMiss
        "smote_nearmiss": ImbPipeline([
            ('oversample', SMOTE(random_state=42)),
            ('undersample', NearMiss(version=3))
        ]),

        # BorderlineSMOTE followed by Tomek Links
        "borderline_tomek": ImbPipeline([
            ('oversample', BorderlineSMOTE(random_state=42)),
            ('undersample', TomekLinks())
        ]),

        # ADASYN followed by ENN
        "adasyn_enn": ImbPipeline([
            ('oversample', ADASYN(random_state=42)),
            ('undersample', EditedNearestNeighbours())
        ])
    }

    return {**basic_samplers, **combined_samplers}


def evaluate_sampling_strategies(
        samplers: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        technique: str,
        strategy: str = "auto"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate different sampling strategies and select the best one.

    Args:
        samplers: Dictionary of sampling strategies
        X: Features
        y: Labels
        technique: Name of propaganda technique
        strategy: Sampling strategy to use

    Returns:
        Resampled features and labels
    """
    if strategy != "auto":
        sampler = samplers.get(strategy)
        if sampler is None:
            print(f"Strategy {strategy} not found, using SMOTE")
            sampler = samplers["smote"]
        return sampler.fit_resample(X, y)

    # Split data for strategy evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_score = -1
    best_strategy = None
    best_X = None
    best_y = None
    results = {}

    class_dist = Counter(y)
    imbalance_ratio = max(class_dist.values()) / min(class_dist.values())
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")

    # Try each strategy
    for name, sampler in samplers.items():
        try:
            print(f"\nEvaluating {name}...")
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

            # Check if the sampling helped with imbalance
            new_dist = Counter(y_resampled)
            new_ratio = max(new_dist.values()) / min(new_dist.values())

            # Quick evaluation
            model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
            model.fit(X_resampled, y_resampled)

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            metrics = calculate_metrics(y_val, y_pred, y_proba)

            score = metrics['roc_auc'] * 0.5 + metrics['pr_auc'] * 0.3 + metrics['f1_score'] * 0.2

            ratio_penalty = max(0, (new_ratio - 3) / 10)  # Start penalizing if ratio > 3
            final_score = score * (1 - ratio_penalty)

            results[name] = {
                'score': final_score,
                'metrics': metrics,
                'distribution': new_dist,
                'imbalance_ratio': new_ratio
            }

            print(f"ROC AUC: {metrics['roc_auc']:.3f}")
            print(f"PR AUC: {metrics['pr_auc']:.3f}")
            print(f"F1 Score: {metrics['f1_score']:.3f}")
            print(f"Final score (with ratio penalty): {final_score:.3f}")
            print(f"New class distribution: {dict(new_dist)}")
            print(f"New imbalance ratio: {new_ratio:.2f}")

            if score > best_score:
                best_score = final_score
                best_strategy = name
                # Apply best strategy to full dataset
                best_X, best_y = sampler.fit_resample(X, y)

        except Exception as e:
            print(f"Strategy {name} failed: {str(e)}")
            continue

    print("\nSampling strategy evaluation results:")
    for name, result in results.items():
        print(f"{name:20} - Score: {result['score']:.3f} - Imbalance ratio: {result['imbalance_ratio']:.2f}")

    print(f"\nSelected strategy: {best_strategy} (score: {best_score:.3f})")

    if best_X is None:
        print("All strategies failed, using original data")
        return X, y

    return best_X, best_y
