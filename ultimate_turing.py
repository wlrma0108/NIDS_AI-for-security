"""
NIDS Advanced Tuning
====================

Ultimate 파이프라인 실행 후 추가 성능 향상을 위한 스크립트

1. Optuna 하이퍼파라미터 튜닝
2. Pseudo Labeling
3. Adversarial Validation

사용법:
    python nids_tuning.py --mode optuna
    python nids_tuning.py --mode pseudo
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
import pickle
import json
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Run: pip install optuna")


# ============================================================
# CONFIG
# ============================================================
TRAIN_PATH = "./train.csv"
TEST_PATH = "./test.csv"
FEATURE_ENGINEER_PATH = "./ultimate_output/feature_engineer.pkl"
OUTPUT_DIR = "./tuning_output"

N_FOLDS = 5
RANDOM_STATE = 42


# ============================================================
# OPTUNA TUNING
# ============================================================
def run_optuna_tuning(X, y, n_classes, n_trials=100):
    """Optuna로 LightGBM 하이퍼파라미터 튜닝"""
    
    if not OPTUNA_AVAILABLE:
        print("Optuna not available")
        return None
    
    print(f"\n{'='*60}")
    print(f"Optuna Hyperparameter Tuning ({n_trials} trials)")
    print(f"{'='*60}")
    
    def objective(trial):
        params = {
            'objective': 'multiclass',
            'num_class': n_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.95),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'class_weight': 'balanced',
            'verbose': -1,
            'n_jobs': -1,
            'random_state': RANDOM_STATE
        }
        
        # 3-Fold CV (속도를 위해)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params, train_data, num_boost_round=1000,
                valid_sets=[val_data], valid_names=['valid'],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            acc = accuracy_score(y_val, val_pred.argmax(axis=1))
            scores.append(acc)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest trial:")
    print(f"  Accuracy: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")
    
    return study.best_trial.params


def train_with_best_params(X, y, X_test, n_classes, best_params):
    """최적 파라미터로 5-Fold 학습"""
    
    print(f"\n{'='*60}")
    print("Training with Best Parameters")
    print(f"{'='*60}")
    
    params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'class_weight': 'balanced',
        'verbose': -1,
        'n_jobs': -1,
        'random_state': RANDOM_STATE,
        **best_params
    }
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_pred = np.zeros((len(X), n_classes))
    test_pred = np.zeros((len(X_test), n_classes))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{N_FOLDS}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params, train_data, num_boost_round=2000,
            valid_sets=[val_data], valid_names=['valid'],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        oof_pred[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred += model.predict(X_test) / N_FOLDS
        
        acc = accuracy_score(y_val, oof_pred[val_idx].argmax(axis=1))
        print(f"  Accuracy: {acc:.4f}")
    
    final_acc = accuracy_score(y, oof_pred.argmax(axis=1))
    print(f"\nFinal OOF Accuracy: {final_acc:.4f}")
    
    return oof_pred, test_pred, final_acc


# ============================================================
# PSEUDO LABELING
# ============================================================
def run_pseudo_labeling(X, y, X_test, n_classes, threshold=0.95, iterations=3):
    """Pseudo Labeling으로 추가 학습"""
    
    print(f"\n{'='*60}")
    print(f"Pseudo Labeling (threshold={threshold}, iterations={iterations})")
    print(f"{'='*60}")
    
    params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'max_depth': 10,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'class_weight': 'balanced',
        'verbose': -1,
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    }
    
    X_current = X.copy()
    y_current = y.copy()
    
    for iteration in range(iterations):
        print(f"\n--- Iteration {iteration + 1}/{iterations} ---")
        print(f"  Training samples: {len(X_current)}")
        
        # Train model
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + iteration)
        
        oof_pred = np.zeros((len(X_current), n_classes))
        test_pred = np.zeros((len(X_test), n_classes))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_current, y_current)):
            X_train, X_val = X_current[train_idx], X_current[val_idx]
            y_train, y_val = y_current[train_idx], y_current[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params, train_data, num_boost_round=1000,
                valid_sets=[val_data], valid_names=['valid'],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            if val_idx.max() < len(oof_pred):
                oof_pred[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            test_pred += model.predict(X_test) / N_FOLDS
        
        # 원본 데이터에 대한 정확도
        original_acc = accuracy_score(y[:len(y)], oof_pred[:len(y)].argmax(axis=1))
        print(f"  Original data accuracy: {original_acc:.4f}")
        
        # High confidence predictions
        max_probs = test_pred.max(axis=1)
        high_conf_mask = max_probs >= threshold
        n_high_conf = high_conf_mask.sum()
        
        print(f"  High confidence test samples: {n_high_conf} ({n_high_conf/len(X_test)*100:.1f}%)")
        
        if n_high_conf == 0:
            print("  No high confidence samples, stopping.")
            break
        
        # Add pseudo labels
        pseudo_X = X_test[high_conf_mask]
        pseudo_y = test_pred[high_conf_mask].argmax(axis=1)
        
        # Distribution of pseudo labels
        print(f"  Pseudo label distribution:")
        for cls in range(n_classes):
            count = (pseudo_y == cls).sum()
            if count > 0:
                print(f"    Class {cls}: {count}")
        
        # Combine
        X_current = np.vstack([X, pseudo_X])
        y_current = np.concatenate([y, pseudo_y])
        
        # Lower threshold for next iteration
        threshold = max(threshold - 0.05, 0.8)
    
    # Final prediction
    print(f"\n--- Final Training ---")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    final_oof = np.zeros((len(X), n_classes))
    final_test = np.zeros((len(X_test), n_classes))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Use augmented data for training
        X_train = np.vstack([X[train_idx], X_current[len(X):]])
        y_train = np.concatenate([y[train_idx], y_current[len(y):]])
        
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params, train_data, num_boost_round=1000,
            valid_sets=[val_data], valid_names=['valid'],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        final_oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        final_test += model.predict(X_test) / N_FOLDS
    
    final_acc = accuracy_score(y, final_oof.argmax(axis=1))
    print(f"\nFinal accuracy with pseudo labeling: {final_acc:.4f}")
    
    return final_oof, final_test, final_acc


# ============================================================
# ADVERSARIAL VALIDATION
# ============================================================
def adversarial_validation(X_train, X_test, feature_names=None):
    """Train/Test 분포 차이 확인"""
    
    print(f"\n{'='*60}")
    print("Adversarial Validation")
    print(f"{'='*60}")
    
    # Train=0, Test=1
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.array([0] * len(X_train) + [1] * len(X_test))
    
    # Binary classification
    from sklearn.model_selection import cross_val_score
    
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=5,
        random_state=RANDOM_STATE, verbose=-1
    )
    
    scores = cross_val_score(model, X_combined, y_combined, cv=5, scoring='roc_auc')
    
    print(f"\nAUC scores: {scores}")
    print(f"Mean AUC: {np.mean(scores):.4f}")
    
    if np.mean(scores) > 0.7:
        print("\n⚠️  WARNING: Train/Test distribution differs significantly!")
        print("   This might cause issues with generalization.")
        
        # Feature importance
        model.fit(X_combined, y_combined)
        importance = model.feature_importances_
        
        if feature_names is not None:
            fi = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("\nTop features causing distribution shift:")
            for _, row in fi.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    else:
        print("\n✓ Train/Test distributions are similar. Good!")
    
    return np.mean(scores)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all', choices=['optuna', 'pseudo', 'adversarial', 'all'])
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    args = parser.parse_args()
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    
    # Load feature engineer
    if Path(FEATURE_ENGINEER_PATH).exists():
        print("Loading feature engineer...")
        from ultimate import UltimateFeatureEngineer
        fe = UltimateFeatureEngineer.load(FEATURE_ENGINEER_PATH)
        X, y, _ = fe.fit_transform(df_train, 'attack_cat')
        X_test, _ = fe.transform(df_test)
    else:
        print("Feature engineer not found. Running basic preprocessing...")
        # Basic preprocessing
        le = LabelEncoder()
        y = le.fit_transform(df_train['attack_cat'].fillna('Normal'))
        
        drop_cols = ['id', 'label', 'attack_cat']
        X = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns]).values
        X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns]).values
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)
        
        fe = None
    
    n_classes = len(np.unique(y))
    print(f"Data loaded: {X.shape}, {n_classes} classes")
    
    # Run selected mode
    if args.mode in ['adversarial', 'all']:
        feature_names = fe.feature_names if fe else None
        adversarial_validation(X, X_test, feature_names)
    
    if args.mode in ['optuna', 'all']:
        best_params = run_optuna_tuning(X, y, n_classes, n_trials=args.trials)
        
        if best_params:
            with open(output_dir / 'best_params.json', 'w') as f:
                json.dump(best_params, f, indent=2)
            
            oof_pred, test_pred, acc = train_with_best_params(X, y, X_test, n_classes, best_params)
            
            np.save(output_dir / 'optuna_oof.npy', oof_pred)
            np.save(output_dir / 'optuna_test.npy', test_pred)
    
    if args.mode in ['pseudo', 'all']:
        oof_pred, test_pred, acc = run_pseudo_labeling(X, y, X_test, n_classes)
        
        np.save(output_dir / 'pseudo_oof.npy', oof_pred)
        np.save(output_dir / 'pseudo_test.npy', test_pred)
    
    print("\nDone!")


if __name__ == "__main__":
    main()