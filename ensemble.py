"""
NIDS Kaggle-Style Pipeline
===========================

LightGBM + CatBoost + XGBoost + 5-Fold CV + Ensemble

사용법:
    1. CONFIG 섹션에서 경로 설정
    2. python nids_ensemble.py 실행

출력:
    - OOF predictions (validation)
    - Test predictions (ensemble)
    - Feature importance
    - 각 모델별/Fold별 성능
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import pickle
import json
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
class Config:
    # 경로
    TRAIN_DATA_PATH = "./train.csv"
    TEST_DATA_PATH = "./test.csv"
    OUTPUT_DIR = "./ensemble_output"
    
    # 데이터
    LABEL_COL = "attack_cat"
    
    # CV 설정
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # 모델 사용 여부
    USE_LGBM = True
    USE_XGB = True
    USE_CATBOOST = True
    
    # Ensemble 가중치 (None이면 자동 계산)
    ENSEMBLE_WEIGHTS = None  # e.g., {'lgbm': 0.4, 'xgb': 0.3, 'catboost': 0.3}
    
config = Config()

# ============================================================
# FEATURE ENGINEER
# ============================================================
class FeatureEngineer:
    """피처 엔지니어링 + 전처리"""
    
    def __init__(self):
        self.label_encoders = {}
        self.target_encoder = None
        self.feature_names = []
        self.categorical_cols = ['proto', 'service', 'state']
        self.cat_indices = []
        
    def fit_transform(self, df: pd.DataFrame, label_col: str = None):
        df = df.copy()
        
        # 레이블 처리
        y = None
        if label_col and label_col in df.columns:
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(df[label_col].fillna('Normal'))
            df = df.drop(columns=[label_col])
        
        # ID 제거
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        if 'label' in df.columns:
            df = df.drop(columns=['label'])
        
        # 기본 전처리
        df = self._preprocess(df)
        
        # 피처 엔지니어링
        df = self._create_features(df)
        
        # Categorical 인코딩 (LightGBM/CatBoost용으로 유지)
        df = self._encode_categorical(df, fit=True)
        
        self.feature_names = df.columns.tolist()
        self.cat_indices = [i for i, col in enumerate(self.feature_names) 
                           if col in self.categorical_cols]
        
        return df.values, y
    
    def transform(self, df: pd.DataFrame):
        df = df.copy()
        
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        if 'label' in df.columns:
            df = df.drop(columns=['label'])
        if 'attack_cat' in df.columns:
            df = df.drop(columns=['attack_cat'])
        
        df = self._preprocess(df)
        df = self._create_features(df)
        df = self._encode_categorical(df, fit=False)
        
        # 컬럼 순서 맞추기
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        
        return df.values
    
    def _preprocess(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('-').astype(str)
        return df
    
    def _create_features(self, df):
        """피처 엔지니어링"""
        
        # 1. Ratio features
        df['bytes_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
        df['pkts_ratio'] = df['spkts'] / (df['dpkts'] + 1)
        df['load_ratio'] = df['sload'] / (df['dload'] + 1)
        
        # 2. Total features
        df['total_bytes'] = df['sbytes'] + df['dbytes']
        df['total_pkts'] = df['spkts'] + df['dpkts']
        df['total_loss'] = df['sloss'] + df['dloss']
        
        # 3. Per-packet features
        df['sbytes_per_pkt'] = df['sbytes'] / (df['spkts'] + 1)
        df['dbytes_per_pkt'] = df['dbytes'] / (df['dpkts'] + 1)
        
        # 4. Rate features
        df['bytes_per_sec'] = df['total_bytes'] / (df['dur'] + 1e-6)
        df['pkts_per_sec'] = df['total_pkts'] / (df['dur'] + 1e-6)
        
        # 5. ct_* 관련
        ct_cols = [c for c in df.columns if c.startswith('ct_')]
        if ct_cols:
            df['ct_sum'] = df[ct_cols].sum(axis=1)
            df['ct_mean'] = df[ct_cols].mean(axis=1)
            df['ct_max'] = df[ct_cols].max(axis=1)
            df['ct_std'] = df[ct_cols].std(axis=1).fillna(0)
        
        # 6. TTL features
        df['ttl_ratio'] = df['sttl'] / (df['dttl'] + 1)
        df['ttl_diff'] = df['sttl'] - df['dttl']
        
        # 7. TCP features
        df['tcp_setup_ratio'] = df['synack'] / (df['tcprtt'] + 1e-6)
        
        # 8. Window features
        df['window_ratio'] = df['swin'] / (df['dwin'] + 1)
        
        # 9. Jitter features
        df['jit_ratio'] = df['sjit'] / (df['djit'] + 1)
        
        # 10. Log transforms (skewed features)
        for col in ['sbytes', 'dbytes', 'sload', 'dload', 'rate', 'total_bytes']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
        
        return df
    
    def _encode_categorical(self, df, fit=True):
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if fit:
                self.label_encoders[col] = LabelEncoder()
                vals = list(df[col].unique()) + ['<UNK>']
                self.label_encoders[col].fit(vals)
            
            known = set(self.label_encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known else '<UNK>')
            df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    @classmethod
    def load(cls, path):
        fe = cls()
        with open(path, 'rb') as f:
            fe.__dict__ = pickle.load(f)
        return fe


# ============================================================
# MODEL CONFIGS
# ============================================================
def get_lgbm_params(n_classes):
    return {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'class_weight': 'balanced',
        'verbose': -1,
        'n_jobs': -1,
        'random_state': config.RANDOM_STATE
    }

def get_xgb_params(n_classes):
    return {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'tree_method': 'hist',
        'random_state': config.RANDOM_STATE,
        'n_jobs': -1
    }

def get_catboost_params(n_classes):
    return {
        'iterations': 500,
        'learning_rate': 0.10,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': config.RANDOM_STATE,
        'auto_class_weights': 'Balanced',
        'early_stopping_rounds': 100,
        'verbose': False
    }


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_lgbm_fold(X_train, y_train, X_val, y_val, params, cat_indices):
    """단일 Fold LightGBM 학습"""
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_indices)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_indices, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=0)  # 로그 끄기
        ]
    )
    
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    return model, val_pred


def train_xgb_fold(X_train, y_train, X_val, y_val, params):
    """단일 Fold XGBoost 학습"""
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Class weight 계산
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    sample_weights = class_weights[y_train]
    dtrain.set_weight(sample_weights)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    val_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
    
    return model, val_pred


def train_catboost_fold(X_train, y_train, X_val, y_val, params, cat_indices, feature_names):
    """단일 Fold CatBoost 학습"""
    
    # CatBoost는 DataFrame으로 전달해야 categorical feature 처리 가능
    # 또는 cat_features 없이 학습 (이미 label encoded 되어있으므로)
    
    # cat_indices가 비어있거나, 이미 숫자로 인코딩된 경우 cat_features 사용 안함
    if len(cat_indices) == 0:
        model = CatBoostClassifier(**params)
    else:
        # DataFrame으로 변환하여 categorical 처리
        train_df = pd.DataFrame(X_train, columns=feature_names)
        val_df = pd.DataFrame(X_val, columns=feature_names)
        
        # Categorical columns를 int로 변환 (CatBoost가 인식하도록)
        cat_cols = [feature_names[i] for i in cat_indices]
        for col in cat_cols:
            train_df[col] = train_df[col].astype(int)
            val_df[col] = val_df[col].astype(int)
        
        model = CatBoostClassifier(**params, cat_features=cat_cols)
        
        model.fit(
            train_df, y_train,
            eval_set=(val_df, y_val),
            use_best_model=True
        )
        
        val_pred = model.predict_proba(val_df)
        return model, val_pred
    
    # cat_features 없이 학습
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    val_pred = model.predict_proba(X_val)
    
    return model, val_pred


# ============================================================
# CROSS VALIDATION
# ============================================================
def run_cv(X, y, X_test, fe, n_classes):
    """5-Fold CV 실행"""
    
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    
    # OOF predictions 저장
    oof_preds = {}
    test_preds = {}
    models = {}
    scores = {}
    
    if config.USE_LGBM:
        oof_preds['lgbm'] = np.zeros((len(X), n_classes))
        test_preds['lgbm'] = np.zeros((len(X_test), n_classes))
        models['lgbm'] = []
        scores['lgbm'] = []
    
    if config.USE_XGB:
        oof_preds['xgb'] = np.zeros((len(X), n_classes))
        test_preds['xgb'] = np.zeros((len(X_test), n_classes))
        models['xgb'] = []
        scores['xgb'] = []
    
    if config.USE_CATBOOST:
        oof_preds['catboost'] = np.zeros((len(X), n_classes))
        test_preds['catboost'] = np.zeros((len(X_test), n_classes))
        models['catboost'] = []
        scores['catboost'] = []
    
    # Feature importance 누적
    feature_importance = pd.DataFrame()
    
    print(f"\n{'='*60}")
    print(f"Starting {config.N_FOLDS}-Fold Cross Validation")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/{config.N_FOLDS}")
        print(f"{'='*40}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # -------------------- LightGBM --------------------
        if config.USE_LGBM:
            print(f"\n[LightGBM] Training...")
            start = time.time()
            
            lgbm_params = get_lgbm_params(n_classes)
            model, val_pred = train_lgbm_fold(
                X_train, y_train, X_val, y_val, 
                lgbm_params, fe.cat_indices
            )
            
            oof_preds['lgbm'][val_idx] = val_pred
            test_preds['lgbm'] += model.predict(X_test) / config.N_FOLDS
            models['lgbm'].append(model)
            
            fold_acc = accuracy_score(y_val, val_pred.argmax(axis=1))
            scores['lgbm'].append(fold_acc)
            
            print(f"  Accuracy: {fold_acc:.4f} | Time: {time.time()-start:.1f}s")
            
            # Feature importance
            fi = pd.DataFrame({
                'feature': fe.feature_names,
                'importance': model.feature_importance(importance_type='gain'),
                'fold': fold,
                'model': 'lgbm'
            })
            feature_importance = pd.concat([feature_importance, fi], ignore_index=True)
        
        # -------------------- XGBoost --------------------
        if config.USE_XGB:
            print(f"\n[XGBoost] Training...")
            start = time.time()
            
            xgb_params = get_xgb_params(n_classes)
            model, val_pred = train_xgb_fold(
                X_train, y_train, X_val, y_val, 
                xgb_params
            )
            
            oof_preds['xgb'][val_idx] = val_pred
            dtest = xgb.DMatrix(X_test)
            test_preds['xgb'] += model.predict(dtest) / config.N_FOLDS
            models['xgb'].append(model)
            
            fold_acc = accuracy_score(y_val, val_pred.argmax(axis=1))
            scores['xgb'].append(fold_acc)
            
            print(f"  Accuracy: {fold_acc:.4f} | Time: {time.time()-start:.1f}s")
            
            # Feature importance
            importance_dict = model.get_score(importance_type='gain')
            fi = pd.DataFrame({
                'feature': fe.feature_names,
                'importance': [importance_dict.get(f'f{i}', 0) for i in range(len(fe.feature_names))],
                'fold': fold,
                'model': 'xgb'
            })
            feature_importance = pd.concat([feature_importance, fi], ignore_index=True)
        
        # -------------------- CatBoost --------------------
        if config.USE_CATBOOST:
            print(f"\n[CatBoost] Training...")
            start = time.time()
            
            catboost_params = get_catboost_params(n_classes)
            model, val_pred = train_catboost_fold(
                X_train, y_train, X_val, y_val,
                catboost_params, fe.cat_indices, fe.feature_names
            )
            
            oof_preds['catboost'][val_idx] = val_pred
            
            # Test prediction
            if len(fe.cat_indices) > 0:
                test_df = pd.DataFrame(X_test, columns=fe.feature_names)
                cat_cols = [fe.feature_names[i] for i in fe.cat_indices]
                for col in cat_cols:
                    test_df[col] = test_df[col].astype(int)
                test_preds['catboost'] += model.predict_proba(test_df) / config.N_FOLDS
            else:
                test_preds['catboost'] += model.predict_proba(X_test) / config.N_FOLDS
            
            models['catboost'].append(model)
            
            fold_acc = accuracy_score(y_val, val_pred.argmax(axis=1))
            scores['catboost'].append(fold_acc)
            
            print(f"  Accuracy: {fold_acc:.4f} | Time: {time.time()-start:.1f}s")
            
            # Feature importance
            fi = pd.DataFrame({
                'feature': fe.feature_names,
                'importance': model.feature_importances_,
                'fold': fold,
                'model': 'catboost'
            })
            feature_importance = pd.concat([feature_importance, fi], ignore_index=True)
    
    return oof_preds, test_preds, models, scores, feature_importance


# ============================================================
# ENSEMBLE
# ============================================================
def ensemble_predictions(oof_preds, test_preds, y_true, class_names):
    """OOF 기반 앙상블 가중치 최적화"""
    
    print(f"\n{'='*60}")
    print("Ensemble Optimization")
    print(f"{'='*60}")
    
    model_names = list(oof_preds.keys())
    n_models = len(model_names)
    
    # 각 모델의 OOF accuracy
    model_scores = {}
    for name in model_names:
        acc = accuracy_score(y_true, oof_preds[name].argmax(axis=1))
        model_scores[name] = acc
        print(f"\n{name} OOF Accuracy: {acc:.4f}")
    
    # 가중치 결정
    if config.ENSEMBLE_WEIGHTS:
        weights = config.ENSEMBLE_WEIGHTS
    else:
        # Accuracy 기반 가중치 (softmax)
        accs = np.array([model_scores[name] for name in model_names])
        exp_accs = np.exp(accs * 10)  # temperature scaling
        weights = {name: w for name, w in zip(model_names, exp_accs / exp_accs.sum())}
    
    print(f"\nEnsemble weights:")
    for name, w in weights.items():
        print(f"  {name}: {w:.4f}")
    
    # 앙상블 예측
    oof_ensemble = np.zeros_like(list(oof_preds.values())[0])
    test_ensemble = np.zeros_like(list(test_preds.values())[0])
    
    for name in model_names:
        oof_ensemble += weights[name] * oof_preds[name]
        test_ensemble += weights[name] * test_preds[name]
    
    # 앙상블 성능
    ensemble_acc = accuracy_score(y_true, oof_ensemble.argmax(axis=1))
    print(f"\nEnsemble OOF Accuracy: {ensemble_acc:.4f}")
    
    # Classification report
    print(f"\n{'='*60}")
    print("Ensemble Classification Report (OOF)")
    print(f"{'='*60}")
    print(classification_report(
        y_true, 
        oof_ensemble.argmax(axis=1), 
        target_names=class_names,
        digits=4
    ))
    
    return oof_ensemble, test_ensemble, weights, ensemble_acc


# ============================================================
# MAIN
# ============================================================
def main():
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("NIDS Ensemble Pipeline")
    print("="*60)
    print(f"Models: LGBM={config.USE_LGBM}, XGB={config.USE_XGB}, CatBoost={config.USE_CATBOOST}")
    print(f"Folds: {config.N_FOLDS}")
    
    # -------------------- Data Loading --------------------
    print(f"\n[1/5] Loading Data...")
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    print(f"  Train: {df_train.shape}")
    
    df_test = None
    test_ids = None
    if Path(config.TEST_DATA_PATH).exists():
        df_test = pd.read_csv(config.TEST_DATA_PATH)
        test_ids = df_test['id'].values if 'id' in df_test.columns else np.arange(len(df_test))
        print(f"  Test: {df_test.shape}")
    
    # -------------------- Feature Engineering --------------------
    print(f"\n[2/5] Feature Engineering...")
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df_train, config.LABEL_COL)
    
    class_names = fe.target_encoder.classes_
    n_classes = len(class_names)
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Classes: {n_classes}")
    
    for i, name in enumerate(class_names):
        count = (y == i).sum()
        print(f"    {i}: {name} ({count:,})")
    
    # Test transform
    X_test = None
    if df_test is not None:
        X_test = fe.transform(df_test)
        print(f"  Test features: {X_test.shape}")
    else:
        # Test 없으면 빈 배열
        X_test = np.zeros((0, X.shape[1]))
    
    # Save feature engineer
    fe.save(output_dir / 'feature_engineer.pkl')
    
    # Class mapping
    class_mapping = {int(i): str(c) for i, c in enumerate(class_names)}
    with open(output_dir / 'class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # -------------------- Cross Validation --------------------
    print(f"\n[3/5] Cross Validation...")
    oof_preds, test_preds, models, scores, feature_importance = run_cv(
        X, y, X_test, fe, n_classes
    )
    
    # -------------------- CV Summary --------------------
    print(f"\n{'='*60}")
    print("CV Summary")
    print(f"{'='*60}")
    
    for name, fold_scores in scores.items():
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f"\n{name}:")
        for i, s in enumerate(fold_scores):
            print(f"  Fold {i+1}: {s:.4f}")
        print(f"  Mean: {mean_score:.4f} (+/- {std_score:.4f})")
    
    # -------------------- Ensemble --------------------
    print(f"\n[4/5] Ensemble...")
    oof_ensemble, test_ensemble, weights, ensemble_acc = ensemble_predictions(
        oof_preds, test_preds, y, class_names
    )
    
    # -------------------- Save Results --------------------
    print(f"\n[5/5] Saving Results...")
    
    # OOF predictions
    np.save(output_dir / 'oof_ensemble.npy', oof_ensemble)
    for name, pred in oof_preds.items():
        np.save(output_dir / f'oof_{name}.npy', pred)
    
    # Test predictions
    if len(X_test) > 0:
        np.save(output_dir / 'test_ensemble.npy', test_ensemble)
        
        # CSV 저장
        pred_labels = test_ensemble.argmax(axis=1)
        pred_names = [class_names[p] for p in pred_labels]
        
        result_df = pd.DataFrame({
            'id': test_ids,
            'prediction': pred_names,
            'prediction_idx': pred_labels
        })
        
        for i, name in enumerate(class_names):
            result_df[f'prob_{name}'] = test_ensemble[:, i]
        
        result_df.to_csv(output_dir / 'predictions.csv', index=False)
        print(f"  predictions.csv saved")
    
    # Feature importance
    fi_summary = feature_importance.groupby(['model', 'feature'])['importance'].mean().reset_index()
    fi_summary = fi_summary.sort_values(['model', 'importance'], ascending=[True, False])
    fi_summary.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # Scores summary
    results_summary = {
        'n_folds': config.N_FOLDS,
        'ensemble_accuracy': ensemble_acc,
        'weights': weights,
        'model_scores': {name: {'mean': np.mean(s), 'std': np.std(s), 'folds': s} 
                        for name, s in scores.items()}
    }
    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    # Confusion matrix
    cm = confusion_matrix(y, oof_ensemble.argmax(axis=1))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_dir / 'confusion_matrix.csv')
    
    # -------------------- Final Summary --------------------
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}")
    
    print(f"\nFinal Ensemble OOF Accuracy: {ensemble_acc:.4f}")
    
    print(f"\nPer-Class Accuracy (OOF Ensemble):")
    oof_pred_labels = oof_ensemble.argmax(axis=1)
    for i, name in enumerate(class_names):
        mask = y == i
        if mask.sum() > 0:
            acc = (oof_pred_labels[mask] == y[mask]).mean()
            print(f"  {name}: {acc:.4f} ({mask.sum()} samples)")
    
    print(f"\nTop 10 Important Features (LightGBM):")
    if 'lgbm' in scores:
        lgbm_fi = fi_summary[fi_summary['model'] == 'lgbm'].head(10)
        for _, row in lgbm_fi.iterrows():
            print(f"  {row['feature']}: {row['importance']:.2f}")
    
    print(f"\nOutput files in {output_dir}:")
    print(f"  - predictions.csv (test predictions)")
    print(f"  - oof_ensemble.npy (OOF predictions)")
    print(f"  - feature_importance.csv")
    print(f"  - results_summary.json")
    print(f"  - confusion_matrix.csv")
    print(f"  - feature_engineer.pkl")
    
    if len(X_test) > 0:
        print(f"\nTest Prediction Summary:")
        pred_counts = pd.Series(pred_names).value_counts()
        for name, count in pred_counts.items():
            print(f"  {name}: {count} ({count/len(pred_names)*100:.2f}%)")
    
    return oof_ensemble, test_ensemble, models, fe


if __name__ == "__main__":
    main()