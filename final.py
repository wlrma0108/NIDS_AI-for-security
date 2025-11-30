"""
NIDS Ultimate Pipeline - Target 90%
=====================================

전략:
1. Confusion Matrix 분석 → 문제 클래스 파악
2. 2-Stage Classification → Hard class 분리
3. Neural Network (MLP) → 다양성 추가
4. Class-wise Best Model Selection
5. Threshold Optimization
6. Pseudo Labeling (optional)

사용법:
    python nids_ultimate.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
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
    OUTPUT_DIR = "./ultimate_output"
    
    LABEL_COL = "attack_cat"
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # 모델 사용 여부
    USE_LGBM = True
    USE_XGB = True
    USE_CATBOOST = True
    USE_EXTRATREES = True
    USE_NEURAL_NET = True
    
    # 2-Stage Classification
    USE_TWO_STAGE = True
    
    # Class-wise Ensemble
    USE_CLASSWISE_ENSEMBLE = True
    
    # Threshold Optimization
    USE_THRESHOLD_OPT = True
    
    # Pseudo Labeling
    USE_PSEUDO_LABELING = False  # 시간 오래 걸림
    PSEUDO_THRESHOLD = 0.95
    
    # Neural Net 설정
    NN_HIDDEN_DIMS = [256, 128, 64]
    NN_DROPOUT = 0.3
    NN_EPOCHS = 50
    NN_BATCH_SIZE = 512
    NN_LR = 1e-3
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

# ============================================================
# FEATURE ENGINEER (이전과 동일 + 추가)
# ============================================================
class UltimateFeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoder = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.categorical_cols = ['proto', 'service', 'state']
        self.target_encoding_maps = {}
        self.frequency_encoding_maps = {}
        self.global_target_mean = None
        
    def fit_transform(self, df: pd.DataFrame, label_col: str = None):
        df = df.copy()
        
        y = None
        if label_col and label_col in df.columns:
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(df[label_col].fillna('Normal'))
            df['_target_binary'] = (df[label_col] != 'Normal').astype(int)
        
        ids = df['id'].values if 'id' in df.columns else None
        df = df.drop(columns=['id', 'label', label_col] if label_col else ['id', 'label'], errors='ignore')
        
        df = self._preprocess(df)
        df = self._frequency_encoding(df, fit=True)
        
        if '_target_binary' in df.columns:
            df = self._target_encoding_cv(df, '_target_binary', fit=True)
            df = df.drop(columns=['_target_binary'])
        
        df = self._create_all_features(df)
        df = self._encode_categorical(df, fit=True)
        
        self.feature_names = df.columns.tolist()
        
        # Scaling
        X = self.scaler.fit_transform(df.values)
        
        return X, y, ids
    
    def transform(self, df: pd.DataFrame):
        df = df.copy()
        ids = df['id'].values if 'id' in df.columns else None
        df = df.drop(columns=['id', 'label', 'attack_cat'], errors='ignore')
        
        df = self._preprocess(df)
        df = self._frequency_encoding(df, fit=False)
        df = self._target_encoding_cv(df, None, fit=False)
        df = self._create_all_features(df)
        df = self._encode_categorical(df, fit=False)
        
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        
        X = self.scaler.transform(df.values)
        return X, ids
    
    def _preprocess(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('-').astype(str)
        return df
    
    def _frequency_encoding(self, df, fit=True):
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if fit:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                self.frequency_encoding_maps[col] = freq_map
            else:
                freq_map = self.frequency_encoding_maps.get(col, {})
            
            default_freq = 1.0 / max(len(freq_map), 1)
            df[f'{col}_freq'] = df[col].map(freq_map).fillna(default_freq)
        return df
    
    def _target_encoding_cv(self, df, target_col, fit=True):
        if fit and target_col and target_col in df.columns:
            self.global_target_mean = df[target_col].mean()
            
            for col in self.categorical_cols:
                if col not in df.columns:
                    continue
                
                target_mean = df.groupby(col)[target_col].mean().to_dict()
                self.target_encoding_maps[col] = target_mean
                
                encoded = np.zeros(len(df))
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                for train_idx, val_idx in skf.split(df, df[target_col]):
                    train_means = df.iloc[train_idx].groupby(col)[target_col].mean()
                    encoded[val_idx] = df.iloc[val_idx][col].map(train_means).fillna(self.global_target_mean)
                
                df[f'{col}_target'] = encoded
        elif not fit:
            for col in self.categorical_cols:
                if col not in df.columns:
                    continue
                target_map = self.target_encoding_maps.get(col, {})
                default_val = self.global_target_mean if self.global_target_mean else 0.5
                df[f'{col}_target'] = df[col].map(target_map).fillna(default_val)
        return df
    
    def _create_all_features(self, df):
        # Basic ratios
        df['bytes_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
        df['pkts_ratio'] = df['spkts'] / (df['dpkts'] + 1)
        df['load_ratio'] = df['sload'] / (df['dload'] + 1)
        
        # Totals
        df['total_bytes'] = df['sbytes'] + df['dbytes']
        df['total_pkts'] = df['spkts'] + df['dpkts']
        df['total_loss'] = df['sloss'] + df['dloss']
        
        # Per-packet
        df['sbytes_per_pkt'] = df['sbytes'] / (df['spkts'] + 1)
        df['dbytes_per_pkt'] = df['dbytes'] / (df['dpkts'] + 1)
        
        # Rate
        df['bytes_per_sec'] = df['total_bytes'] / (df['dur'] + 1e-6)
        df['pkts_per_sec'] = df['total_pkts'] / (df['dur'] + 1e-6)
        
        # ct_* features
        ct_cols = [c for c in df.columns if c.startswith('ct_') and not c.endswith(('_freq', '_target'))]
        if ct_cols:
            df['ct_sum'] = df[ct_cols].sum(axis=1)
            df['ct_mean'] = df[ct_cols].mean(axis=1)
            df['ct_max'] = df[ct_cols].max(axis=1)
            df['ct_min'] = df[ct_cols].min(axis=1)
            df['ct_std'] = df[ct_cols].std(axis=1).fillna(0)
            df['ct_range'] = df['ct_max'] - df['ct_min']
        
        # TTL
        df['ttl_ratio'] = df['sttl'] / (df['dttl'] + 1)
        df['ttl_diff'] = df['sttl'] - df['dttl']
        df['ttl_sum'] = df['sttl'] + df['dttl']
        
        # TCP
        df['tcp_setup_ratio'] = df['synack'] / (df['tcprtt'] + 1e-6)
        df['tcp_ack_ratio'] = df['ackdat'] / (df['tcprtt'] + 1e-6)
        
        # Window
        df['window_ratio'] = df['swin'] / (df['dwin'] + 1)
        df['window_diff'] = df['swin'] - df['dwin']
        df['window_sum'] = df['swin'] + df['dwin']
        
        # Jitter
        df['jit_ratio'] = df['sjit'] / (df['djit'] + 1)
        df['jit_diff'] = df['sjit'] - df['djit']
        
        # Inter-packet
        df['intpkt_ratio'] = df['sinpkt'] / (df['dinpkt'] + 1)
        df['intpkt_diff'] = df['sinpkt'] - df['dinpkt']
        
        # Loss ratio
        df['loss_ratio'] = df['sloss'] / (df['total_loss'] + 1)
        
        # Interactions (고판별력 피처)
        if 'ct_dst_sport_ltm' in df.columns:
            df['ct_dst_sport_x_sttl'] = df['ct_dst_sport_ltm'] * df['sttl']
        if 'ct_srv_src' in df.columns and 'ct_state_ttl' in df.columns:
            df['ct_srv_x_state_ttl'] = df['ct_srv_src'] * df['ct_state_ttl']
        if 'ct_dst_ltm' in df.columns and 'ct_src_ltm' in df.columns:
            df['ct_dst_src_ltm_ratio'] = df['ct_dst_ltm'] / (df['ct_src_ltm'] + 1)
        if 'ct_srv_src' in df.columns and 'ct_srv_dst' in df.columns:
            df['ct_srv_ratio'] = df['ct_srv_src'] / (df['ct_srv_dst'] + 1)
        
        df['rate_x_dur'] = df['rate'] * df['dur']
        df['sbytes_x_sttl'] = df['sbytes'] * df['sttl']
        
        # Log transforms
        for col in ['sbytes', 'dbytes', 'sload', 'dload', 'rate', 'total_bytes', 'bytes_per_sec']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
        
        # Squared features (비선형성)
        for col in ['ct_dst_sport_ltm', 'ct_srv_src', 'sttl', 'rate']:
            if col in df.columns:
                df[f'{col}_sq'] = df[col] ** 2
        
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
# NEURAL NETWORK
# ============================================================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_neural_net(X_train, y_train, X_val, y_val, n_classes, config):
    """Neural Network 학습"""
    device = config.DEVICE
    
    # Class weights
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * n_classes
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.NN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.NN_BATCH_SIZE, shuffle=False)
    
    # Model
    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_dims=config.NN_HIDDEN_DIMS,
        n_classes=n_classes,
        dropout=config.NN_DROPOUT
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = AdamW(model.parameters(), lr=config.NN_LR, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=config.NN_LR, 
                          total_steps=config.NN_EPOCHS * len(train_loader))
    
    best_val_acc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(config.NN_EPOCHS):
        # Train
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # Validate
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                val_preds.append(F.softmax(logits, dim=1).cpu().numpy())
        
        val_preds = np.vstack(val_preds)
        val_acc = accuracy_score(y_val, val_preds.argmax(axis=1))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Get predictions
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            val_preds.append(F.softmax(logits, dim=1).cpu().numpy())
    
    val_preds = np.vstack(val_preds)
    
    return model, val_preds, best_val_acc


def predict_neural_net(model, X, config):
    """Neural Network 예측"""
    device = config.DEVICE
    model.eval()
    
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=config.NN_BATCH_SIZE, shuffle=False)
    
    preds = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds.append(F.softmax(logits, dim=1).cpu().numpy())
    
    return np.vstack(preds)


# ============================================================
# GBDT MODELS
# ============================================================
def get_lgbm_params(n_classes):
    return {
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
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
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
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'tree_method': 'hist',
        'random_state': config.RANDOM_STATE,
        'n_jobs': -1
    }


def get_catboost_params(n_classes):
    return {
        'iterations': 500,
        'learning_rate': 0.1,
        'depth': 8,
        'l2_leaf_reg': 3,
        'random_seed': config.RANDOM_STATE,
        'auto_class_weights': 'Balanced',
        'early_stopping_rounds': 50,
        'verbose': False,
        'thread_count': -1
    }


def train_lgbm_fold(X_train, y_train, X_val, y_val, params):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params, train_data, num_boost_round=2000,
        valid_sets=[val_data], valid_names=['valid'],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    return model, val_pred


def train_xgb_fold(X_train, y_train, X_val, y_val, params, n_classes):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = len(y_train) / (n_classes * class_counts + 1)
    dtrain.set_weight(class_weights[y_train])
    
    model = xgb.train(
        params, dtrain, num_boost_round=2000,
        evals=[(dval, 'valid')],
        early_stopping_rounds=100, verbose_eval=False
    )
    
    val_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
    return model, val_pred


def train_catboost_fold(X_train, y_train, X_val, y_val, params):
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    val_pred = model.predict_proba(X_val)
    return model, val_pred


def train_extratrees_fold(X_train, y_train, X_val, y_val, n_classes):
    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)
    val_pred = model.predict_proba(X_val)
    return model, val_pred


# ============================================================
# 2-STAGE CLASSIFICATION
# ============================================================
class TwoStageClassifier:
    """
    Stage 1: Hard classes (Analysis, Backdoor, DoS) vs Others
    Stage 2: 각 그룹 내에서 세부 분류
    """
    
    def __init__(self, hard_classes, n_classes, class_names):
        self.hard_classes = hard_classes  # indices
        self.n_classes = n_classes
        self.class_names = class_names
        self.stage1_model = None
        self.stage2_hard_model = None
        self.stage2_easy_model = None
    
    def fit(self, X, y):
        # Stage 1: Binary (Hard vs Easy)
        y_binary = np.isin(y, self.hard_classes).astype(int)
        
        print("  Stage 1: Hard vs Easy classification")
        self.stage1_model = lgb.LGBMClassifier(
            n_estimators=500, num_leaves=63, max_depth=8,
            learning_rate=0.05, class_weight='balanced',
            random_state=config.RANDOM_STATE, verbose=-1
        )
        self.stage1_model.fit(X, y_binary)
        
        # Stage 2 - Hard classes
        hard_mask = np.isin(y, self.hard_classes)
        if hard_mask.sum() > 0:
            print(f"  Stage 2 (Hard): {hard_mask.sum()} samples, classes {[self.class_names[i] for i in self.hard_classes]}")
            
            # Remap labels to 0, 1, 2, ...
            hard_y = y[hard_mask]
            self.hard_label_map = {old: new for new, old in enumerate(sorted(set(hard_y)))}
            self.hard_label_map_inv = {v: k for k, v in self.hard_label_map.items()}
            hard_y_mapped = np.array([self.hard_label_map[yi] for yi in hard_y])
            
            self.stage2_hard_model = lgb.LGBMClassifier(
                n_estimators=500, num_leaves=127, max_depth=10,
                learning_rate=0.05, class_weight='balanced',
                random_state=config.RANDOM_STATE, verbose=-1
            )
            self.stage2_hard_model.fit(X[hard_mask], hard_y_mapped)
        
        # Stage 2 - Easy classes
        easy_mask = ~hard_mask
        if easy_mask.sum() > 0:
            print(f"  Stage 2 (Easy): {easy_mask.sum()} samples")
            
            easy_y = y[easy_mask]
            self.easy_label_map = {old: new for new, old in enumerate(sorted(set(easy_y)))}
            self.easy_label_map_inv = {v: k for k, v in self.easy_label_map.items()}
            easy_y_mapped = np.array([self.easy_label_map[yi] for yi in easy_y])
            
            self.stage2_easy_model = lgb.LGBMClassifier(
                n_estimators=500, num_leaves=63, max_depth=8,
                learning_rate=0.05, class_weight='balanced',
                random_state=config.RANDOM_STATE, verbose=-1
            )
            self.stage2_easy_model.fit(X[easy_mask], easy_y_mapped)
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, self.n_classes))
        
        # Stage 1 probabilities
        stage1_probs = self.stage1_model.predict_proba(X)  # (n, 2): [easy, hard]
        
        # Stage 2 Hard
        if self.stage2_hard_model is not None:
            hard_probs = self.stage2_hard_model.predict_proba(X)
            for new_idx in range(hard_probs.shape[1]):
                old_idx = self.hard_label_map_inv[new_idx]
                probs[:, old_idx] = stage1_probs[:, 1] * hard_probs[:, new_idx]
        
        # Stage 2 Easy
        if self.stage2_easy_model is not None:
            easy_probs = self.stage2_easy_model.predict_proba(X)
            for new_idx in range(easy_probs.shape[1]):
                old_idx = self.easy_label_map_inv[new_idx]
                probs[:, old_idx] = stage1_probs[:, 0] * easy_probs[:, new_idx]
        
        # Normalize
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
        
        return probs


# ============================================================
# CLASS-WISE ENSEMBLE
# ============================================================
def classwise_ensemble(oof_preds_dict, y_true, n_classes, class_names):
    """각 클래스별로 가장 잘 맞추는 모델 선택"""
    
    print("\n[Class-wise Best Model Selection]")
    
    best_model_per_class = {}
    class_weights_per_model = {name: np.zeros(n_classes) for name in oof_preds_dict.keys()}
    
    for cls in range(n_classes):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        
        best_acc = 0
        best_model = None
        
        for name, pred in oof_preds_dict.items():
            # 이 클래스의 recall
            cls_pred = pred[mask].argmax(axis=1)
            acc = (cls_pred == cls).mean()
            
            if acc > best_acc:
                best_acc = acc
                best_model = name
        
        best_model_per_class[cls] = (best_model, best_acc)
        print(f"  {class_names[cls]}: {best_model} ({best_acc:.4f})")
    
    # 가중치 계산: 각 모델이 각 클래스에서 얼마나 좋은지
    for name, pred in oof_preds_dict.items():
        for cls in range(n_classes):
            mask = y_true == cls
            if mask.sum() == 0:
                continue
            cls_pred = pred[mask].argmax(axis=1)
            class_weights_per_model[name][cls] = (cls_pred == cls).mean()
    
    return best_model_per_class, class_weights_per_model


def apply_classwise_ensemble(oof_preds_dict, class_weights_per_model, n_classes):
    """클래스별 가중치로 앙상블"""
    
    n_samples = list(oof_preds_dict.values())[0].shape[0]
    ensemble_pred = np.zeros((n_samples, n_classes))
    
    # 각 클래스별로 가중 평균
    for cls in range(n_classes):
        total_weight = sum(class_weights_per_model[name][cls] for name in oof_preds_dict.keys())
        
        for name, pred in oof_preds_dict.items():
            weight = class_weights_per_model[name][cls] / (total_weight + 1e-8)
            ensemble_pred[:, cls] += weight * pred[:, cls]
    
    return ensemble_pred


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================
def optimize_thresholds(y_true, y_probs, n_classes, class_names):
    """클래스별 threshold 최적화"""
    
    print("\n[Threshold Optimization]")
    
    thresholds = {}
    
    for cls in range(n_classes):
        best_thresh = 0.5
        best_f1 = 0
        
        for thresh in np.arange(0.1, 0.9, 0.02):
            # 이 threshold 이상이면 이 클래스로 예측
            pred = np.zeros(len(y_true))
            pred[y_probs[:, cls] >= thresh] = cls
            pred[y_probs[:, cls] < thresh] = y_probs[:, :].argmax(axis=1)[y_probs[:, cls] < thresh]
            
            # 이 클래스에 대한 F1
            true_binary = (y_true == cls).astype(int)
            pred_binary = (pred == cls).astype(int)
            
            if pred_binary.sum() == 0:
                continue
            
            f1 = f1_score(true_binary, pred_binary)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        thresholds[cls] = best_thresh
        print(f"  {class_names[cls]}: threshold={best_thresh:.2f}, F1={best_f1:.4f}")
    
    return thresholds


def apply_thresholds(y_probs, thresholds, n_classes):
    """최적화된 threshold 적용"""
    
    # 기본 예측
    pred = y_probs.argmax(axis=1)
    
    # threshold 조정 (confidence가 낮으면 다른 클래스 고려)
    for i in range(len(pred)):
        max_prob = y_probs[i].max()
        predicted_class = pred[i]
        
        # 예측 confidence가 낮으면 threshold 기반으로 재조정
        if max_prob < thresholds.get(predicted_class, 0.5):
            # 다음으로 높은 확률의 클래스 선택
            sorted_idx = np.argsort(y_probs[i])[::-1]
            for idx in sorted_idx:
                if y_probs[i, idx] >= thresholds.get(idx, 0.3):
                    pred[i] = idx
                    break
    
    return pred


# ============================================================
# CONFUSION MATRIX ANALYSIS
# ============================================================
def analyze_confusion(y_true, y_pred, class_names):
    """Confusion matrix 분석"""
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*60)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 가장 많이 혼동되는 쌍 찾기
    misclassified = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                misclassified.append((class_names[i], class_names[j], cm[i, j], cm[i, j] / cm[i].sum()))
    
    # 정렬
    misclassified.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 10 Misclassification Pairs:")
    print("-" * 60)
    for true_cls, pred_cls, count, ratio in misclassified[:10]:
        print(f"  {true_cls:15} → {pred_cls:15}: {count:5} ({ratio*100:.1f}%)")
    
    # 클래스별 문제점
    print("\nPer-Class Analysis:")
    print("-" * 60)
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i, i]
        acc = correct / total if total > 0 else 0
        
        if acc < 0.7:  # 문제 클래스
            print(f"\n  {name} (Accuracy: {acc:.2%}, Total: {total})")
            
            # 가장 많이 잘못 예측한 클래스
            wrong_preds = [(class_names[j], cm[i, j]) for j in range(len(class_names)) if j != i and cm[i, j] > 0]
            wrong_preds.sort(key=lambda x: x[1], reverse=True)
            
            for pred_cls, count in wrong_preds[:3]:
                print(f"    → {pred_cls}: {count} ({count/total*100:.1f}%)")
    
    return cm


# ============================================================
# MAIN CROSS VALIDATION
# ============================================================
def run_cv(X, y, X_test, n_classes, class_names):
    """전체 CV 실행"""
    
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    
    # OOF / Test predictions
    oof_preds = {}
    test_preds = {}
    scores = {}
    
    model_names = []
    if config.USE_LGBM:
        model_names.append('lgbm')
    if config.USE_XGB:
        model_names.append('xgb')
    if config.USE_CATBOOST:
        model_names.append('catboost')
    if config.USE_EXTRATREES:
        model_names.append('extratrees')
    if config.USE_NEURAL_NET:
        model_names.append('nn')
    if config.USE_TWO_STAGE:
        model_names.append('two_stage')
    
    for name in model_names:
        oof_preds[name] = np.zeros((len(X), n_classes))
        test_preds[name] = np.zeros((len(X_test), n_classes))
        scores[name] = []
    
    print(f"\n{'='*60}")
    print(f"Starting {config.N_FOLDS}-Fold CV with {len(model_names)} models")
    print(f"Models: {model_names}")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/{config.N_FOLDS}")
        print(f"{'='*40}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # LightGBM
        if config.USE_LGBM:
            print(f"\n[LightGBM]", end=" ")
            start = time.time()
            model, val_pred = train_lgbm_fold(X_train, y_train, X_val, y_val, get_lgbm_params(n_classes))
            oof_preds['lgbm'][val_idx] = val_pred
            test_preds['lgbm'] += model.predict(X_test) / config.N_FOLDS
            acc = accuracy_score(y_val, val_pred.argmax(axis=1))
            scores['lgbm'].append(acc)
            print(f"Acc: {acc:.4f} | Time: {time.time()-start:.1f}s")
        
        # XGBoost
        if config.USE_XGB:
            print(f"[XGBoost]", end=" ")
            start = time.time()
            model, val_pred = train_xgb_fold(X_train, y_train, X_val, y_val, get_xgb_params(n_classes), n_classes)
            oof_preds['xgb'][val_idx] = val_pred
            test_preds['xgb'] += model.predict(xgb.DMatrix(X_test)) / config.N_FOLDS
            acc = accuracy_score(y_val, val_pred.argmax(axis=1))
            scores['xgb'].append(acc)
            print(f"Acc: {acc:.4f} | Time: {time.time()-start:.1f}s")
        
        # CatBoost
        if config.USE_CATBOOST:
            print(f"[CatBoost]", end=" ")
            start = time.time()
            model, val_pred = train_catboost_fold(X_train, y_train, X_val, y_val, get_catboost_params(n_classes))
            oof_preds['catboost'][val_idx] = val_pred
            test_preds['catboost'] += model.predict_proba(X_test) / config.N_FOLDS
            acc = accuracy_score(y_val, val_pred.argmax(axis=1))
            scores['catboost'].append(acc)
            print(f"Acc: {acc:.4f} | Time: {time.time()-start:.1f}s")
        
        # ExtraTrees
        if config.USE_EXTRATREES:
            print(f"[ExtraTrees]", end=" ")
            start = time.time()
            model, val_pred = train_extratrees_fold(X_train, y_train, X_val, y_val, n_classes)
            oof_preds['extratrees'][val_idx] = val_pred
            test_preds['extratrees'] += model.predict_proba(X_test) / config.N_FOLDS
            acc = accuracy_score(y_val, val_pred.argmax(axis=1))
            scores['extratrees'].append(acc)
            print(f"Acc: {acc:.4f} | Time: {time.time()-start:.1f}s")
        
        # Neural Network
        if config.USE_NEURAL_NET:
            print(f"[Neural Net]", end=" ")
            start = time.time()
            model, val_pred, best_acc = train_neural_net(X_train, y_train, X_val, y_val, n_classes, config)
            oof_preds['nn'][val_idx] = val_pred
            test_preds['nn'] += predict_neural_net(model, X_test, config) / config.N_FOLDS
            acc = accuracy_score(y_val, val_pred.argmax(axis=1))
            scores['nn'].append(acc)
            print(f"Acc: {acc:.4f} | Time: {time.time()-start:.1f}s")
        
        # 2-Stage Classification
        if config.USE_TWO_STAGE:
            print(f"[2-Stage]", end=" ")
            start = time.time()
            
            # Hard classes 식별 (Analysis=0, Backdoor=1, DoS=3)
            hard_classes = [0, 1, 3]  # Analysis, Backdoor, DoS
            
            two_stage = TwoStageClassifier(hard_classes, n_classes, class_names)
            two_stage.fit(X_train, y_train)
            
            val_pred = two_stage.predict_proba(X_val)
            oof_preds['two_stage'][val_idx] = val_pred
            test_preds['two_stage'] += two_stage.predict_proba(X_test) / config.N_FOLDS
            
            acc = accuracy_score(y_val, val_pred.argmax(axis=1))
            scores['two_stage'].append(acc)
            print(f"Acc: {acc:.4f} | Time: {time.time()-start:.1f}s")
    
    return oof_preds, test_preds, scores


# ============================================================
# FINAL ENSEMBLE
# ============================================================
def create_ultimate_ensemble(oof_preds, test_preds, y_true, n_classes, class_names):
    """최종 앙상블 생성"""
    
    print(f"\n{'='*60}")
    print("ULTIMATE ENSEMBLE")
    print(f"{'='*60}")
    
    # 1. 개별 모델 성능
    print("\n[Individual Model Scores]")
    model_scores = {}
    for name, pred in oof_preds.items():
        acc = accuracy_score(y_true, pred.argmax(axis=1))
        model_scores[name] = acc
        print(f"  {name}: {acc:.4f}")
    
    # 2. Simple Average
    simple_oof = sum(oof_preds.values()) / len(oof_preds)
    simple_test = sum(test_preds.values()) / len(test_preds)
    simple_acc = accuracy_score(y_true, simple_oof.argmax(axis=1))
    print(f"\n[Simple Average] {simple_acc:.4f}")
    
    # 3. Weighted Average (성능 기반)
    total_score = sum(model_scores.values())
    weighted_oof = sum((model_scores[name] / total_score) * oof_preds[name] for name in oof_preds.keys())
    weighted_test = sum((model_scores[name] / total_score) * test_preds[name] for name in test_preds.keys())
    weighted_acc = accuracy_score(y_true, weighted_oof.argmax(axis=1))
    print(f"[Weighted Average] {weighted_acc:.4f}")
    
    # 4. Class-wise Ensemble
    if config.USE_CLASSWISE_ENSEMBLE:
        best_model_per_class, class_weights = classwise_ensemble(oof_preds, y_true, n_classes, class_names)
        classwise_oof = apply_classwise_ensemble(oof_preds, class_weights, n_classes)
        classwise_test = apply_classwise_ensemble(test_preds, class_weights, n_classes)
        classwise_acc = accuracy_score(y_true, classwise_oof.argmax(axis=1))
        print(f"\n[Class-wise Ensemble] {classwise_acc:.4f}")
    else:
        classwise_oof, classwise_test, classwise_acc = None, None, 0
    
    # 5. Stacking
    print("\n[Stacking Meta Learner]")
    meta_features_oof = np.hstack([pred for pred in oof_preds.values()])
    meta_features_test = np.hstack([pred for pred in test_preds.values()])
    
    # 추가 meta features
    for pred in oof_preds.values():
        meta_features_oof = np.hstack([meta_features_oof, pred.max(axis=1, keepdims=True), pred.std(axis=1, keepdims=True)])
    for pred in test_preds.values():
        meta_features_test = np.hstack([meta_features_test, pred.max(axis=1, keepdims=True), pred.std(axis=1, keepdims=True)])
    
    meta_model = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial', random_state=42)
    meta_model.fit(meta_features_oof, y_true)
    
    stacking_oof = meta_model.predict_proba(meta_features_oof)
    stacking_test = meta_model.predict_proba(meta_features_test)
    stacking_acc = accuracy_score(y_true, stacking_oof.argmax(axis=1))
    print(f"  Stacking Accuracy: {stacking_acc:.4f}")
    
    # 6. 최고 성능 선택
    results = {
        'simple': (simple_oof, simple_test, simple_acc),
        'weighted': (weighted_oof, weighted_test, weighted_acc),
        'stacking': (stacking_oof, stacking_test, stacking_acc),
    }
    
    if classwise_oof is not None:
        results['classwise'] = (classwise_oof, classwise_test, classwise_acc)
    
    best_method = max(results.items(), key=lambda x: x[1][2])
    print(f"\n[Best Base Method] {best_method[0]}: {best_method[1][2]:.4f}")
    
    # 7. Threshold Optimization
    final_oof, final_test, final_acc = best_method[1]
    
    if config.USE_THRESHOLD_OPT:
        thresholds = optimize_thresholds(y_true, final_oof, n_classes, class_names)
        
        # Threshold 적용한 예측
        thresh_pred = apply_thresholds(final_oof, thresholds, n_classes)
        thresh_acc = accuracy_score(y_true, thresh_pred)
        print(f"\n[After Threshold Optimization] {thresh_acc:.4f}")
        
        if thresh_acc > final_acc:
            final_acc = thresh_acc
    else:
        thresholds = None
    
    return {
        'final_oof': final_oof,
        'final_test': final_test,
        'final_acc': final_acc,
        'best_method': best_method[0],
        'all_results': {k: v[2] for k, v in results.items()},
        'model_scores': model_scores,
        'thresholds': thresholds
    }


# ============================================================
# MAIN
# ============================================================
def main():
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("NIDS ULTIMATE PIPELINE - Target 90%")
    print("="*60)
    
    # 1. Load Data
    print("\n[1/6] Loading Data...")
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    print(f"  Train: {df_train.shape}")
    
    df_test = None
    if Path(config.TEST_DATA_PATH).exists():
        df_test = pd.read_csv(config.TEST_DATA_PATH)
        print(f"  Test: {df_test.shape}")
    
    # 2. Feature Engineering
    print("\n[2/6] Feature Engineering...")
    fe = UltimateFeatureEngineer()
    X, y, train_ids = fe.fit_transform(df_train, config.LABEL_COL)
    
    class_names = list(fe.target_encoder.classes_)
    n_classes = len(class_names)
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {n_classes}")
    
    for i, name in enumerate(class_names):
        count = (y == i).sum()
        print(f"    {i}: {name} ({count:,})")
    
    if df_test is not None:
        X_test, test_ids = fe.transform(df_test)
    else:
        X_test = np.zeros((0, X.shape[1]))
        test_ids = None
    
    fe.save(output_dir / 'feature_engineer.pkl')
    
    with open(output_dir / 'class_mapping.json', 'w') as f:
        json.dump({int(i): str(c) for i, c in enumerate(class_names)}, f, indent=2)
    
    # 3. Cross Validation
    print("\n[3/6] Cross Validation...")
    oof_preds, test_preds, scores = run_cv(X, y, X_test, n_classes, class_names)
    
    # 4. CV Summary
    print(f"\n{'='*60}")
    print("CV Summary")
    print(f"{'='*60}")
    
    for name, fold_scores in scores.items():
        print(f"  {name}: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    
    # 5. Confusion Matrix Analysis
    print("\n[4/6] Confusion Matrix Analysis...")
    
    # 가장 좋은 단일 모델로 분석
    best_single = max(scores.items(), key=lambda x: np.mean(x[1]))
    best_pred = oof_preds[best_single[0]].argmax(axis=1)
    analyze_confusion(y, best_pred, class_names)
    
    # 6. Ultimate Ensemble
    print("\n[5/6] Ultimate Ensemble...")
    result = create_ultimate_ensemble(oof_preds, test_preds, y, n_classes, class_names)
    
    # 7. Final Report
    print("\n[6/6] Final Report...")
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    print(f"\nBest Method: {result['best_method']}")
    print(f"Final Accuracy: {result['final_acc']:.4f}")
    
    print("\nPer-Class Accuracy:")
    final_pred = result['final_oof'].argmax(axis=1)
    for i, name in enumerate(class_names):
        mask = y == i
        if mask.sum() > 0:
            acc = (final_pred[mask] == y[mask]).mean()
            status = "✓" if acc >= 0.7 else "✗"
            print(f"  {status} {name}: {acc:.4f} ({mask.sum()} samples)")
    
    print("\nClassification Report:")
    print(classification_report(y, final_pred, target_names=class_names, digits=4))
    
    # Save results
    if len(X_test) > 0:
        pred_labels = result['final_test'].argmax(axis=1)
        pred_names = [class_names[p] for p in pred_labels]
        
        result_df = pd.DataFrame({
            'id': test_ids if test_ids is not None else np.arange(len(pred_labels)),
            'prediction': pred_names
        })
        result_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    # Save OOF
    np.save(output_dir / 'oof_final.npy', result['final_oof'])
    np.save(output_dir / 'test_final.npy', result['final_test'])
    
    # Save summary
    summary = {
        'final_accuracy': result['final_acc'],
        'best_method': result['best_method'],
        'all_results': result['all_results'],
        'model_scores': result['model_scores'],
        'per_class_accuracy': {
            class_names[i]: float((final_pred[y == i] == y[y == i]).mean())
            for i in range(n_classes) if (y == i).sum() > 0
        }
    }
    
    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"\nOutput: {output_dir}")
    
    return result


if __name__ == "__main__":
    main()