"""
NIDS Ultimate Boosted Pipeline
===============================

통합 성능 향상 기법:
1. 2-Stage Classification (Hard class 분리)
2. Feature Engineering 추가 (피처 차이, 3차 상호작용, 클러스터링)
3. Pseudo Labeling
4. SMOTE (소수 클래스 오버샘플링)
5. ExtraTrees 추가

사용법:
    python nids_boosted.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import pickle
import json
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# SMOTE
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("imblearn not installed. Run: pip install imbalanced-learn")

# ============================================================
# CONFIG
# ============================================================
class Config:
    TRAIN_DATA_PATH = "./train.csv"
    TEST_DATA_PATH = "./test.csv"
    OUTPUT_DIR = "./boosted_output"
    
    LABEL_COL = "attack_cat"
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # 기능 ON/OFF
    USE_TWO_STAGE = True
    USE_EXTRA_FEATURES = True
    USE_PSEUDO_LABELING = True
    USE_SMOTE = True
    USE_EXTRATREES = True
    
    # Pseudo Labeling 설정
    PSEUDO_THRESHOLD = 0.95
    PSEUDO_ITERATIONS = 2
    
    # Hard classes (Exploits로 많이 잘못 분류되는 클래스)
    HARD_CLASSES = ['DoS', 'Analysis', 'Backdoor']

config = Config()

# ============================================================
# OPTUNA BEST PARAMETERS
# ============================================================
BEST_PARAMS = {
    "lgbm": {
        "num_leaves": 115,
        "max_depth": 15,
        "learning_rate": 0.07259248719561363,
        "n_estimators": 1018,
        "min_child_samples": 24,
        "subsample": 0.5779972601681014,
        "colsample_bytree": 0.5290418060840998,
        "reg_alpha": 0.6245760287469893,
        "reg_lambda": 0.002570603566117598,
        "min_split_gain": 0.7080725777960455,
    },
    "xgb": {
        "max_depth": 10,
        "learning_rate": 0.01024816971544603,
        "n_estimators": 685,
        "min_child_weight": 1,
        "subsample": 0.7259507714964042,
        "colsample_bytree": 0.6867771857270699,
        "colsample_bylevel": 0.8315080516469936,
        "reg_alpha": 7.192693627056316e-06,
        "reg_lambda": 0.9478184390242359,
        "gamma": 0.08065222147507445
    },
    "catboost": {
        "iterations": 963,
        "depth": 10,
        "learning_rate": 0.021389493672398414,
        "l2_leaf_reg": 1.2288041310295368e-08,
        "bagging_temperature": 0.5738463260563077,
        "random_strength": 2.922384655019864,
        "border_count": 127,
    }
}

# ============================================================
# ENHANCED FEATURE ENGINEER
# ============================================================
class EnhancedFeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoder = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.categorical_cols = ['proto', 'service', 'state']
        self.target_encoding_maps = {}
        self.frequency_encoding_maps = {}
        self.global_target_mean = None
        self.kmeans = None
        
    def fit_transform(self, df, label_col=None):
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
        
        # 기본 피처
        df = self._create_basic_features(df)
        
        # 추가 피처 (새로 추가)
        if config.USE_EXTRA_FEATURES:
            df = self._create_extra_features(df, fit=True)
        
        df = self._encode_categorical(df, fit=True)
        
        self.feature_names = df.columns.tolist()
        X = self.scaler.fit_transform(df.values)
        
        return X, y, ids
    
    def transform(self, df):
        df = df.copy()
        ids = df['id'].values if 'id' in df.columns else None
        df = df.drop(columns=['id', 'label', 'attack_cat'], errors='ignore')
        
        df = self._preprocess(df)
        df = self._frequency_encoding(df, fit=False)
        df = self._target_encoding_cv(df, None, fit=False)
        df = self._create_basic_features(df)
        
        if config.USE_EXTRA_FEATURES:
            df = self._create_extra_features(df, fit=False)
        
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
                self.frequency_encoding_maps[col] = df[col].value_counts(normalize=True).to_dict()
            freq_map = self.frequency_encoding_maps.get(col, {})
            df[f'{col}_freq'] = df[col].map(freq_map).fillna(1.0/max(len(freq_map),1))
        return df
    
    def _target_encoding_cv(self, df, target_col, fit=True):
        if fit and target_col and target_col in df.columns:
            self.global_target_mean = df[target_col].mean()
            for col in self.categorical_cols:
                if col not in df.columns:
                    continue
                self.target_encoding_maps[col] = df.groupby(col)[target_col].mean().to_dict()
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
                df[f'{col}_target'] = df[col].map(target_map).fillna(self.global_target_mean or 0.5)
        return df
    
    def _create_basic_features(self, df):
        """기본 피처 (이전과 동일)"""
        # Ratios
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
        ct_cols = [c for c in df.columns if c.startswith('ct_') and not c.endswith(('_freq','_target'))]
        if ct_cols:
            df['ct_sum'] = df[ct_cols].sum(axis=1)
            df['ct_mean'] = df[ct_cols].mean(axis=1)
            df['ct_max'] = df[ct_cols].max(axis=1)
            df['ct_min'] = df[ct_cols].min(axis=1)
            df['ct_std'] = df[ct_cols].std(axis=1).fillna(0)
        
        # TTL
        df['ttl_ratio'] = df['sttl'] / (df['dttl'] + 1)
        df['ttl_diff'] = df['sttl'] - df['dttl']
        df['ttl_sum'] = df['sttl'] + df['dttl']
        
        # Window
        df['window_ratio'] = df['swin'] / (df['dwin'] + 1)
        df['window_diff'] = df['swin'] - df['dwin']
        
        # Interactions
        if 'ct_dst_sport_ltm' in df.columns:
            df['ct_dst_sport_x_sttl'] = df['ct_dst_sport_ltm'] * df['sttl']
        if 'ct_srv_src' in df.columns and 'ct_srv_dst' in df.columns:
            df['ct_srv_ratio'] = df['ct_srv_src'] / (df['ct_srv_dst'] + 1)
        df['sbytes_x_sttl'] = df['sbytes'] * df['sttl']
        df['rate_x_dur'] = df['rate'] * df['dur']
        
        # Log transforms
        for col in ['sbytes', 'dbytes', 'sload', 'dload', 'rate', 'total_bytes']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
        
        return df
    
    def _create_extra_features(self, df, fit=True):
        """추가 피처 (새로 추가)"""
        
        # 1. 피처 간 차이 (Difference features)
        df['bytes_diff'] = df['sbytes'] - df['dbytes']
        df['pkts_diff'] = df['spkts'] - df['dpkts']
        df['load_diff'] = df['sload'] - df['dload']
        df['loss_diff'] = df['sloss'] - df['dloss']
        df['mean_diff'] = df['smean'] - df['dmean']
        df['jit_diff'] = df['sjit'] - df['djit']
        df['win_diff'] = df['swin'] - df['dwin']
        df['intpkt_diff'] = df['sinpkt'] - df['dinpkt']
        
        # 2. 비대칭성 (Asymmetry)
        df['bytes_asym'] = (df['sbytes'] - df['dbytes']) / (df['total_bytes'] + 1)
        df['pkts_asym'] = (df['spkts'] - df['dpkts']) / (df['total_pkts'] + 1)
        df['ttl_asym'] = (df['sttl'] - df['dttl']) / (df['ttl_sum'] + 1)
        
        # 3. 3차 상호작용 (Triple interactions)
        df['bytes_rate_dur'] = df['total_bytes'] * df['rate'] * df['dur']
        df['ttl_pkts_bytes'] = df['ttl_sum'] * df['total_pkts'] * np.log1p(df['total_bytes'])
        if 'ct_srv_src' in df.columns and 'ct_dst_sport_ltm' in df.columns:
            df['ct_triple'] = df['ct_srv_src'] * df['ct_dst_sport_ltm'] * df['sttl']
        
        # 4. 상대적 비율 (Relative ratios)
        df['sbytes_pct'] = df['sbytes'] / (df['total_bytes'] + 1)
        df['spkts_pct'] = df['spkts'] / (df['total_pkts'] + 1)
        df['sload_pct'] = df['sload'] / (df['sload'] + df['dload'] + 1)
        
        # 5. 효율성 (Efficiency metrics)
        df['bytes_efficiency'] = df['total_bytes'] / (df['total_pkts'] + 1)
        df['load_efficiency'] = (df['sload'] + df['dload']) / (df['dur'] + 1e-6)
        
        # 6. TCP 관련 추가
        df['tcp_setup_time'] = df['synack'] + df['ackdat']
        df['tcp_rtt_ratio'] = df['tcprtt'] / (df['dur'] + 1e-6)
        
        # 7. ct_* 추가 통계
        ct_cols = [c for c in df.columns if c.startswith('ct_') and not c.endswith(('_freq','_target','_triple'))]
        if ct_cols:
            df['ct_range'] = df['ct_max'] - df['ct_min']
            df['ct_cv'] = df['ct_std'] / (df['ct_mean'] + 1e-6)  # coefficient of variation
        
        # 8. Squared features (비선형성)
        for col in ['rate', 'dur', 'sttl', 'ct_sum']:
            if col in df.columns:
                df[f'{col}_sq'] = df[col] ** 2
        
        # 9. 클러스터링 기반 피처
        cluster_features = ['total_bytes', 'total_pkts', 'dur', 'rate', 'sttl', 'dttl']
        cluster_features = [c for c in cluster_features if c in df.columns]
        
        if len(cluster_features) >= 3:
            cluster_data = df[cluster_features].fillna(0).values
            cluster_data = np.clip(cluster_data, -1e10, 1e10)
            
            if fit:
                self.kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
                self.kmeans.fit(cluster_data)
            
            if self.kmeans is not None:
                df['cluster_id'] = self.kmeans.predict(cluster_data)
                distances = self.kmeans.transform(cluster_data)
                df['cluster_dist'] = distances.min(axis=1)
        
        return df
    
    def _encode_categorical(self, df, fit=True):
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if fit:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(list(df[col].unique()) + ['<UNK>'])
            known = set(self.label_encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known else '<UNK>')
            df[col] = self.label_encoders[col].transform(df[col])
        return df
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)


# ============================================================
# 2-STAGE CLASSIFIER
# ============================================================
class TwoStageClassifier:
    """
    Stage 1: Hard class (DoS, Analysis, Backdoor) vs Others
    Stage 2: 각 그룹 내 세부 분류
    """
    
    def __init__(self, hard_classes, n_classes, class_names):
        self.hard_classes = hard_classes
        self.n_classes = n_classes
        self.class_names = list(class_names)
        self.hard_indices = [self.class_names.index(c) for c in hard_classes if c in self.class_names]
        
        self.stage1_model = None
        self.stage2_hard_model = None
        self.stage2_easy_model = None
        self.hard_label_map = {}
        self.hard_label_map_inv = {}
        self.easy_label_map = {}
        self.easy_label_map_inv = {}
    
    def fit(self, X, y, X_val=None, y_val=None):
        # Stage 1: Binary (Hard vs Easy)
        y_binary = np.isin(y, self.hard_indices).astype(int)
        
        self.stage1_model = lgb.LGBMClassifier(
            n_estimators=500, num_leaves=63, max_depth=8,
            learning_rate=0.05, class_weight='balanced',
            random_state=config.RANDOM_STATE, verbose=-1, n_jobs=-1
        )
        self.stage1_model.fit(X, y_binary)
        
        # Stage 2 - Hard classes
        hard_mask = np.isin(y, self.hard_indices)
        if hard_mask.sum() > 0:
            hard_y = y[hard_mask]
            self.hard_label_map = {old: new for new, old in enumerate(sorted(set(hard_y)))}
            self.hard_label_map_inv = {v: k for k, v in self.hard_label_map.items()}
            hard_y_mapped = np.array([self.hard_label_map[yi] for yi in hard_y])
            
            # Hard class에 대해 더 강한 class weight
            self.stage2_hard_model = lgb.LGBMClassifier(
                n_estimators=500, num_leaves=127, max_depth=10,
                learning_rate=0.05, class_weight='balanced',
                random_state=config.RANDOM_STATE, verbose=-1, n_jobs=-1
            )
            self.stage2_hard_model.fit(X[hard_mask], hard_y_mapped)
        
        # Stage 2 - Easy classes
        easy_mask = ~hard_mask
        if easy_mask.sum() > 0:
            easy_y = y[easy_mask]
            self.easy_label_map = {old: new for new, old in enumerate(sorted(set(easy_y)))}
            self.easy_label_map_inv = {v: k for k, v in self.easy_label_map.items()}
            easy_y_mapped = np.array([self.easy_label_map[yi] for yi in easy_y])
            
            self.stage2_easy_model = lgb.LGBMClassifier(
                n_estimators=500, num_leaves=63, max_depth=8,
                learning_rate=0.05, class_weight='balanced',
                random_state=config.RANDOM_STATE, verbose=-1, n_jobs=-1
            )
            self.stage2_easy_model.fit(X[easy_mask], easy_y_mapped)
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, self.n_classes))
        
        # Stage 1
        stage1_probs = self.stage1_model.predict_proba(X)  # [easy, hard]
        
        # Stage 2 Hard
        if self.stage2_hard_model is not None:
            hard_probs = self.stage2_hard_model.predict_proba(X)
            for new_idx in range(hard_probs.shape[1]):
                old_idx = self.hard_label_map_inv.get(new_idx)
                if old_idx is not None:
                    probs[:, old_idx] = stage1_probs[:, 1] * hard_probs[:, new_idx]
        
        # Stage 2 Easy
        if self.stage2_easy_model is not None:
            easy_probs = self.stage2_easy_model.predict_proba(X)
            for new_idx in range(easy_probs.shape[1]):
                old_idx = self.easy_label_map_inv.get(new_idx)
                if old_idx is not None:
                    probs[:, old_idx] = stage1_probs[:, 0] * easy_probs[:, new_idx]
        
        # Normalize
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = np.where(row_sums > 0, probs / row_sums, 1.0 / self.n_classes)
        
        return probs


# ============================================================
# SMOTE WRAPPER
# ============================================================
def apply_smote(X, y, random_state=42):
    """SMOTE로 소수 클래스 오버샘플링"""
    
    if not IMBLEARN_AVAILABLE:
        print("  SMOTE skipped (imblearn not installed)")
        return X, y
    
    print("  Applying SMOTE...")
    original_counts = np.bincount(y)
    
    # 소수 클래스만 오버샘플링 (최소 샘플 수의 2배까지)
    min_samples = original_counts.min()
    target_samples = min(min_samples * 3, original_counts.max() // 5)
    
    sampling_strategy = {}
    for cls, count in enumerate(original_counts):
        if count < target_samples:
            sampling_strategy[cls] = target_samples
    
    if not sampling_strategy:
        print("  No classes need oversampling")
        return X, y
    
    try:
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=3)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        new_counts = np.bincount(y_resampled)
        print(f"  Samples: {len(y)} → {len(y_resampled)}")
        
        return X_resampled, y_resampled
    except Exception as e:
        print(f"  SMOTE failed: {e}")
        return X, y


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_lgbm_fold(X_train, y_train, X_val, y_val, n_classes, use_smote=False):
    """LightGBM single fold"""
    
    if use_smote and IMBLEARN_AVAILABLE:
        X_train, y_train = apply_smote(X_train, y_train)
    
    params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': config.RANDOM_STATE,
        'n_jobs': -1,
        'class_weight': 'balanced',
        **BEST_PARAMS['lgbm']
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    val_pred = model.predict_proba(X_val)
    return model, val_pred


def train_xgb_fold(X_train, y_train, X_val, y_val, n_classes, use_smote=False):
    """XGBoost single fold"""
    
    if use_smote and IMBLEARN_AVAILABLE:
        X_train, y_train = apply_smote(X_train, y_train)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'random_state': config.RANDOM_STATE,
        'n_jobs': -1,
        'early_stopping_rounds': 100,
        **BEST_PARAMS['xgb']
    }
    
    # Class weight
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = len(y_train) / (n_classes * class_counts + 1)
    sample_weights = class_weights[y_train]
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    val_pred = model.predict_proba(X_val)
    return model, val_pred


def train_catboost_fold(X_train, y_train, X_val, y_val, n_classes, use_smote=False):
    """CatBoost single fold"""
    
    if use_smote and IMBLEARN_AVAILABLE:
        X_train, y_train = apply_smote(X_train, y_train)
    
    params = {
        'loss_function': 'MultiClass',
        'classes_count': n_classes,
        'random_seed': config.RANDOM_STATE,
        'verbose': False,
        'thread_count': -1,
        'early_stopping_rounds': 100,
        'auto_class_weights': 'Balanced',
        **BEST_PARAMS['catboost']
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    
    val_pred = model.predict_proba(X_val)
    return model, val_pred


def train_extratrees_fold(X_train, y_train, X_val, y_val, n_classes, use_smote=False):
    """ExtraTrees single fold"""
    
    if use_smote and IMBLEARN_AVAILABLE:
        X_train, y_train = apply_smote(X_train, y_train)
    
    model = ExtraTreesClassifier(
        n_estimators=500,
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


def train_two_stage_fold(X_train, y_train, X_val, y_val, n_classes, class_names, use_smote=False):
    """2-Stage single fold"""
    
    if use_smote and IMBLEARN_AVAILABLE:
        X_train, y_train = apply_smote(X_train, y_train)
    
    model = TwoStageClassifier(config.HARD_CLASSES, n_classes, class_names)
    model.fit(X_train, y_train)
    
    val_pred = model.predict_proba(X_val)
    return model, val_pred


# ============================================================
# PSEUDO LABELING
# ============================================================
def pseudo_labeling(X_train, y_train, X_test, n_classes, threshold=0.95, iterations=2):
    """Pseudo Labeling으로 Test 데이터 활용"""
    
    print(f"\n[Pseudo Labeling] threshold={threshold}, iterations={iterations}")
    
    X_current = X_train.copy()
    y_current = y_train.copy()
    
    for it in range(iterations):
        print(f"\n  Iteration {it+1}/{iterations}")
        print(f"  Training samples: {len(X_current)}")
        
        # Train model
        model = lgb.LGBMClassifier(
            n_estimators=500, num_leaves=63, max_depth=8,
            learning_rate=0.05, class_weight='balanced',
            random_state=config.RANDOM_STATE, verbose=-1, n_jobs=-1
        )
        model.fit(X_current, y_current)
        
        # Predict on test
        test_probs = model.predict_proba(X_test)
        max_probs = test_probs.max(axis=1)
        
        # High confidence samples
        high_conf_mask = max_probs >= threshold
        n_high_conf = high_conf_mask.sum()
        
        print(f"  High confidence samples: {n_high_conf} ({n_high_conf/len(X_test)*100:.1f}%)")
        
        if n_high_conf == 0:
            break
        
        # Add pseudo labels
        pseudo_X = X_test[high_conf_mask]
        pseudo_y = test_probs[high_conf_mask].argmax(axis=1)
        
        X_current = np.vstack([X_current, pseudo_X])
        y_current = np.concatenate([y_current, pseudo_y])
        
        # Lower threshold
        threshold = max(threshold - 0.02, 0.9)
    
    return X_current, y_current


# ============================================================
# CROSS VALIDATION
# ============================================================
def run_cv(X, y, X_test, n_classes, class_names):
    """전체 CV 실행"""
    
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    
    # Pseudo labeling으로 확장된 데이터
    if config.USE_PSEUDO_LABELING and len(X_test) > 0:
        X_extended, y_extended = pseudo_labeling(
            X, y, X_test, n_classes,
            config.PSEUDO_THRESHOLD, config.PSEUDO_ITERATIONS
        )
    else:
        X_extended, y_extended = X, y
    
    # OOF / Test predictions
    model_names = ['lgbm', 'xgb', 'catboost']
    if config.USE_EXTRATREES:
        model_names.append('extratrees')
    if config.USE_TWO_STAGE:
        model_names.append('two_stage')
    
    oof_preds = {name: np.zeros((len(X), n_classes)) for name in model_names}
    test_preds = {name: np.zeros((len(X_test), n_classes)) for name in model_names}
    scores = {name: [] for name in model_names}
    
    print(f"\n{'='*60}")
    print(f"Cross Validation with {len(model_names)} models")
    print(f"Models: {model_names}")
    print(f"SMOTE: {config.USE_SMOTE}, Extra Features: {config.USE_EXTRA_FEATURES}")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/{config.N_FOLDS}")
        print(f"{'='*40}")
        
        # 원본 데이터에서 fold 분리
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Pseudo labeling 데이터가 있으면 train에 추가
        if config.USE_PSEUDO_LABELING and len(X_extended) > len(X):
            pseudo_X = X_extended[len(X):]
            pseudo_y = y_extended[len(X):]
            X_train = np.vstack([X_train, pseudo_X])
            y_train = np.concatenate([y_train, pseudo_y])
        
        # LightGBM
        print(f"\n  [LightGBM]", end=" ")
        start = time.time()
        model, val_pred = train_lgbm_fold(X_train, y_train, X_val, y_val, n_classes, config.USE_SMOTE)
        oof_preds['lgbm'][val_idx] = val_pred
        test_preds['lgbm'] += model.predict_proba(X_test) / config.N_FOLDS
        acc = accuracy_score(y_val, val_pred.argmax(axis=1))
        scores['lgbm'].append(acc)
        print(f"Acc: {acc:.4f} ({time.time()-start:.1f}s)")
        
        # XGBoost
        print(f"  [XGBoost]", end=" ")
        start = time.time()
        model, val_pred = train_xgb_fold(X_train, y_train, X_val, y_val, n_classes, config.USE_SMOTE)
        oof_preds['xgb'][val_idx] = val_pred
        test_preds['xgb'] += model.predict_proba(X_test) / config.N_FOLDS
        acc = accuracy_score(y_val, val_pred.argmax(axis=1))
        scores['xgb'].append(acc)
        print(f"Acc: {acc:.4f} ({time.time()-start:.1f}s)")
        
        # CatBoost
        print(f"  [CatBoost]", end=" ")
        start = time.time()
        model, val_pred = train_catboost_fold(X_train, y_train, X_val, y_val, n_classes, config.USE_SMOTE)
        oof_preds['catboost'][val_idx] = val_pred
        test_preds['catboost'] += model.predict_proba(X_test) / config.N_FOLDS
        acc = accuracy_score(y_val, val_pred.argmax(axis=1))
        scores['catboost'].append(acc)
        print(f"Acc: {acc:.4f} ({time.time()-start:.1f}s)")
        
        # ExtraTrees
        if config.USE_EXTRATREES:
            print(f"  [ExtraTrees]", end=" ")
            start = time.time()
            model, val_pred = train_extratrees_fold(X_train, y_train, X_val, y_val, n_classes, config.USE_SMOTE)
            oof_preds['extratrees'][val_idx] = val_pred
            test_preds['extratrees'] += model.predict_proba(X_test) / config.N_FOLDS
            acc = accuracy_score(y_val, val_pred.argmax(axis=1))
            scores['extratrees'].append(acc)
            print(f"Acc: {acc:.4f} ({time.time()-start:.1f}s)")
        
        # 2-Stage
        if config.USE_TWO_STAGE:
            print(f"  [2-Stage]", end=" ")
            start = time.time()
            model, val_pred = train_two_stage_fold(X_train, y_train, X_val, y_val, n_classes, class_names, config.USE_SMOTE)
            oof_preds['two_stage'][val_idx] = val_pred
            test_preds['two_stage'] += model.predict_proba(X_test) / config.N_FOLDS
            acc = accuracy_score(y_val, val_pred.argmax(axis=1))
            scores['two_stage'].append(acc)
            print(f"Acc: {acc:.4f} ({time.time()-start:.1f}s)")
    
    return oof_preds, test_preds, scores


# ============================================================
# ENSEMBLE
# ============================================================
def optimize_ensemble_weights(oof_preds_dict, y_true):
    """앙상블 가중치 최적화"""
    
    model_names = list(oof_preds_dict.keys())
    n_models = len(model_names)
    oof_list = [oof_preds_dict[name] for name in model_names]
    
    def objective(weights):
        w = np.exp(weights) / np.exp(weights).sum()
        ensemble = sum(w[i] * oof_list[i] for i in range(n_models))
        return -accuracy_score(y_true, ensemble.argmax(axis=1))
    
    result = minimize(objective, x0=np.zeros(n_models), method='Nelder-Mead')
    opt_weights = np.exp(result.x) / np.exp(result.x).sum()
    
    return {name: float(w) for name, w in zip(model_names, opt_weights)}


def stacking_ensemble(oof_preds_dict, test_preds_dict, y_true, n_classes):
    """Stacking Meta Learner"""
    
    # Meta features
    oof_meta = np.hstack([oof for oof in oof_preds_dict.values()])
    test_meta = np.hstack([test for test in test_preds_dict.values()])
    
    # 추가 meta features
    for oof in oof_preds_dict.values():
        oof_meta = np.hstack([oof_meta, oof.max(axis=1, keepdims=True), oof.std(axis=1, keepdims=True)])
    for test in test_preds_dict.values():
        test_meta = np.hstack([test_meta, test.max(axis=1, keepdims=True), test.std(axis=1, keepdims=True)])
    
    meta_model = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial', random_state=42)
    meta_model.fit(oof_meta, y_true)
    
    oof_stacking = meta_model.predict_proba(oof_meta)
    test_stacking = meta_model.predict_proba(test_meta)
    
    return oof_stacking, test_stacking


# ============================================================
# MAIN
# ============================================================
def main():
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("NIDS Ultimate Boosted Pipeline")
    print("="*60)
    print(f"2-Stage: {config.USE_TWO_STAGE}")
    print(f"Extra Features: {config.USE_EXTRA_FEATURES}")
    print(f"Pseudo Labeling: {config.USE_PSEUDO_LABELING}")
    print(f"SMOTE: {config.USE_SMOTE}")
    print(f"ExtraTrees: {config.USE_EXTRATREES}")
    
    # 1. Load Data
    print("\n[1/5] Loading Data...")
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_test = pd.read_csv(config.TEST_DATA_PATH) if Path(config.TEST_DATA_PATH).exists() else None
    
    print(f"  Train: {df_train.shape}")
    if df_test is not None:
        print(f"  Test: {df_test.shape}")
    
    # 2. Feature Engineering
    print("\n[2/5] Enhanced Feature Engineering...")
    fe = EnhancedFeatureEngineer()
    X, y, _ = fe.fit_transform(df_train, config.LABEL_COL)
    
    class_names = list(fe.target_encoder.classes_)
    n_classes = len(class_names)
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {n_classes}")
    
    if df_test is not None:
        X_test, test_ids = fe.transform(df_test)
    else:
        X_test = np.zeros((0, X.shape[1]))
        test_ids = None
    
    fe.save(output_dir / 'feature_engineer.pkl')
    
    # 3. Cross Validation
    print("\n[3/5] Cross Validation...")
    oof_preds, test_preds, scores = run_cv(X, y, X_test, n_classes, class_names)
    
    # 4. Ensemble
    print("\n[4/5] Ensemble...")
    
    # Individual scores
    print("\nIndividual Model OOF Accuracy:")
    for name, oof in oof_preds.items():
        acc = accuracy_score(y, oof.argmax(axis=1))
        mean_score = np.mean(scores[name])
        print(f"  {name}: {acc:.4f} (CV mean: {mean_score:.4f})")
    
    # Simple average
    oof_avg = sum(oof_preds.values()) / len(oof_preds)
    test_avg = sum(test_preds.values()) / len(test_preds)
    avg_acc = accuracy_score(y, oof_avg.argmax(axis=1))
    print(f"\nSimple Average: {avg_acc:.4f}")
    
    # Optimized weights
    opt_weights = optimize_ensemble_weights(oof_preds, y)
    print(f"Optimized weights:")
    for name, w in opt_weights.items():
        print(f"  {name}: {w:.4f}")
    
    oof_opt = sum(opt_weights[name] * oof_preds[name] for name in oof_preds.keys())
    test_opt = sum(opt_weights[name] * test_preds[name] for name in test_preds.keys())
    opt_acc = accuracy_score(y, oof_opt.argmax(axis=1))
    print(f"Weighted Average: {opt_acc:.4f}")
    
    # Stacking
    print("\n[Stacking]")
    oof_stack, test_stack = stacking_ensemble(oof_preds, test_preds, y, n_classes)
    stack_acc = accuracy_score(y, oof_stack.argmax(axis=1))
    print(f"Stacking: {stack_acc:.4f}")
    
    # Best ensemble
    results = {
        'average': (oof_avg, test_avg, avg_acc),
        'weighted': (oof_opt, test_opt, opt_acc),
        'stacking': (oof_stack, test_stack, stack_acc)
    }
    
    best_name = max(results.keys(), key=lambda k: results[k][2])
    final_oof, final_test, final_acc = results[best_name]
    
    print(f"\nBest Ensemble: {best_name} ({final_acc:.4f})")
    
    # 5. Results
    print("\n[5/5] Final Results...")
    
    final_pred = final_oof.argmax(axis=1)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    print(f"\nBest Ensemble: {best_name}")
    print(f"Final OOF Accuracy: {final_acc:.4f}")
    
    print("\nPer-Class Accuracy:")
    for i, name in enumerate(class_names):
        mask = y == i
        if mask.sum() > 0:
            acc = (final_pred[mask] == i).mean()
            marker = " ← HARD" if name in config.HARD_CLASSES else ""
            print(f"  {name}: {acc:.2%} ({mask.sum():,}){marker}")
    
    print("\nClassification Report:")
    print(classification_report(y, final_pred, target_names=class_names, digits=4))
    
    # Confusion pairs
    print("\nTop Confusion Pairs:")
    cm = confusion_matrix(y, final_pred)
    confusion_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 100:
                confusion_pairs.append((class_names[i], class_names[j], cm[i, j], cm[i, j] / cm[i].sum()))
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_cls, pred_cls, count, rate in confusion_pairs[:10]:
        print(f"  {true_cls:15} → {pred_cls:15}: {count:5} ({rate*100:.1f}%)")
    
    # Save
    np.save(output_dir / 'oof_final.npy', final_oof)
    np.save(output_dir / 'test_final.npy', final_test)
    
    for name, oof in oof_preds.items():
        np.save(output_dir / f'oof_{name}.npy', oof)
    
    # Predictions CSV
    if len(X_test) > 0:
        pred_labels = final_test.argmax(axis=1)
        pred_names = [class_names[p] for p in pred_labels]
        
        result_df = pd.DataFrame({
            'id': test_ids if test_ids is not None else np.arange(len(pred_labels)),
            'prediction': pred_names
        })
        result_df.to_csv(output_dir / 'predictions.csv', index=False)
        
        print("\nTest Prediction Distribution:")
        for name, count in pd.Series(pred_names).value_counts().items():
            print(f"  {name}: {count} ({count/len(pred_names)*100:.1f}%)")
    
    # Summary
    summary = {
        'final_accuracy': float(final_acc),
        'best_ensemble': best_name,
        'config': {
            'use_two_stage': config.USE_TWO_STAGE,
            'use_extra_features': config.USE_EXTRA_FEATURES,
            'use_pseudo_labeling': config.USE_PSEUDO_LABELING,
            'use_smote': config.USE_SMOTE,
            'use_extratrees': config.USE_EXTRATREES
        },
        'ensemble_results': {k: float(v[2]) for k, v in results.items()},
        'optimized_weights': opt_weights,
        'individual_scores': {name: float(np.mean(s)) for name, s in scores.items()},
        'per_class_accuracy': {
            class_names[i]: float((final_pred[y == i] == i).mean())
            for i in range(n_classes) if (y == i).sum() > 0
        }
    }
    
    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / 'class_mapping.json', 'w') as f:
        json.dump({int(i): str(c) for i, c in enumerate(class_names)}, f, indent=2)
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    
    return final_acc


if __name__ == "__main__":
    main()
