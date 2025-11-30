"""
NIDS Advanced Ensemble Pipeline v2
===================================

개선사항:
1. Target Encoding (K-Fold 기반)
2. Frequency Encoding
3. Feature Interactions
4. Stacking Meta Learner
5. 앙상블 가중치 최적화
6. Threshold Optimization

사용법:
    1. CONFIG 섹션에서 경로 설정
    2. python nids_ensemble_v2.py 실행
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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

# ============================================================
# CONFIG
# ============================================================
class Config:
    # 경로
    TRAIN_DATA_PATH = "./train.csv"
    TEST_DATA_PATH = "./test.csv"
    OUTPUT_DIR = "./ensemble_v2_output"
    
    # 데이터
    LABEL_COL = "attack_cat"
    
    # CV 설정
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # 모델 사용 여부
    USE_LGBM = True
    USE_XGB = True
    USE_CATBOOST = True
    
    # Stacking
    USE_STACKING = True
    
    # Feature Engineering
    USE_TARGET_ENCODING = True
    USE_FREQUENCY_ENCODING = True
    USE_FEATURE_INTERACTIONS = True
    
config = Config()

# ============================================================
# ADVANCED FEATURE ENGINEER
# ============================================================
class AdvancedFeatureEngineer:
    """고급 피처 엔지니어링"""
    
    def __init__(self):
        self.label_encoders = {}
        self.target_encoder = None
        self.feature_names = []
        self.categorical_cols = ['proto', 'service', 'state']
        self.cat_indices = []
        
        # Target/Frequency encoding 저장
        self.target_encoding_maps = {}
        self.frequency_encoding_maps = {}
        self.global_target_mean = None
        
    def fit_transform(self, df: pd.DataFrame, label_col: str = None):
        df = df.copy()
        
        # 레이블 처리
        y = None
        if label_col and label_col in df.columns:
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(df[label_col].fillna('Normal'))
            
            # Binary label for target encoding
            df['_target_binary'] = (df[label_col] != 'Normal').astype(int)
        
        # ID 제거
        ids = None
        if 'id' in df.columns:
            ids = df['id'].values
            df = df.drop(columns=['id'])
        if 'label' in df.columns:
            df = df.drop(columns=['label'])
        if label_col and label_col in df.columns:
            df = df.drop(columns=[label_col])
        
        # 기본 전처리
        df = self._preprocess(df)
        
        # 1. Frequency Encoding (fit)
        if config.USE_FREQUENCY_ENCODING:
            df = self._frequency_encoding(df, fit=True)
        
        # 2. Target Encoding (K-Fold, fit)
        if config.USE_TARGET_ENCODING and '_target_binary' in df.columns:
            df = self._target_encoding_cv(df, '_target_binary', fit=True)
            df = df.drop(columns=['_target_binary'])
        elif '_target_binary' in df.columns:
            df = df.drop(columns=['_target_binary'])
        
        # 3. 기본 피처 엔지니어링
        df = self._create_basic_features(df)
        
        # 4. Feature Interactions
        if config.USE_FEATURE_INTERACTIONS:
            df = self._create_interaction_features(df)
        
        # 5. Categorical 인코딩
        df = self._encode_categorical(df, fit=True)
        
        self.feature_names = df.columns.tolist()
        self.cat_indices = [i for i, col in enumerate(self.feature_names) 
                           if col in self.categorical_cols]
        
        return df.values, y, ids
    
    def transform(self, df: pd.DataFrame):
        df = df.copy()
        
        ids = None
        if 'id' in df.columns:
            ids = df['id'].values
            df = df.drop(columns=['id'])
        if 'label' in df.columns:
            df = df.drop(columns=['label'])
        if 'attack_cat' in df.columns:
            df = df.drop(columns=['attack_cat'])
        
        df = self._preprocess(df)
        
        if config.USE_FREQUENCY_ENCODING:
            df = self._frequency_encoding(df, fit=False)
        
        if config.USE_TARGET_ENCODING:
            df = self._target_encoding_cv(df, None, fit=False)
        
        df = self._create_basic_features(df)
        
        if config.USE_FEATURE_INTERACTIONS:
            df = self._create_interaction_features(df)
        
        df = self._encode_categorical(df, fit=False)
        
        # 컬럼 순서 맞추기
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        
        return df.values, ids
    
    def _preprocess(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('-').astype(str)
        return df
    
    def _frequency_encoding(self, df, fit=True):
        """빈도 기반 인코딩"""
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            
            if fit:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                self.frequency_encoding_maps[col] = freq_map
            else:
                freq_map = self.frequency_encoding_maps.get(col, {})
            
            # 빈도 피처 추가
            default_freq = 1.0 / len(freq_map) if freq_map else 0
            df[f'{col}_freq'] = df[col].map(freq_map).fillna(default_freq)
        
        return df
    
    def _target_encoding_cv(self, df, target_col, fit=True):
        """K-Fold 기반 Target Encoding (data leakage 방지)"""
        
        if fit and target_col and target_col in df.columns:
            self.global_target_mean = df[target_col].mean()
            
            for col in self.categorical_cols:
                if col not in df.columns:
                    continue
                
                # 전체 데이터에서 각 카테고리별 평균 (test용)
                target_mean = df.groupby(col)[target_col].mean().to_dict()
                self.target_encoding_maps[col] = target_mean
                
                # CV로 train encoding (leakage 방지)
                encoded = np.zeros(len(df))
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                for train_idx, val_idx in skf.split(df, df[target_col]):
                    train_means = df.iloc[train_idx].groupby(col)[target_col].mean()
                    encoded[val_idx] = df.iloc[val_idx][col].map(train_means).fillna(self.global_target_mean)
                
                df[f'{col}_target'] = encoded
        
        elif not fit:
            # Test data: 저장된 매핑 사용
            for col in self.categorical_cols:
                if col not in df.columns:
                    continue
                
                target_map = self.target_encoding_maps.get(col, {})
                default_val = self.global_target_mean if self.global_target_mean else 0.5
                df[f'{col}_target'] = df[col].map(target_map).fillna(default_val)
        
        return df
    
    def _create_basic_features(self, df):
        """기본 피처 엔지니어링"""
        
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
        ct_cols = [c for c in df.columns if c.startswith('ct_') and not c.endswith('_freq') and not c.endswith('_target')]
        if ct_cols:
            df['ct_sum'] = df[ct_cols].sum(axis=1)
            df['ct_mean'] = df[ct_cols].mean(axis=1)
            df['ct_max'] = df[ct_cols].max(axis=1)
            df['ct_min'] = df[ct_cols].min(axis=1)
            df['ct_std'] = df[ct_cols].std(axis=1).fillna(0)
            df['ct_range'] = df['ct_max'] - df['ct_min']
        
        # 6. TTL features
        df['ttl_ratio'] = df['sttl'] / (df['dttl'] + 1)
        df['ttl_diff'] = df['sttl'] - df['dttl']
        df['ttl_sum'] = df['sttl'] + df['dttl']
        
        # 7. TCP features
        df['tcp_setup_ratio'] = df['synack'] / (df['tcprtt'] + 1e-6)
        df['tcp_ack_ratio'] = df['ackdat'] / (df['tcprtt'] + 1e-6)
        
        # 8. Window features
        df['window_ratio'] = df['swin'] / (df['dwin'] + 1)
        df['window_diff'] = df['swin'] - df['dwin']
        df['window_sum'] = df['swin'] + df['dwin']
        
        # 9. Jitter features
        df['jit_ratio'] = df['sjit'] / (df['djit'] + 1)
        df['jit_diff'] = df['sjit'] - df['djit']
        
        # 10. Inter-packet time
        df['intpkt_ratio'] = df['sinpkt'] / (df['dinpkt'] + 1)
        df['intpkt_diff'] = df['sinpkt'] - df['dinpkt']
        
        # 11. Loss ratio
        df['loss_ratio'] = df['sloss'] / (df['total_loss'] + 1)
        
        # 12. Log transforms
        for col in ['sbytes', 'dbytes', 'sload', 'dload', 'rate', 'total_bytes', 'bytes_per_sec']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
        
        return df
    
    def _create_interaction_features(self, df):
        """피처 상호작용"""
        
        # 고판별력 피처 간 상호작용 (EDA 결과 기반)
        if 'ct_dst_sport_ltm' in df.columns and 'sttl' in df.columns:
            df['ct_dst_sport_x_sttl'] = df['ct_dst_sport_ltm'] * df['sttl']
        
        if 'ct_srv_src' in df.columns and 'ct_state_ttl' in df.columns:
            df['ct_srv_x_state_ttl'] = df['ct_srv_src'] * df['ct_state_ttl']
        
        if 'ct_dst_ltm' in df.columns and 'ct_src_ltm' in df.columns:
            df['ct_dst_src_ltm_ratio'] = df['ct_dst_ltm'] / (df['ct_src_ltm'] + 1)
        
        # Rate와 duration 조합
        if 'rate' in df.columns and 'dur' in df.columns:
            df['rate_x_dur'] = df['rate'] * df['dur']
        
        # Bytes와 TTL 조합
        df['sbytes_x_sttl'] = df['sbytes'] * df['sttl']
        
        # ct_* 기반 비율 피처
        if 'ct_srv_src' in df.columns and 'ct_srv_dst' in df.columns:
            df['ct_srv_ratio'] = df['ct_srv_src'] / (df['ct_srv_dst'] + 1)
        
        if 'ct_dst_ltm' in df.columns and 'ct_dst_src_ltm' in df.columns:
            df['ct_dst_complexity'] = df['ct_dst_ltm'] * df['ct_dst_src_ltm']
        
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
        'max_depth': 8,
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
        'learning_rate': 0.10,
        'depth': 8,
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
            lgb.log_evaluation(period=0)
        ]
    )
    
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    return model, val_pred


def train_xgb_fold(X_train, y_train, X_val, y_val, params):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts + 1)
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
    if len(cat_indices) == 0:
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    else:
        train_df = pd.DataFrame(X_train, columns=feature_names)
        val_df = pd.DataFrame(X_val, columns=feature_names)
        
        cat_cols = [feature_names[i] for i in cat_indices]
        for col in cat_cols:
            train_df[col] = train_df[col].astype(int)
            val_df[col] = val_df[col].astype(int)
        
        model = CatBoostClassifier(**params, cat_features=cat_cols)
        model.fit(train_df, y_train, eval_set=(val_df, y_val), use_best_model=True)
        
        val_pred = model.predict_proba(val_df)
        return model, val_pred
    
    val_pred = model.predict_proba(X_val)
    return model, val_pred


# ============================================================
# STACKING
# ============================================================
class StackingEnsemble:
    """Stacking Meta Learner"""
    
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.meta_model = LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            multi_class='multinomial',
            solver='lbfgs',
            random_state=config.RANDOM_STATE
        )
        self.is_fitted = False
    
    def fit(self, oof_preds_dict, y_true):
        """OOF predictions로 meta model 학습"""
        
        # OOF를 피처로 변환
        meta_features = self._create_meta_features(oof_preds_dict)
        
        print(f"  Meta features shape: {meta_features.shape}")
        
        self.meta_model.fit(meta_features, y_true)
        self.is_fitted = True
        
        # Meta model 성능
        meta_pred = self.meta_model.predict(meta_features)
        meta_acc = accuracy_score(y_true, meta_pred)
        print(f"  Meta model OOF accuracy: {meta_acc:.4f}")
        
        return meta_acc
    
    def predict_proba(self, preds_dict):
        """Test predictions로 최종 예측"""
        if not self.is_fitted:
            raise ValueError("Meta model not fitted")
        
        meta_features = self._create_meta_features(preds_dict)
        return self.meta_model.predict_proba(meta_features)
    
    def _create_meta_features(self, preds_dict):
        """예측 확률을 meta features로 변환"""
        features_list = []
        
        for name, pred in preds_dict.items():
            features_list.append(pred)
            
            # 추가 meta features
            features_list.append(pred.max(axis=1, keepdims=True))  # max prob
            features_list.append(pred.std(axis=1, keepdims=True))  # uncertainty
        
        return np.hstack(features_list)


# ============================================================
# WEIGHT OPTIMIZATION
# ============================================================
def optimize_ensemble_weights(oof_preds_dict, y_true):
    """앙상블 가중치 최적화"""
    
    model_names = list(oof_preds_dict.keys())
    n_models = len(model_names)
    
    if n_models == 1:
        return {model_names[0]: 1.0}
    
    oof_list = [oof_preds_dict[name] for name in model_names]
    
    def objective(weights):
        # Softmax 정규화
        w = np.exp(weights) / np.exp(weights).sum()
        ensemble = sum(w[i] * oof_list[i] for i in range(n_models))
        return -accuracy_score(y_true, ensemble.argmax(axis=1))
    
    # 초기값
    x0 = np.zeros(n_models)
    
    # 최적화
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000})
    
    # 최적 가중치
    opt_weights = np.exp(result.x) / np.exp(result.x).sum()
    
    return {name: w for name, w in zip(model_names, opt_weights)}


# ============================================================
# CROSS VALIDATION
# ============================================================
def run_cv(X, y, X_test, fe, n_classes):
    """5-Fold CV 실행"""
    
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    
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
    
    return oof_preds, test_preds, models, scores, feature_importance


# ============================================================
# ENSEMBLE
# ============================================================
def create_ensemble(oof_preds, test_preds, y_true, class_names, n_classes):
    """앙상블 생성"""
    
    print(f"\n{'='*60}")
    print("Ensemble Creation")
    print(f"{'='*60}")
    
    # 1. 개별 모델 성능
    print("\n[Individual Model OOF Scores]")
    model_scores = {}
    for name, pred in oof_preds.items():
        acc = accuracy_score(y_true, pred.argmax(axis=1))
        model_scores[name] = acc
        print(f"  {name}: {acc:.4f}")
    
    # 2. Simple Average
    simple_ensemble = sum(oof_preds.values()) / len(oof_preds)
    simple_acc = accuracy_score(y_true, simple_ensemble.argmax(axis=1))
    print(f"\n[Simple Average] {simple_acc:.4f}")
    
    # 3. Optimized Weights
    print("\n[Optimizing Weights...]")
    opt_weights = optimize_ensemble_weights(oof_preds, y_true)
    
    opt_oof = sum(opt_weights[name] * oof_preds[name] for name in oof_preds.keys())
    opt_test = sum(opt_weights[name] * test_preds[name] for name in test_preds.keys())
    opt_acc = accuracy_score(y_true, opt_oof.argmax(axis=1))
    
    print(f"  Optimized weights: {opt_weights}")
    print(f"  Optimized accuracy: {opt_acc:.4f}")
    
    # 4. Stacking
    stacking_oof = None
    stacking_test = None
    stacking_acc = 0
    
    if config.USE_STACKING and len(oof_preds) >= 2:
        print("\n[Stacking Meta Learner]")
        stacker = StackingEnsemble(n_classes)
        stacking_acc = stacker.fit(oof_preds, y_true)
        
        stacking_oof = stacker.predict_proba(oof_preds)
        stacking_test = stacker.predict_proba(test_preds)
    
    # 5. 최고 성능 선택
    results = {
        'simple': (simple_ensemble, sum(test_preds.values()) / len(test_preds), simple_acc),
        'optimized': (opt_oof, opt_test, opt_acc),
    }
    
    if stacking_oof is not None:
        results['stacking'] = (stacking_oof, stacking_test, stacking_acc)
    
    best_method = max(results.items(), key=lambda x: x[1][2])
    print(f"\n[Best Method] {best_method[0]}: {best_method[1][2]:.4f}")
    
    final_oof, final_test, final_acc = best_method[1]
    
    # Classification Report
    print(f"\n{'='*60}")
    print(f"Best Ensemble Classification Report")
    print(f"{'='*60}")
    print(classification_report(
        y_true, 
        final_oof.argmax(axis=1), 
        target_names=class_names,
        digits=4
    ))
    
    return {
        'final_oof': final_oof,
        'final_test': final_test,
        'final_acc': final_acc,
        'best_method': best_method[0],
        'opt_weights': opt_weights,
        'all_results': {k: v[2] for k, v in results.items()}
    }


# ============================================================
# MAIN
# ============================================================
def main():
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("NIDS Advanced Ensemble Pipeline v2")
    print("="*60)
    print(f"\nFeature Engineering:")
    print(f"  Target Encoding: {config.USE_TARGET_ENCODING}")
    print(f"  Frequency Encoding: {config.USE_FREQUENCY_ENCODING}")
    print(f"  Feature Interactions: {config.USE_FEATURE_INTERACTIONS}")
    print(f"\nModels: LGBM={config.USE_LGBM}, XGB={config.USE_XGB}, CatBoost={config.USE_CATBOOST}")
    print(f"Stacking: {config.USE_STACKING}")
    print(f"Folds: {config.N_FOLDS}")
    
    # -------------------- Data Loading --------------------
    print(f"\n[1/6] Loading Data...")
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    print(f"  Train: {df_train.shape}")
    
    df_test = None
    if Path(config.TEST_DATA_PATH).exists():
        df_test = pd.read_csv(config.TEST_DATA_PATH)
        print(f"  Test: {df_test.shape}")
    
    # -------------------- Feature Engineering --------------------
    print(f"\n[2/6] Feature Engineering...")
    fe = AdvancedFeatureEngineer()
    X, y, train_ids = fe.fit_transform(df_train, config.LABEL_COL)
    
    class_names = fe.target_encoder.classes_
    n_classes = len(class_names)
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Classes: {n_classes}")
    
    for i, name in enumerate(class_names):
        count = (y == i).sum()
        print(f"    {i}: {name} ({count:,})")
    
    # Test transform
    test_ids = None
    if df_test is not None:
        X_test, test_ids = fe.transform(df_test)
        print(f"  Test features: {X_test.shape}")
    else:
        X_test = np.zeros((0, X.shape[1]))
    
    # Save
    fe.save(output_dir / 'feature_engineer.pkl')
    
    class_mapping = {int(i): str(c) for i, c in enumerate(class_names)}
    with open(output_dir / 'class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # -------------------- Cross Validation --------------------
    print(f"\n[3/6] Cross Validation...")
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
    print(f"\n[4/6] Ensemble...")
    ensemble_result = create_ensemble(oof_preds, test_preds, y, class_names, n_classes)
    
    # -------------------- Save Results --------------------
    print(f"\n[5/6] Saving Results...")
    
    # OOF predictions
    np.save(output_dir / 'oof_final.npy', ensemble_result['final_oof'])
    for name, pred in oof_preds.items():
        np.save(output_dir / f'oof_{name}.npy', pred)
    
    # Test predictions
    if len(X_test) > 0:
        np.save(output_dir / 'test_final.npy', ensemble_result['final_test'])
        
        pred_labels = ensemble_result['final_test'].argmax(axis=1)
        pred_names = [class_names[p] for p in pred_labels]
        
        result_df = pd.DataFrame({
            'id': test_ids if test_ids is not None else np.arange(len(pred_labels)),
            'prediction': pred_names,
            'prediction_idx': pred_labels
        })
        
        for i, name in enumerate(class_names):
            result_df[f'prob_{name}'] = ensemble_result['final_test'][:, i]
        
        result_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    # Feature importance
    if len(feature_importance) > 0:
        fi_summary = feature_importance.groupby(['model', 'feature'])['importance'].mean().reset_index()
        fi_summary = fi_summary.sort_values(['model', 'importance'], ascending=[True, False])
        fi_summary.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # Summary
    results_summary = {
        'n_folds': config.N_FOLDS,
        'n_features': X.shape[1],
        'feature_engineering': {
            'target_encoding': config.USE_TARGET_ENCODING,
            'frequency_encoding': config.USE_FREQUENCY_ENCODING,
            'feature_interactions': config.USE_FEATURE_INTERACTIONS
        },
        'best_method': ensemble_result['best_method'],
        'final_accuracy': ensemble_result['final_acc'],
        'all_ensemble_results': ensemble_result['all_results'],
        'optimal_weights': ensemble_result['opt_weights'],
        'model_cv_scores': {name: {'mean': float(np.mean(s)), 'std': float(np.std(s)), 'folds': [float(x) for x in s]} 
                          for name, s in scores.items()}
    }
    
    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Confusion matrix
    cm = confusion_matrix(y, ensemble_result['final_oof'].argmax(axis=1))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_dir / 'confusion_matrix.csv')
    
    # -------------------- Final Report --------------------
    print(f"\n[6/6] Final Report")
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}")
    
    print(f"\nBest Ensemble: {ensemble_result['best_method']}")
    print(f"Final OOF Accuracy: {ensemble_result['final_acc']:.4f}")
    
    print(f"\nPer-Class Accuracy:")
    final_pred = ensemble_result['final_oof'].argmax(axis=1)
    for i, name in enumerate(class_names):
        mask = y == i
        if mask.sum() > 0:
            acc = (final_pred[mask] == y[mask]).mean()
            print(f"  {name}: {acc:.4f} ({mask.sum()} samples)")
    
    if len(feature_importance) > 0:
        print(f"\nTop 15 Important Features (LightGBM):")
        lgbm_fi = fi_summary[fi_summary['model'] == 'lgbm'].head(15)
        for _, row in lgbm_fi.iterrows():
            print(f"  {row['feature']}: {row['importance']:.2f}")
    
    print(f"\nOutput files in {output_dir}:")
    print(f"  - predictions.csv")
    print(f"  - oof_final.npy")
    print(f"  - test_final.npy")
    print(f"  - feature_importance.csv")
    print(f"  - results_summary.json")
    print(f"  - confusion_matrix.csv")
    print(f"  - feature_engineer.pkl")
    
    if len(X_test) > 0:
        print(f"\nTest Prediction Summary:")
        pred_counts = pd.Series(pred_names).value_counts()
        for name, count in pred_counts.items():
            print(f"  {name}: {count} ({count/len(pred_names)*100:.2f}%)")
    
    return ensemble_result, models, fe


if __name__ == "__main__":
    main()