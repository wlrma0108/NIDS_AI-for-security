"""
NIDS Calibration + Smart Threshold
====================================

Ultimate 파이프라인 결과(85.72%)를 후처리로 개선

전략:
1. Probability Calibration (Isotonic Regression)
2. 클래스별 Asymmetric Threshold
3. Exploits 억제 + Hard class 부스팅
4. Confidence-based Decision Logic

사용법:
    # Ultimate 먼저 실행 후
    python nids_calibration.py
    
    # 또는 단독 실행 (내부에서 모델 학습)
    python nids_calibration.py --train
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
import lightgbm as lgb
import pickle
import json
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
class Config:
    TRAIN_DATA_PATH = "./train.csv"
    TEST_DATA_PATH = "./test.csv"
    
    # Ultimate 결과 경로 (있으면 로드, 없으면 새로 학습)
    ULTIMATE_OOF_PATH = "./ultimate_output/oof_final.npy"
    ULTIMATE_TEST_PATH = "./ultimate_output/test_final.npy"
    ULTIMATE_FE_PATH = "./ultimate_output/feature_engineer.pkl"
    ULTIMATE_CLASS_MAP_PATH = "./ultimate_output/class_mapping.json"
    
    OUTPUT_DIR = "./calibration_output"
    
    LABEL_COL = "attack_cat"
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # Hard classes (recall 개선 대상)
    HARD_CLASSES = ['DoS', 'Analysis', 'Backdoor']
    
    # Threshold 설정
    EXPLOITS_HIGH_THRESHOLD = 0.65    # Exploits는 이 이상일 때만
    HARD_CLASS_LOW_THRESHOLD = 0.15   # Hard class는 이 이상이면 고려
    
config = Config()

# ============================================================
# FEATURE ENGINEER (Ultimate과 동일)
# ============================================================
class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoder = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.categorical_cols = ['proto', 'service', 'state']
        self.target_encoding_maps = {}
        self.frequency_encoding_maps = {}
        self.global_target_mean = None
        
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
        df = self._create_features(df)
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
        df = self._create_features(df)
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
    
    def _create_features(self, df):
        df['bytes_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
        df['pkts_ratio'] = df['spkts'] / (df['dpkts'] + 1)
        df['load_ratio'] = df['sload'] / (df['dload'] + 1)
        df['total_bytes'] = df['sbytes'] + df['dbytes']
        df['total_pkts'] = df['spkts'] + df['dpkts']
        df['total_loss'] = df['sloss'] + df['dloss']
        df['sbytes_per_pkt'] = df['sbytes'] / (df['spkts'] + 1)
        df['dbytes_per_pkt'] = df['dbytes'] / (df['dpkts'] + 1)
        df['bytes_per_sec'] = df['total_bytes'] / (df['dur'] + 1e-6)
        df['pkts_per_sec'] = df['total_pkts'] / (df['dur'] + 1e-6)
        
        ct_cols = [c for c in df.columns if c.startswith('ct_') and not c.endswith(('_freq','_target'))]
        if ct_cols:
            df['ct_sum'] = df[ct_cols].sum(axis=1)
            df['ct_mean'] = df[ct_cols].mean(axis=1)
            df['ct_max'] = df[ct_cols].max(axis=1)
            df['ct_min'] = df[ct_cols].min(axis=1)
            df['ct_std'] = df[ct_cols].std(axis=1).fillna(0)
        
        df['ttl_ratio'] = df['sttl'] / (df['dttl'] + 1)
        df['ttl_diff'] = df['sttl'] - df['dttl']
        df['ttl_sum'] = df['sttl'] + df['dttl']
        df['window_ratio'] = df['swin'] / (df['dwin'] + 1)
        df['window_diff'] = df['swin'] - df['dwin']
        
        if 'ct_dst_sport_ltm' in df.columns:
            df['ct_dst_sport_x_sttl'] = df['ct_dst_sport_ltm'] * df['sttl']
        if 'ct_srv_src' in df.columns and 'ct_srv_dst' in df.columns:
            df['ct_srv_ratio'] = df['ct_srv_src'] / (df['ct_srv_dst'] + 1)
        df['sbytes_x_sttl'] = df['sbytes'] * df['sttl']
        df['rate_x_dur'] = df['rate'] * df['dur']
        
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
                self.label_encoders[col].fit(list(df[col].unique()) + ['<UNK>'])
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
# PROBABILITY CALIBRATION
# ============================================================
class ProbabilityCalibrator:
    """
    클래스별 Isotonic Regression으로 확률 보정
    """
    
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.calibrators = {}
    
    def fit(self, probs, y_true):
        """
        probs: (n_samples, n_classes) - raw probabilities
        y_true: (n_samples,) - true labels
        """
        print("\n  Fitting probability calibrators...")
        
        for cls in range(self.n_classes):
            # Binary target for this class
            y_binary = (y_true == cls).astype(int)
            
            # Isotonic regression
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(probs[:, cls], y_binary)
            
            self.calibrators[cls] = ir
        
        print(f"    Fitted {len(self.calibrators)} calibrators")
    
    def transform(self, probs):
        """확률 보정"""
        calibrated = np.zeros_like(probs)
        
        for cls in range(self.n_classes):
            if cls in self.calibrators:
                calibrated[:, cls] = self.calibrators[cls].predict(probs[:, cls])
            else:
                calibrated[:, cls] = probs[:, cls]
        
        # Normalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = np.where(row_sums > 0, calibrated / row_sums, probs)
        
        return calibrated


# ============================================================
# SMART THRESHOLD OPTIMIZER
# ============================================================
class SmartThresholdOptimizer:
    """
    클래스별 비대칭 threshold 최적화
    - Hard class: 낮은 threshold (적극적 예측)
    - Exploits: 높은 threshold (확신할 때만)
    """
    
    def __init__(self, class_names, hard_classes):
        self.class_names = list(class_names)
        self.hard_classes = hard_classes
        self.hard_indices = [self.class_names.index(c) for c in hard_classes if c in self.class_names]
        self.exploits_idx = self.class_names.index('Exploits') if 'Exploits' in self.class_names else -1
        
        self.thresholds = {}
        self.boost_factors = {}
    
    def optimize(self, probs, y_true):
        """
        Grid search로 최적 threshold/boost 조합 찾기
        """
        print("\n  Optimizing thresholds...")
        
        n_classes = len(self.class_names)
        best_acc = 0
        best_config = None
        
        # Grid search
        for exploits_thresh in np.arange(0.5, 0.85, 0.05):
            for hard_boost in np.arange(1.0, 3.0, 0.25):
                for hard_thresh in np.arange(0.1, 0.35, 0.05):
                    
                    # 확률 조정
                    adjusted = self._adjust_probs(probs, exploits_thresh, hard_boost)
                    
                    # Smart decision
                    pred = self._smart_predict(adjusted, hard_thresh, exploits_thresh)
                    
                    acc = accuracy_score(y_true, pred)
                    
                    # Hard class recall 보너스
                    hard_recall_sum = 0
                    for idx in self.hard_indices:
                        mask = y_true == idx
                        if mask.sum() > 0:
                            hard_recall_sum += (pred[mask] == idx).mean()
                    hard_recall_avg = hard_recall_sum / max(len(self.hard_indices), 1)
                    
                    # 종합 점수: accuracy + hard recall 보너스
                    score = acc + 0.05 * hard_recall_avg
                    
                    if score > best_acc:
                        best_acc = score
                        best_config = {
                            'exploits_thresh': exploits_thresh,
                            'hard_boost': hard_boost,
                            'hard_thresh': hard_thresh,
                            'accuracy': acc,
                            'hard_recall_avg': hard_recall_avg
                        }
        
        self.best_config = best_config
        print(f"    Best config: {best_config}")
        
        return best_config
    
    def _adjust_probs(self, probs, exploits_thresh, hard_boost):
        """확률 조정"""
        adjusted = probs.copy()
        
        # Hard class 부스팅
        for idx in self.hard_indices:
            adjusted[:, idx] *= hard_boost
        
        # Exploits 확률이 threshold 미만이면 감소
        if self.exploits_idx >= 0:
            low_conf_mask = probs[:, self.exploits_idx] < exploits_thresh
            adjusted[low_conf_mask, self.exploits_idx] *= 0.5
        
        # Normalize
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
        
        return adjusted
    
    def _smart_predict(self, probs, hard_thresh, exploits_thresh):
        """Smart decision logic"""
        n_samples = len(probs)
        pred = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            p = probs[i]
            
            # 1. Exploits가 매우 높으면 Exploits
            if self.exploits_idx >= 0 and p[self.exploits_idx] > exploits_thresh:
                pred[i] = self.exploits_idx
                continue
            
            # 2. Hard class 중 threshold 넘는 게 있으면 그것으로
            hard_candidates = []
            for idx in self.hard_indices:
                if p[idx] > hard_thresh:
                    hard_candidates.append((idx, p[idx]))
            
            if hard_candidates:
                # 가장 높은 확률의 hard class
                pred[i] = max(hard_candidates, key=lambda x: x[1])[0]
                continue
            
            # 3. 그 외에는 argmax
            pred[i] = p.argmax()
        
        return pred
    
    def transform(self, probs):
        """최적화된 설정으로 예측"""
        cfg = self.best_config
        adjusted = self._adjust_probs(probs, cfg['exploits_thresh'], cfg['hard_boost'])
        pred = self._smart_predict(adjusted, cfg['hard_thresh'], cfg['exploits_thresh'])
        return pred, adjusted


# ============================================================
# CONFUSION-AWARE ADJUSTMENT
# ============================================================
def analyze_confusion_pairs(y_true, y_pred, class_names):
    """혼동 패턴 분석"""
    
    print("\n  Analyzing confusion patterns...")
    
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)
    
    # 가장 많이 혼동되는 쌍
    confusion_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 100:  # 최소 100개 이상
                confusion_pairs.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': cm[i, j],
                    'rate': cm[i, j] / cm[i].sum() if cm[i].sum() > 0 else 0
                })
    
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
    
    print("\n    Top confusion pairs:")
    for pair in confusion_pairs[:10]:
        print(f"      {pair['true']:15} → {pair['pred']:15}: {pair['count']:5} ({pair['rate']*100:.1f}%)")
    
    return confusion_pairs


def apply_confusion_aware_adjustment(probs, y_true, class_names, confusion_pairs):
    """
    혼동이 많은 쌍에 대해 확률 조정
    """
    print("\n  Applying confusion-aware adjustment...")
    
    adjusted = probs.copy()
    
    # DoS → Exploits 혼동이 많으면
    # Exploits 확률이 높고 DoS 확률도 어느정도 있으면 → DoS 부스팅
    
    dos_idx = class_names.index('DoS') if 'DoS' in class_names else -1
    exploits_idx = class_names.index('Exploits') if 'Exploits' in class_names else -1
    analysis_idx = class_names.index('Analysis') if 'Analysis' in class_names else -1
    backdoor_idx = class_names.index('Backdoor') if 'Backdoor' in class_names else -1
    
    for i in range(len(probs)):
        p = probs[i]
        
        # Case 1: Exploits 예측이지만 DoS도 가능성 있음
        if exploits_idx >= 0 and dos_idx >= 0:
            if p.argmax() == exploits_idx and p[dos_idx] > 0.1 and p[exploits_idx] < 0.6:
                adjusted[i, dos_idx] *= 1.5
                adjusted[i, exploits_idx] *= 0.8
        
        # Case 2: Exploits 예측이지만 Analysis도 가능성 있음
        if exploits_idx >= 0 and analysis_idx >= 0:
            if p.argmax() == exploits_idx and p[analysis_idx] > 0.08 and p[exploits_idx] < 0.6:
                adjusted[i, analysis_idx] *= 1.5
                adjusted[i, exploits_idx] *= 0.85
        
        # Case 3: Exploits 예측이지만 Backdoor도 가능성 있음
        if exploits_idx >= 0 and backdoor_idx >= 0:
            if p.argmax() == exploits_idx and p[backdoor_idx] > 0.08 and p[exploits_idx] < 0.6:
                adjusted[i, backdoor_idx] *= 1.5
                adjusted[i, exploits_idx] *= 0.85
    
    # Normalize
    adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
    
    return adjusted


# ============================================================
# TRAIN BASE MODELS (if Ultimate results not available)
# ============================================================
def train_base_models(X, y, X_test, n_classes):
    """기본 모델 학습"""
    
    print("\n  Training base models...")
    
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    
    oof_lgbm = np.zeros((len(X), n_classes))
    test_lgbm = np.zeros((len(X_test), n_classes))
    
    params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'num_leaves': 127,
        'max_depth': 10,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'class_weight': 'balanced',
        'verbose': -1,
        'random_state': config.RANDOM_STATE
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"    Fold {fold+1}/{config.N_FOLDS}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params, train_data, num_boost_round=1000,
            valid_sets=[val_data], valid_names=['valid'],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        oof_lgbm[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_lgbm += model.predict(X_test) / config.N_FOLDS
    
    acc = accuracy_score(y, oof_lgbm.argmax(axis=1))
    print(f"    Base model OOF accuracy: {acc:.4f}")
    
    return oof_lgbm, test_lgbm


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train from scratch instead of loading Ultimate results')
    args = parser.parse_args()
    
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("NIDS Calibration + Smart Threshold")
    print("="*60)
    
    # 1. Load data
    print("\n[1/6] Loading Data...")
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_test = pd.read_csv(config.TEST_DATA_PATH) if Path(config.TEST_DATA_PATH).exists() else None
    
    print(f"  Train: {df_train.shape}")
    if df_test is not None:
        print(f"  Test: {df_test.shape}")
    
    # 2. Load or train base predictions
    print("\n[2/6] Loading/Training Base Model...")
    
    ultimate_exists = (
        Path(config.ULTIMATE_OOF_PATH).exists() and
        Path(config.ULTIMATE_CLASS_MAP_PATH).exists()
    )
    
    if ultimate_exists and not args.train:
        print("  Loading Ultimate results...")
        
        oof_probs = np.load(config.ULTIMATE_OOF_PATH)
        
        with open(config.ULTIMATE_CLASS_MAP_PATH) as f:
            class_mapping = json.load(f)
        class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
        n_classes = len(class_names)
        
        # y_true 로드
        le = LabelEncoder()
        le.classes_ = np.array(class_names)
        y = le.transform(df_train[config.LABEL_COL].fillna('Normal'))
        
        # Test probs
        if Path(config.ULTIMATE_TEST_PATH).exists():
            test_probs = np.load(config.ULTIMATE_TEST_PATH)
        else:
            test_probs = None
        
        test_ids = df_test['id'].values if df_test is not None else None
        
        print(f"  Loaded OOF: {oof_probs.shape}")
        print(f"  Classes: {class_names}")
        
    else:
        print("  Training from scratch...")
        
        fe = FeatureEngineer()
        X, y, _ = fe.fit_transform(df_train, config.LABEL_COL)
        
        class_names = list(fe.target_encoder.classes_)
        n_classes = len(class_names)
        
        if df_test is not None:
            X_test, test_ids = fe.transform(df_test)
        else:
            X_test = np.zeros((0, X.shape[1]))
            test_ids = None
        
        oof_probs, test_probs = train_base_models(X, y, X_test, n_classes)
        
        fe.save(output_dir / 'feature_engineer.pkl')
    
    # Baseline accuracy
    baseline_acc = accuracy_score(y, oof_probs.argmax(axis=1))
    print(f"\n  Baseline OOF Accuracy: {baseline_acc:.4f}")
    
    # 3. Probability Calibration
    print("\n[3/6] Probability Calibration...")
    
    calibrator = ProbabilityCalibrator(n_classes)
    calibrator.fit(oof_probs, y)
    
    calibrated_probs = calibrator.transform(oof_probs)
    
    calib_acc = accuracy_score(y, calibrated_probs.argmax(axis=1))
    print(f"  After calibration: {calib_acc:.4f}")
    
    # 4. Confusion Analysis
    print("\n[4/6] Confusion Analysis...")
    
    baseline_pred = oof_probs.argmax(axis=1)
    confusion_pairs = analyze_confusion_pairs(y, baseline_pred, class_names)
    
    # 5. Smart Threshold Optimization
    print("\n[5/6] Smart Threshold Optimization...")
    
    # Confusion-aware adjustment 먼저
    adjusted_probs = apply_confusion_aware_adjustment(calibrated_probs, y, class_names, confusion_pairs)
    
    adj_acc = accuracy_score(y, adjusted_probs.argmax(axis=1))
    print(f"  After confusion adjustment: {adj_acc:.4f}")
    
    # Threshold optimization
    optimizer = SmartThresholdOptimizer(class_names, config.HARD_CLASSES)
    best_config = optimizer.optimize(adjusted_probs, y)
    
    final_pred, final_probs = optimizer.transform(adjusted_probs)
    final_acc = accuracy_score(y, final_pred)
    
    print(f"\n  Final OOF Accuracy: {final_acc:.4f}")
    
    # 6. Results
    print("\n[6/6] Final Results...")
    
    print(f"\n{'='*60}")
    print("IMPROVEMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline:              {baseline_acc:.4f}")
    print(f"  After Calibration:     {calib_acc:.4f} ({calib_acc-baseline_acc:+.4f})")
    print(f"  After Confusion Adj:   {adj_acc:.4f} ({adj_acc-baseline_acc:+.4f})")
    print(f"  After Smart Threshold: {final_acc:.4f} ({final_acc-baseline_acc:+.4f})")
    
    print(f"\n{'='*60}")
    print("PER-CLASS PERFORMANCE")
    print(f"{'='*60}")
    
    print(f"\n{'Class':<15} {'Baseline':>10} {'Final':>10} {'Change':>10} {'Samples':>10}")
    print("-" * 60)
    
    for i, name in enumerate(class_names):
        mask = y == i
        if mask.sum() == 0:
            continue
        
        base_recall = (baseline_pred[mask] == i).mean()
        final_recall = (final_pred[mask] == i).mean()
        change = final_recall - base_recall
        
        marker = " ← HARD" if name in config.HARD_CLASSES else ""
        status = "↑" if change > 0.01 else ("↓" if change < -0.01 else "→")
        
        print(f"{name:<15} {base_recall:>10.2%} {final_recall:>10.2%} {status}{abs(change):>8.2%} {mask.sum():>10,}{marker}")
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(classification_report(y, final_pred, target_names=class_names, digits=4))
    
    # Save results
    np.save(output_dir / 'oof_calibrated.npy', final_probs)
    
    # Apply to test
    if test_probs is not None and len(test_probs) > 0:
        print("\nApplying to test set...")
        
        test_calibrated = calibrator.transform(test_probs)
        test_adjusted = apply_confusion_aware_adjustment(test_calibrated, y, class_names, confusion_pairs)
        test_pred, test_final_probs = optimizer.transform(test_adjusted)
        
        np.save(output_dir / 'test_calibrated.npy', test_final_probs)
        
        pred_names = [class_names[p] for p in test_pred]
        
        result_df = pd.DataFrame({
            'id': test_ids if test_ids is not None else np.arange(len(test_pred)),
            'prediction': pred_names
        })
        result_df.to_csv(output_dir / 'predictions.csv', index=False)
        
        print("\nTest Prediction Distribution:")
        for name, count in pd.Series(pred_names).value_counts().items():
            print(f"  {name}: {count} ({count/len(pred_names)*100:.1f}%)")
    
    # Save summary
    summary = {
        'baseline_accuracy': float(baseline_acc),
        'calibrated_accuracy': float(calib_acc),
        'adjusted_accuracy': float(adj_acc),
        'final_accuracy': float(final_acc),
        'improvement': float(final_acc - baseline_acc),
        'best_config': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in best_config.items()},
        'per_class_recall': {
            class_names[i]: {
                'baseline': float((baseline_pred[y == i] == i).mean()),
                'final': float((final_pred[y == i] == i).mean())
            }
            for i in range(n_classes) if (y == i).sum() > 0
        }
    }
    
    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    
    return final_acc


if __name__ == "__main__":
    main()