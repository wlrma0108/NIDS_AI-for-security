"""
NIDS EDA & Feature Engineering 통합 스크립트
============================================

사용법:
    1. 아래 CONFIG 섹션에서 경로만 수정
    2. python nids_eda_fe.py 실행
    3. 결과는 OUTPUT_DIR에 저장됨

출력 파일:
    - eda_results.json: EDA 분석 결과
    - numerical_statistics.csv: 수치형 변수 통계
    - discriminative_power.csv: 피처 판별력 점수
    - feature_importance.csv: MI 기반 피처 중요도
    - feature_engineer.pkl: 저장된 전처리기 (테스트 시 사용)
    - class_mapping.json: 클래스 인코딩 매핑
    - X_train.npy, y_train.npy: 전처리된 학습 데이터
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG - 여기만 수정하면 됨
# ============================================================
TRAIN_DATA_PATH = "./train.csv"      # 학습 데이터 경로
OUTPUT_DIR = "./nids_output"          # 출력 디렉토리
LABEL_COL = "attack_cat"              # 레이블 컬럼명
SAVE_PROCESSED_DATA = True            # 전처리된 데이터 저장 여부
# ============================================================


class FeatureEngineer:
    """NIDS 피처 엔지니어링 클래스"""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []
        self.categorical_cols = ['proto', 'service', 'state']
        self.drop_cols = ['id', 'attack_cat', 'label']
        
    def fit_transform(self, df: pd.DataFrame, label_col: str = 'attack_cat'):
        """학습 데이터 fit + transform"""
        df = df.copy()
        
        # 레이블 분리 및 인코딩
        y_encoded = None
        if label_col in df.columns:
            y = df[label_col].fillna('Normal')
            self.label_encoders['target'] = LabelEncoder()
            y_encoded = self.label_encoders['target'].fit_transform(y)
        
        # 피처 엔지니어링 파이프라인
        df = self._preprocess(df, fit=True)
        df = self._create_ratio_features(df)
        df = self._create_interaction_features(df)
        df = self._create_aggregation_features(df)
        df = self._log_transform(df)
        df = self._encode_categorical(df, fit=True)
        
        # 불필요 컬럼 제거
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns])
        self.feature_names = df.columns.tolist()
        
        return df, y_encoded
    
    def transform(self, df: pd.DataFrame):
        """테스트 데이터 transform"""
        df = df.copy()
        
        df = self._preprocess(df, fit=False)
        df = self._create_ratio_features(df)
        df = self._create_interaction_features(df)
        df = self._create_aggregation_features(df)
        df = self._log_transform(df)
        df = self._encode_categorical(df, fit=False)
        
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns])
        
        # 컬럼 맞추기
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_names]
    
    def _preprocess(self, df, fit):
        """기본 전처리"""
        df = df.replace([np.inf, -np.inf], np.nan)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('-').astype(str)
        return df
    
    def _create_ratio_features(self, df):
        """비율 피처"""
        # bytes/packets ratio
        df['bytes_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
        df['pkts_ratio'] = df['spkts'] / (df['dpkts'] + 1)
        
        # load ratio
        if 'sload' in df.columns and 'dload' in df.columns:
            df['load_ratio'] = df['sload'] / (df['dload'] + 1)
        
        # mean size ratio
        if 'smean' in df.columns and 'dmean' in df.columns:
            df['mean_ratio'] = df['smean'] / (df['dmean'] + 1)
        
        # TTL ratio
        if 'sttl' in df.columns and 'dttl' in df.columns:
            df['ttl_ratio'] = df['sttl'] / (df['dttl'] + 1)
        
        # bytes per packet
        df['sbytes_per_pkt'] = df['sbytes'] / (df['spkts'] + 1)
        df['dbytes_per_pkt'] = df['dbytes'] / (df['dpkts'] + 1)
        
        # loss ratio
        if 'sloss' in df.columns and 'dloss' in df.columns:
            df['loss_ratio'] = df['sloss'] / (df['sloss'] + df['dloss'] + 1)
        
        # jitter ratio
        if 'sjit' in df.columns and 'djit' in df.columns:
            df['jit_ratio'] = df['sjit'] / (df['djit'] + 1)
        
        # inter-packet time ratio
        if 'sinpkt' in df.columns and 'dinpkt' in df.columns:
            df['intpkt_ratio'] = df['sinpkt'] / (df['dinpkt'] + 1)
        
        return df
    
    def _create_interaction_features(self, df):
        """상호작용 피처"""
        # totals
        df['total_bytes'] = df['sbytes'] + df['dbytes']
        df['total_pkts'] = df['spkts'] + df['dpkts']
        
        if 'sloss' in df.columns and 'dloss' in df.columns:
            df['total_loss'] = df['sloss'] + df['dloss']
        
        # per-second features
        if 'dur' in df.columns:
            df['bytes_per_sec'] = df['total_bytes'] / (df['dur'] + 1e-6)
            df['pkts_per_sec'] = df['total_pkts'] / (df['dur'] + 1e-6)
        
        # TCP features
        if all(c in df.columns for c in ['tcprtt', 'synack', 'ackdat']):
            df['tcp_setup_ratio'] = df['synack'] / (df['tcprtt'] + 1e-6)
            df['tcp_setup_diff'] = df['synack'] - df['ackdat']
        
        # window features
        if 'swin' in df.columns and 'dwin' in df.columns:
            df['window_product'] = df['swin'] * df['dwin']
            df['window_diff'] = df['swin'] - df['dwin']
        
        # inter-packet diff
        if 'sinpkt' in df.columns and 'dinpkt' in df.columns:
            df['intpkt_diff'] = df['sinpkt'] - df['dinpkt']
        
        return df
    
    def _create_aggregation_features(self, df):
        """ct_* 집계 피처"""
        ct_cols = [c for c in df.columns if c.startswith('ct_')]
        if ct_cols:
            df['ct_sum'] = df[ct_cols].sum(axis=1)
            df['ct_mean'] = df[ct_cols].mean(axis=1)
            df['ct_max'] = df[ct_cols].max(axis=1)
            df['ct_std'] = df[ct_cols].std(axis=1).fillna(0)
            
            if 'ct_srv_src' in df.columns and 'ct_srv_dst' in df.columns:
                df['srv_concentration'] = df['ct_srv_src'] / (df['ct_srv_dst'] + 1)
            
            if 'ct_dst_ltm' in df.columns and 'ct_src_ltm' in df.columns:
                df['ltm_ratio'] = df['ct_dst_ltm'] / (df['ct_src_ltm'] + 1)
        return df
    
    def _log_transform(self, df):
        """로그 변환"""
        log_cols = ['sbytes', 'dbytes', 'sload', 'dload', 'rate', 
                    'total_bytes', 'bytes_per_sec', 'response_body_len']
        for col in log_cols:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
        return df
    
    def _encode_categorical(self, df, fit):
        """범주형 인코딩"""
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if fit:
                self.label_encoders[col] = LabelEncoder()
                vals = df[col].unique().tolist() + ['<UNK>']
                self.label_encoders[col].fit(vals)
            
            if col in self.label_encoders:
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


def run_eda(df, label_col, output_dir):
    """EDA 실행"""
    print("\n" + "="*60)
    print("EDA ANALYSIS")
    print("="*60)
    
    results = {}
    
    # 기본 정보
    results['shape'] = list(df.shape)
    results['columns'] = df.columns.tolist()
    print(f"\n[Shape] {df.shape}")
    
    # 컬럼 분류
    cat_cols = ['proto', 'service', 'state', 'attack_cat', 'label']
    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols + ['id']]
    
    # 1. 결측치
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    results['missing'] = missing.to_dict()
    print(f"\n[Missing Values] {len(missing)} columns with missing values")
    if len(missing) > 0:
        for col, cnt in missing.items():
            print(f"  - {col}: {cnt} ({cnt/len(df)*100:.2f}%)")
    
    # 2. 클래스 분포
    if label_col in df.columns:
        class_dist = df[label_col].value_counts()
        results['class_distribution'] = class_dist.to_dict()
        results['imbalance_ratio'] = float(class_dist.max() / class_dist.min())
        
        print(f"\n[Class Distribution]")
        for cls, cnt in class_dist.items():
            print(f"  {cls}: {cnt:,} ({cnt/len(df)*100:.2f}%)")
        print(f"  Imbalance Ratio: {results['imbalance_ratio']:.2f}")
    
    # 3. 수치형 통계
    num_stats = df[num_cols].describe(percentiles=[.01, .25, .5, .75, .99]).T
    num_stats['skewness'] = df[num_cols].skew()
    num_stats['zero_ratio'] = (df[num_cols] == 0).sum() / len(df) * 100
    num_stats.to_csv(output_dir / 'numerical_statistics.csv')
    
    # 4. 고상관 피처
    print(f"\n[High Correlation Pairs (>0.95)]")
    corr = df[num_cols].corr()
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.95:
                high_corr.append({
                    'f1': corr.columns[i],
                    'f2': corr.columns[j],
                    'corr': round(corr.iloc[i, j], 4)
                })
    results['high_correlation_pairs'] = high_corr
    print(f"  Found {len(high_corr)} pairs")
    for p in high_corr[:10]:
        print(f"  - {p['f1']} <-> {p['f2']}: {p['corr']}")
    if len(high_corr) > 10:
        print(f"  ... and {len(high_corr) - 10} more")
    
    # 5. 판별력 (Discriminative Power)
    if label_col in df.columns:
        print(f"\n[Top 15 Discriminative Features]")
        disc_power = {}
        overall_mean = df[num_cols].mean()
        class_means = df.groupby(label_col)[num_cols].mean()
        
        for col in num_cols:
            between_var = ((class_means[col] - overall_mean[col])**2).mean()
            within_var = df.groupby(label_col)[col].var().mean()
            disc_power[col] = between_var / (within_var + 1e-10)
        
        disc_df = pd.DataFrame([
            {'feature': k, 'disc_power': v} 
            for k, v in sorted(disc_power.items(), key=lambda x: -x[1])
        ])
        disc_df.to_csv(output_dir / 'discriminative_power.csv', index=False)
        results['top_discriminative'] = disc_df.head(15).to_dict('records')
        
        for i, row in disc_df.head(15).iterrows():
            print(f"  {i+1:2d}. {row['feature']}: {row['disc_power']:.4f}")
    
    # 6. 범주형 분석
    print(f"\n[Categorical Features]")
    for col in cat_cols:
        if col in df.columns and col != label_col:
            nunique = df[col].nunique()
            print(f"  {col}: {nunique} unique values")
            top3 = df[col].value_counts().head(3)
            for val, cnt in top3.items():
                print(f"    - {val}: {cnt:,}")
    
    # 결과 저장
    with open(output_dir / 'eda_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def run_feature_engineering(df, label_col, output_dir):
    """피처 엔지니어링 실행"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df, label_col)
    
    print(f"\n[Transformation Complete]")
    print(f"  Original: {df.shape}")
    print(f"  Transformed: {X.shape}")
    print(f"  New features: {X.shape[1] - df.shape[1] + len(fe.drop_cols)}")
    
    # 피처 중요도 (MI)
    print(f"\n[Computing Feature Importance...]")
    n_samples = min(10000, len(X))
    idx = np.random.choice(len(X), n_samples, replace=False)
    mi_scores = mutual_info_classif(X.iloc[idx], y[idx], random_state=42)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    print(f"\n[Top 20 Features by Mutual Information]")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']}: {row['mi_score']:.4f}")
    
    # 클래스 매핑 저장
    class_map = {int(i): c for i, c in enumerate(fe.label_encoders['target'].classes_)}
    with open(output_dir / 'class_mapping.json', 'w') as f:
        json.dump(class_map, f, indent=2)
    
    print(f"\n[Class Mapping]")
    for idx, cls in class_map.items():
        print(f"  {idx}: {cls}")
    
    # 전처리기 저장
    fe.save(output_dir / 'feature_engineer.pkl')
    
    return X, y, fe, importance_df


def main():
    """메인 실행"""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("NIDS EDA & Feature Engineering")
    print("="*60)
    print(f"\nConfig:")
    print(f"  Data: {TRAIN_DATA_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Label: {LABEL_COL}")
    
    # 데이터 로드
    print(f"\nLoading data...")
    df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"Loaded: {df.shape}")
    
    # EDA
    eda_results = run_eda(df, LABEL_COL, output_dir)
    
    # Feature Engineering
    X, y, fe, importance_df = run_feature_engineering(df, LABEL_COL, output_dir)
    
    # 전처리된 데이터 저장
    if SAVE_PROCESSED_DATA:
        print(f"\n[Saving processed data...]")
        np.save(output_dir / 'X_train.npy', X.values)
        np.save(output_dir / 'y_train.npy', y)
        
        # 피처 이름도 저장
        with open(output_dir / 'feature_names.json', 'w') as f:
            json.dump(fe.feature_names, f)
    
    # 요약
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nOutput files in {OUTPUT_DIR}:")
    print(f"  - eda_results.json")
    print(f"  - numerical_statistics.csv")
    print(f"  - discriminative_power.csv")
    print(f"  - feature_importance.csv")
    print(f"  - feature_engineer.pkl")
    print(f"  - class_mapping.json")
    if SAVE_PROCESSED_DATA:
        print(f"  - X_train.npy ({X.shape})")
        print(f"  - y_train.npy ({y.shape})")
        print(f"  - feature_names.json")
    
    print(f"\n다음 단계: 결과 파일 확인 후 GNN 모델링 진행")
    
    return X, y, fe


if __name__ == "__main__":
    main()