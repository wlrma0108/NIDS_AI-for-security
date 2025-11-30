"""
NIDS Feature Engineering - Step 2: Feature Engineering
EDA 결과를 기반으로 피처 엔지니어링 수행

사용법:
    python 02_feature_engineering.py --data_path /path/to/train.csv --output_dir ./fe_results

이 코드는 다음을 수행:
1. 기본 전처리 (결측치, 인코딩)
2. Ratio/Interaction features
3. Log transformation on skewed features
4. Aggregation features (ct_* 계열 활용)
5. 피처 중요도 기반 선택
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
import argparse
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """NIDS 데이터를 위한 피처 엔지니어링 클래스"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []
        self.categorical_cols = ['proto', 'service', 'state']
        self.drop_cols = ['id', 'attack_cat', 'label']  # 필요시 조정
        
    def fit_transform(self, df: pd.DataFrame, label_col: str = 'attack_cat') -> tuple:
        """학습 데이터에 대해 fit하고 transform"""
        df = df.copy()
        
        # 레이블 분리
        if label_col in df.columns:
            y = df[label_col].copy()
            # 레이블 인코딩
            self.label_encoders['target'] = LabelEncoder()
            y_encoded = self.label_encoders['target'].fit_transform(y.fillna('Normal'))
        else:
            y_encoded = None
        
        # 피처 엔지니어링
        df = self._basic_preprocessing(df, fit=True)
        df = self._create_ratio_features(df)
        df = self._create_interaction_features(df)
        df = self._create_aggregation_features(df)
        df = self._log_transform_skewed(df)
        df = self._encode_categorical(df, fit=True)
        
        # 불필요한 컬럼 제거
        for col in self.drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        self.feature_names = df.columns.tolist()
        
        return df, y_encoded
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """테스트 데이터에 대해 transform만 수행"""
        df = df.copy()
        
        df = self._basic_preprocessing(df, fit=False)
        df = self._create_ratio_features(df)
        df = self._create_interaction_features(df)
        df = self._create_aggregation_features(df)
        df = self._log_transform_skewed(df)
        df = self._encode_categorical(df, fit=False)
        
        # 불필요한 컬럼 제거
        for col in self.drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # 학습 때 있던 컬럼만 유지 (순서 맞춤)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.feature_names]
        
        return df
    
    def _basic_preprocessing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """기본 전처리: 결측치, 무한값 처리"""
        # 무한값 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 수치형 컬럼 결측치: 0으로 (네트워크 데이터 특성상)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(0)
        
        # 범주형 컬럼 결측치: '-'로
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('-').astype(str)
        
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """비율 피처 생성 - 공격 패턴 탐지에 유용"""
        
        # Bytes ratio (송수신 비율)
        df['bytes_ratio'] = np.where(
            df['dbytes'] > 0,
            df['sbytes'] / (df['dbytes'] + 1),
            df['sbytes']
        )
        
        # Packets ratio
        df['pkts_ratio'] = np.where(
            df['dpkts'] > 0,
            df['spkts'] / (df['dpkts'] + 1),
            df['spkts']
        )
        
        # Load ratio
        if 'sload' in df.columns and 'dload' in df.columns:
            df['load_ratio'] = np.where(
                df['dload'] > 0,
                df['sload'] / (df['dload'] + 1),
                df['sload']
            )
        
        # Mean size ratio
        if 'smean' in df.columns and 'dmean' in df.columns:
            df['mean_size_ratio'] = np.where(
                df['dmean'] > 0,
                df['smean'] / (df['dmean'] + 1),
                df['smean']
            )
        
        # TTL ratio
        if 'sttl' in df.columns and 'dttl' in df.columns:
            df['ttl_ratio'] = np.where(
                df['dttl'] > 0,
                df['sttl'] / (df['dttl'] + 1),
                df['sttl']
            )
        
        # Loss ratio
        if 'sloss' in df.columns and 'dloss' in df.columns:
            total_loss = df['sloss'] + df['dloss']
            df['loss_ratio'] = np.where(
                total_loss > 0,
                df['sloss'] / (total_loss + 1),
                0
            )
        
        # Bytes per packet
        df['sbytes_per_pkt'] = np.where(
            df['spkts'] > 0,
            df['sbytes'] / df['spkts'],
            0
        )
        df['dbytes_per_pkt'] = np.where(
            df['dpkts'] > 0,
            df['dbytes'] / df['dpkts'],
            0
        )
        
        # Jitter ratio
        if 'sjit' in df.columns and 'djit' in df.columns:
            df['jit_ratio'] = np.where(
                df['djit'] > 0,
                df['sjit'] / (df['djit'] + 1),
                df['sjit']
            )
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """상호작용 피처 생성"""
        
        # Total bytes & packets
        df['total_bytes'] = df['sbytes'] + df['dbytes']
        df['total_pkts'] = df['spkts'] + df['dpkts']
        
        # Total loss
        if 'sloss' in df.columns and 'dloss' in df.columns:
            df['total_loss'] = df['sloss'] + df['dloss']
        
        # Duration interactions
        if 'dur' in df.columns:
            df['bytes_per_sec'] = np.where(
                df['dur'] > 0,
                df['total_bytes'] / df['dur'],
                df['total_bytes']
            )
            df['pkts_per_sec'] = np.where(
                df['dur'] > 0,
                df['total_pkts'] / df['dur'],
                df['total_pkts']
            )
        
        # TCP handshake features
        if 'tcprtt' in df.columns and 'synack' in df.columns and 'ackdat' in df.columns:
            df['tcp_setup_ratio'] = np.where(
                df['tcprtt'] > 0,
                df['synack'] / (df['tcprtt'] + 1e-6),
                0
            )
            df['tcp_setup_diff'] = df['synack'] - df['ackdat']
        
        # Window size product (연결 품질 지표)
        if 'swin' in df.columns and 'dwin' in df.columns:
            df['window_product'] = df['swin'] * df['dwin']
            df['window_diff'] = df['swin'] - df['dwin']
        
        # Inter-packet time features
        if 'sinpkt' in df.columns and 'dinpkt' in df.columns:
            df['intpkt_diff'] = df['sinpkt'] - df['dinpkt']
            df['intpkt_ratio'] = np.where(
                df['dinpkt'] > 0,
                df['sinpkt'] / (df['dinpkt'] + 1),
                df['sinpkt']
            )
        
        return df
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ct_* 계열 피처 기반 집계 피처"""
        
        ct_cols = [col for col in df.columns if col.startswith('ct_')]
        
        if ct_cols:
            # ct_* 피처들의 합, 평균, 최대값
            df['ct_sum'] = df[ct_cols].sum(axis=1)
            df['ct_mean'] = df[ct_cols].mean(axis=1)
            df['ct_max'] = df[ct_cols].max(axis=1)
            df['ct_std'] = df[ct_cols].std(axis=1)
            
            # 연결 집중도 지표
            if 'ct_srv_src' in df.columns and 'ct_srv_dst' in df.columns:
                df['srv_concentration'] = np.where(
                    df['ct_srv_dst'] > 0,
                    df['ct_srv_src'] / (df['ct_srv_dst'] + 1),
                    df['ct_srv_src']
                )
            
            if 'ct_dst_ltm' in df.columns and 'ct_src_ltm' in df.columns:
                df['ltm_ratio'] = np.where(
                    df['ct_src_ltm'] > 0,
                    df['ct_dst_ltm'] / (df['ct_src_ltm'] + 1),
                    df['ct_dst_ltm']
                )
        
        return df
    
    def _log_transform_skewed(self, df: pd.DataFrame) -> pd.DataFrame:
        """왜도가 높은 피처에 log 변환 적용"""
        
        # 알려진 고왜도 피처들
        skewed_candidates = [
            'sbytes', 'dbytes', 'sload', 'dload', 'rate',
            'total_bytes', 'bytes_per_sec', 'response_body_len'
        ]
        
        for col in skewed_candidates:
            if col in df.columns:
                # log1p 사용 (0 처리)
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """범주형 변수 인코딩"""
        
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
                
            if fit:
                self.label_encoders[col] = LabelEncoder()
                # unknown 카테고리 대비
                unique_vals = df[col].unique().tolist()
                if '<UNKNOWN>' not in unique_vals:
                    unique_vals.append('<UNKNOWN>')
                self.label_encoders[col].fit(unique_vals)
            
            if col in self.label_encoders:
                # 학습 때 없던 카테고리는 <UNKNOWN>으로
                known_classes = set(self.label_encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known_classes else '<UNKNOWN>')
                df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def save(self, path: str):
        """전처리기 저장"""
        with open(path, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'categorical_cols': self.categorical_cols,
                'drop_cols': self.drop_cols
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """전처리기 로드"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        fe = cls()
        fe.label_encoders = data['label_encoders']
        fe.feature_names = data['feature_names']
        fe.categorical_cols = data['categorical_cols']
        fe.drop_cols = data['drop_cols']
        return fe


def compute_feature_importance(X: pd.DataFrame, y: np.ndarray, n_samples: int = 10000) -> pd.DataFrame:
    """Mutual Information 기반 피처 중요도 계산"""
    
    # 샘플링 (대용량 데이터 대비)
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y[idx]
    else:
        X_sample = X
        y_sample = y
    
    print(f"Computing mutual information on {len(X_sample)} samples...")
    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    return importance_df


def main(data_path: str, output_dir: str, label_col: str = 'attack_cat'):
    """메인 피처 엔지니어링 파이프라인"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Original shape: {df.shape}")
    
    # 피처 엔지니어링
    print("\nApplying feature engineering...")
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df, label_col)
    
    print(f"Transformed shape: {X.shape}")
    print(f"Features created: {len(fe.feature_names)}")
    
    # 피처 중요도 계산
    print("\nComputing feature importance...")
    importance_df = compute_feature_importance(X, y)
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # 전처리기 저장
    fe.save(output_dir / 'feature_engineer.pkl')
    
    # 클래스 매핑 저장
    class_mapping = dict(zip(
        range(len(fe.label_encoders['target'].classes_)),
        fe.label_encoders['target'].classes_
    ))
    with open(output_dir / 'class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # 결과 요약
    results = {
        'original_shape': df.shape,
        'transformed_shape': X.shape,
        'n_features': len(fe.feature_names),
        'n_classes': len(class_mapping),
        'class_mapping': class_mapping,
        'feature_names': fe.feature_names,
        'top_20_features': importance_df.head(20).to_dict('records')
    }
    
    with open(output_dir / 'fe_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Feature Engineering Complete!")
    print(f"  - feature_engineer.pkl: 저장된 전처리기")
    print(f"  - feature_importance.csv: 피처 중요도")
    print(f"  - class_mapping.json: 클래스 매핑")
    print(f"  - fe_summary.json: 요약 정보")
    
    # 콘솔 출력
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"\nOriginal features: {df.shape[1]}")
    print(f"Engineered features: {X.shape[1]}")
    print(f"Classes: {len(class_mapping)}")
    
    print("\n[Class Mapping]")
    for idx, cls in class_mapping.items():
        print(f"  {idx}: {cls}")
    
    print("\n[Top 20 Important Features]")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']}: {row['mi_score']:.4f}")
    
    return X, y, fe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NIDS Feature Engineering')
    parser.add_argument('--data_path', type=str, required=True, help='training-set.csv')
    parser.add_argument('--output_dir', type=str, default='./fe_results', help='Output directory')
    parser.add_argument('--label_col', type=str, default='attack_cat', help='Label column name')
    
    args = parser.parse_args()
    main(args.data_path, args.output_dir, args.label_col)