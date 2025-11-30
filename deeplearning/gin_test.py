"""
GIN NIDS - Test Inference Script
=================================

학습된 모델로 테스트 데이터 추론

사용법:
    1. CONFIG에서 경로 설정
    2. python nids_gin_test.py 실행
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import pickle
import json
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = "./gin_output"           # 학습 결과 폴더
TEST_DATA_PATH = "./test.csv"        # 테스트 데이터
OUTPUT_PATH = "./predictions.csv"    # 예측 결과 저장

BATCH_SIZE = 2048
K_NEIGHBORS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# Model Definition (학습 코드와 동일해야 함)
# ============================================================
import torch.nn as nn

class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, eps=0.1, train_eps=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer('eps', torch.tensor([eps]))
    
    def forward(self, x, edge_index):
        src, dst = edge_index
        N = x.size(0)
        aggr = torch.zeros_like(x)
        aggr.index_add_(0, dst, x[src])
        out = (1 + self.eps) * x + aggr
        out = self.mlp(out)
        return out


class GINModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, 
                 dropout=0.3, eps=0.1, use_batch_norm=True, use_skip=True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_skip = use_skip
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gin_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gin_layers.append(GINLayer(hidden_dim, hidden_dim, eps=eps))
        
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
            )
            self.dropouts.append(nn.Dropout(dropout))
        
        if use_skip:
            self.output = nn.Sequential(
                nn.Linear(hidden_dim * (num_layers + 1), hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim)
            )
    
    def forward(self, x, edge_index):
        h = self.input_proj(x)
        if self.use_skip:
            layer_outputs = [h]
        
        for i in range(self.num_layers):
            h = self.gin_layers[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = self.dropouts[i](h)
            if self.use_skip:
                layer_outputs.append(h)
        
        if self.use_skip:
            h = torch.cat(layer_outputs, dim=-1)
        
        out = self.output(h)
        return out


# ============================================================
# Feature Engineer (로드용)
# ============================================================
class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []
        self.categorical_cols = ['proto', 'service', 'state']
        self.drop_cols = ['id', 'attack_cat', 'label']
    
    def transform(self, df):
        df = df.copy()
        df = self._preprocess(df)
        df = self._encode_categorical(df)
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors='ignore')
        
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        
        X = self.scaler.transform(df.values)
        return X
    
    def _preprocess(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('-').astype(str)
        return df
    
    def _encode_categorical(self, df):
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if col in self.label_encoders:
                known = set(self.label_encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known else '<UNK>')
                df[col] = self.label_encoders[col].transform(df[col])
        return df
    
    @classmethod
    def load(cls, path):
        fe = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        fe.label_encoders = data['label_encoders']
        fe.scaler = data['scaler']
        fe.feature_names = data['feature_names']
        return fe


# ============================================================
# Graph Constructor
# ============================================================
def build_batch_graph(X_batch, k=10):
    n = X_batch.shape[0]
    k_actual = min(k, n - 1)
    
    if k_actual < 1:
        return np.array([[i, i] for i in range(n)]).T
    
    X_norm = X_batch / (np.linalg.norm(X_batch, axis=1, keepdims=True) + 1e-8)
    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric='cosine')
    nn.fit(X_norm)
    _, indices = nn.kneighbors(X_norm)
    
    src, dst = [], []
    for i in range(n):
        for j in indices[i]:
            if i != j:
                src.append(i)
                dst.append(j)
    
    return np.array([src, dst], dtype=np.int64)


# ============================================================
# Main
# ============================================================
def main():
    model_dir = Path(MODEL_DIR)
    
    print("="*60)
    print("GIN NIDS - Test Inference")
    print("="*60)
    
    # Load class mapping
    with open(model_dir / 'class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
    n_classes = len(class_names)
    print(f"\nClasses: {class_names}")
    
    # Load feature engineer
    print("\nLoading feature engineer...")
    fe = FeatureEngineer.load(model_dir / 'feature_engineer.pkl')
    n_features = len(fe.feature_names)
    print(f"  Features: {n_features}")
    
    # Load model
    print("\nLoading model...")
    model = GINModel(
        in_dim=n_features,
        hidden_dim=256,
        out_dim=n_classes,
        num_layers=3,
        dropout=0.3,
        eps=0.1,
        use_batch_norm=True,
        use_skip=True
    )
    model.load_state_dict(torch.load(model_dir / 'best_model.pt', map_location=DEVICE)['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    print("  Model loaded!")
    
    # Load test data
    print(f"\nLoading test data from {TEST_DATA_PATH}...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    print(f"  Samples: {len(df_test)}")
    
    # Get IDs if present
    ids = df_test['id'].values if 'id' in df_test.columns else np.arange(len(df_test))
    
    # Transform
    print("\nTransforming features...")
    X_test = fe.transform(df_test)
    print(f"  Shape: {X_test.shape}")
    
    # Inference
    print("\nRunning inference...")
    all_preds = []
    all_probs = []
    
    n_batches = (len(X_test) + BATCH_SIZE - 1) // BATCH_SIZE
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(X_test))
            
            X_batch = X_test[start:end]
            edge_index = build_batch_graph(X_batch, k=K_NEIGHBORS)
            
            X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)
            edge_tensor = torch.tensor(edge_index, dtype=torch.long).to(DEVICE)
            
            out = model(X_tensor, edge_tensor)
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {end}/{len(X_test)} samples")
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Convert to class names
    pred_names = [class_names[p] for p in all_preds]
    
    # Create output DataFrame
    result_df = pd.DataFrame({
        'id': ids,
        'prediction': pred_names,
        'prediction_idx': all_preds
    })
    
    # Add probability columns
    for i, name in enumerate(class_names):
        result_df[f'prob_{name}'] = all_probs[:, i]
    
    # Save
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPredictions saved to {OUTPUT_PATH}")
    
    # Summary
    print("\n" + "="*60)
    print("Prediction Summary")
    print("="*60)
    pred_counts = pd.Series(pred_names).value_counts()
    for name, count in pred_counts.items():
        print(f"  {name}: {count} ({count/len(pred_names)*100:.2f}%)")
    
    print("\nDone!")


if __name__ == "__main__":
    main()