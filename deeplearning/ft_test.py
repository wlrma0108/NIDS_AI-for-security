"""
FT-Transformer NIDS - Test Inference
=====================================

학습된 FT-Transformer로 테스트 데이터 추론

사용법:
    1. CONFIG에서 경로 설정
    2. python nids_ft_transformer_test.py 실행
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import json
from pathlib import Path
import math

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = "./ft_transformer_output"
TEST_DATA_PATH = "./test.csv"
OUTPUT_PATH = "./predictions.csv"

BATCH_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model config (학습 때와 동일하게)
D_MODEL = 192
N_HEADS = 8
N_LAYERS = 3
D_FF_FACTOR = 4
DROPOUT = 0.2
ATTENTION_DROPOUT = 0.2
FFN_DROPOUT = 0.1

# ============================================================
# MODEL DEFINITION (학습 코드와 동일)
# ============================================================
class NumericalEmbedding(nn.Module):
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(num_features, d_model))
        self.biases = nn.Parameter(torch.empty(num_features, d_model))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.zeros_(self.biases)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        return x * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class CategoricalEmbedding(nn.Module):
    def __init__(self, cardinalities: list, d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, d_model) for card in cardinalities
        ])
    
    def forward(self, x):
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embedded, dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        B, L, D = x.shape
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(context), attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, attention_dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out, attn_weights = self.attention(self.norm1(x))
        x = x + self.dropout1(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        return x, attn_weights


class FTTransformer(nn.Module):
    def __init__(self, num_features, cat_cardinalities, n_classes,
                 d_model=192, n_heads=8, n_layers=3, d_ff_factor=4,
                 dropout=0.2, attention_dropout=0.2, ffn_dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.n_cat_features = len(cat_cardinalities)
        self.d_model = d_model
        
        self.num_embedding = NumericalEmbedding(num_features, d_model)
        if self.n_cat_features > 0:
            self.cat_embedding = CategoricalEmbedding(cat_cardinalities, d_model)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        d_ff = d_model * d_ff_factor
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, attention_dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_model, n_classes)
        )
    
    def forward(self, x_num, x_cat=None):
        B = x_num.size(0)
        tokens = self.num_embedding(x_num)
        
        if x_cat is not None and self.n_cat_features > 0:
            cat_tokens = self.cat_embedding(x_cat)
            tokens = torch.cat([tokens, cat_tokens], dim=1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        for layer in self.layers:
            tokens, _ = layer(tokens)
        
        tokens = self.final_norm(tokens)
        cls_output = tokens[:, 0]
        return self.head(cls_output), None


# ============================================================
# FEATURE ENGINEER (로드용)
# ============================================================
class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []
        self.num_features = 0
        self.cat_cardinalities = {}
        self.categorical_cols = ['proto', 'service', 'state']
        self.drop_cols = ['id', 'attack_cat', 'label']
    
    def transform(self, df):
        df = df.copy()
        df = self._preprocess(df)
        cat_df, num_df = self._split_features(df)
        num_scaled = self.scaler.transform(num_df.values)
        cat_encoded = self._encode_categorical(cat_df)
        return num_scaled, cat_encoded
    
    def _preprocess(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('-').astype(str)
        return df
    
    def _split_features(self, df):
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors='ignore')
        cat_cols = [c for c in self.categorical_cols if c in df.columns]
        num_cols = [c for c in df.columns if c not in cat_cols]
        return df[cat_cols], df[num_cols]
    
    def _encode_categorical(self, df):
        encoded = {}
        for col in df.columns:
            if col in self.label_encoders:
                known = set(self.label_encoders[col].classes_)
                col_data = df[col].apply(lambda x: x if x in known else '<UNK>')
                encoded[col] = self.label_encoders[col].transform(col_data)
        return np.column_stack(list(encoded.values())) if encoded else np.array([])
    
    @classmethod
    def load(cls, path):
        fe = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        fe.label_encoders = data['label_encoders']
        fe.scaler = data['scaler']
        fe.feature_names = data['feature_names']
        fe.num_features = data['num_features']
        fe.cat_cardinalities = data['cat_cardinalities']
        return fe


# ============================================================
# MAIN
# ============================================================
def main():
    model_dir = Path(MODEL_DIR)
    
    print("="*60)
    print("FT-Transformer NIDS - Test Inference")
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
    print(f"  Numerical features: {fe.num_features}")
    print(f"  Categorical cardinalities: {fe.cat_cardinalities}")
    
    # Load model
    print("\nLoading model...")
    cat_cardinalities = list(fe.cat_cardinalities.values())
    
    model = FTTransformer(
        num_features=fe.num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff_factor=D_FF_FACTOR,
        dropout=DROPOUT,
        attention_dropout=ATTENTION_DROPOUT,
        ffn_dropout=FFN_DROPOUT
    )
    
    checkpoint = torch.load(model_dir / 'best_model.pt', map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    print("  Model loaded!")
    
    # Load test data
    print(f"\nLoading test data from {TEST_DATA_PATH}...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    print(f"  Samples: {len(df_test)}")
    
    # Get IDs
    ids = df_test['id'].values if 'id' in df_test.columns else np.arange(len(df_test))
    
    # Transform
    print("\nTransforming features...")
    X_num, X_cat = fe.transform(df_test)
    print(f"  Numerical shape: {X_num.shape}")
    print(f"  Categorical shape: {X_cat.shape}")
    
    # Create DataLoader
    test_dataset = TensorDataset(
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(X_cat, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Inference
    print("\nRunning inference...")
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x_num, x_cat = batch
            x_num = x_num.to(DEVICE)
            x_cat = x_cat.to(DEVICE) if x_cat.numel() > 0 else None
            
            logits, _ = model(x_num, x_cat)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {(i+1) * BATCH_SIZE}/{len(df_test)} samples")
    
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