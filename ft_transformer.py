"""
FT-Transformer (Feature Tokenizer Transformer) for NIDS
========================================================

Tabular 데이터를 위한 Transformer 아키텍처
각 피처를 토큰으로 변환 후 Self-Attention으로 피처 간 관계 학습

사용법:
    1. CONFIG 섹션에서 경로 설정
    2. python nids_ft_transformer.py 실행

Reference:
    "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
from pathlib import Path
import math
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
class Config:
    # 경로 설정
    TRAIN_DATA_PATH = "./train.csv"
    TEST_DATA_PATH = "./test.csv"
    OUTPUT_DIR = "./ft_transformer_output"
    
    # 데이터 설정
    LABEL_COL = "attack_cat"
    VAL_RATIO = 0.15
    RANDOM_STATE = 42
    
    # 모델 설정
    D_MODEL = 192                # 임베딩 차원
    N_HEADS = 8                  # Attention heads
    N_LAYERS = 3                 # Transformer layers
    D_FF_FACTOR = 4              # FFN 확장 비율 (d_ff = d_model * factor)
    DROPOUT = 0.2
    ATTENTION_DROPOUT = 0.2
    FFN_DROPOUT = 0.1
    
    # 학습 설정
    BATCH_SIZE = 512
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 15
    WARMUP_RATIO = 0.1
    
    # Class Imbalance 설정
    USE_CLASS_WEIGHT = True
    CLASS_WEIGHT_SMOOTHING = 0.6     # 0: no smoothing, 1: uniform
    USE_WEIGHTED_SAMPLER = False     # True면 class weight 대신 sampler 사용
    FOCAL_LOSS = True
    FOCAL_GAMMA = 1.0
    
    # 기타
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0
    GRADIENT_CLIP = 1.0
    
config = Config()

# ============================================================
# FEATURE ENGINEER
# ============================================================
class FeatureEngineer:
    """필수적인 전처리"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.categorical_cols = ['proto', 'service', 'state']
        self.drop_cols = ['id', 'attack_cat', 'label']
        self.num_features = 0
        self.cat_cardinalities = {}  # 각 categorical feature의 unique 값 수
        
    def fit_transform(self, df: pd.DataFrame, label_col: str = 'attack_cat'):
        df = df.copy()
        
        # 레이블 처리
        y_encoded = None
        if label_col in df.columns:
            y = df[label_col].fillna('Normal')
            self.label_encoders['target'] = LabelEncoder()
            y_encoded = self.label_encoders['target'].fit_transform(y)
        
        # 기본 전처리
        df = self._preprocess(df)
        
        # Categorical/Numerical 분리
        cat_df, num_df = self._split_features(df, fit=True)
        
        # Numerical scaling
        num_scaled = self.scaler.fit_transform(num_df.values)
        
        # Categorical encoding
        cat_encoded = self._encode_categorical(cat_df, fit=True)
        
        self.feature_names = list(num_df.columns) + list(cat_df.columns)
        self.num_features = num_df.shape[1]
        
        return num_scaled, cat_encoded, y_encoded
    
    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df = self._preprocess(df)
        cat_df, num_df = self._split_features(df, fit=False)
        num_scaled = self.scaler.transform(num_df.values)
        cat_encoded = self._encode_categorical(cat_df, fit=False)
        return num_scaled, cat_encoded
    
    def _preprocess(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('-').astype(str)
        return df
    
    def _split_features(self, df, fit=True):
        # Drop unnecessary columns
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors='ignore')
        
        # Split
        cat_cols = [c for c in self.categorical_cols if c in df.columns]
        num_cols = [c for c in df.columns if c not in cat_cols]
        
        return df[cat_cols], df[num_cols]
    
    def _encode_categorical(self, df, fit=True):
        encoded = {}
        for col in df.columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                vals = list(df[col].unique()) + ['<UNK>']
                self.label_encoders[col].fit(vals)
                self.cat_cardinalities[col] = len(vals)
            
            known = set(self.label_encoders[col].classes_)
            col_data = df[col].apply(lambda x: x if x in known else '<UNK>')
            encoded[col] = self.label_encoders[col].transform(col_data)
        
        return np.column_stack(list(encoded.values())) if encoded else np.array([])
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'num_features': self.num_features,
                'cat_cardinalities': self.cat_cardinalities
            }, f)
    
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
# FT-TRANSFORMER MODEL
# ============================================================
class NumericalEmbedding(nn.Module):
    """수치형 피처를 토큰 임베딩으로 변환"""
    
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        
        # 각 수치형 피처마다 별도의 linear projection
        self.weights = nn.Parameter(torch.empty(num_features, d_model))
        self.biases = nn.Parameter(torch.empty(num_features, d_model))
        
        # Initialize
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.zeros_(self.biases)
    
    def forward(self, x):
        """
        x: (batch_size, num_features)
        return: (batch_size, num_features, d_model)
        """
        # x: (B, N) -> (B, N, 1)
        x = x.unsqueeze(-1)
        # weights: (N, D) -> (1, N, D)
        # output: (B, N, D)
        out = x * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)
        return out


class CategoricalEmbedding(nn.Module):
    """범주형 피처를 토큰 임베딩으로 변환"""
    
    def __init__(self, cardinalities: list, d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, d_model)  # +1 for unknown
            for card in cardinalities
        ])
    
    def forward(self, x):
        """
        x: (batch_size, num_cat_features) - LongTensor
        return: (batch_size, num_cat_features, d_model)
        """
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embedded, dim=1)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
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
        """
        x: (batch_size, seq_len, d_model)
        """
        B, L, D = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        
        output = self.W_o(context)
        return output, attn_weights


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single Transformer Block"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, attention_dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, attention_dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # Pre-norm architecture
        attn_out, attn_weights = self.attention(self.norm1(x))
        x = x + self.dropout1(attn_out)
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        
        return x, attn_weights


class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer
    
    각 피처(수치형/범주형)를 토큰으로 변환 후
    [CLS] 토큰과 함께 Transformer에 입력
    """
    
    def __init__(
        self,
        num_features: int,
        cat_cardinalities: list,
        n_classes: int,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff_factor: int = 4,
        dropout: float = 0.2,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_features = num_features
        self.n_cat_features = len(cat_cardinalities)
        self.d_model = d_model
        
        # Feature embeddings
        self.num_embedding = NumericalEmbedding(num_features, d_model)
        
        if self.n_cat_features > 0:
            self.cat_embedding = CategoricalEmbedding(cat_cardinalities, d_model)
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Transformer layers
        d_ff = d_model * d_ff_factor
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, attention_dropout)
            for _ in range(n_layers)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_model, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x_num, x_cat=None):
        """
        x_num: (batch_size, num_features) - FloatTensor
        x_cat: (batch_size, n_cat_features) - LongTensor (optional)
        """
        B = x_num.size(0)
        
        # Embed numerical features
        tokens = self.num_embedding(x_num)  # (B, num_features, d_model)
        
        # Embed categorical features
        if x_cat is not None and self.n_cat_features > 0:
            cat_tokens = self.cat_embedding(x_cat)  # (B, n_cat_features, d_model)
            tokens = torch.cat([tokens, cat_tokens], dim=1)
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1+N, d_model)
        
        # Transformer layers
        attention_maps = []
        for layer in self.layers:
            tokens, attn_weights = layer(tokens)
            attention_maps.append(attn_weights)
        
        # Final normalization
        tokens = self.final_norm(tokens)
        
        # Use [CLS] token for classification
        cls_output = tokens[:, 0]  # (B, d_model)
        
        logits = self.head(cls_output)
        
        return logits, attention_maps


# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================
# TRAINER
# ============================================================
class Trainer:
    def __init__(self, model, config, class_weights=None):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE
        
        # Loss
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        if config.FOCAL_LOSS:
            self.criterion = FocalLoss(gamma=config.FOCAL_GAMMA, weight=weight_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Tracking
        self.best_val_acc = 0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.scheduler = None
    
    def _create_scheduler(self, num_training_steps):
        warmup_steps = int(num_training_steps * self.config.WARMUP_RATIO)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.LEARNING_RATE,
            total_steps=num_training_steps,
            pct_start=self.config.WARMUP_RATIO,
            anneal_strategy='cos'
        )
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            x_num, x_cat, y = batch
            x_num = x_num.to(self.device)
            x_cat = x_cat.to(self.device) if x_cat.numel() > 0 else None
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(x_num, x_cat)
            loss = self.criterion(logits, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            pred = logits.argmax(dim=1)
            total_loss += loss.item() * len(y)
            total_correct += (pred == y).sum().item()
            total_samples += len(y)
        
        return total_loss / total_samples, total_correct / total_samples
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in val_loader:
            x_num, x_cat, y = batch
            x_num = x_num.to(self.device)
            x_cat = x_cat.to(self.device) if x_cat.numel() > 0 else None
            y = y.to(self.device)
            
            logits, _ = self.model(x_num, x_cat)
            loss = self.criterion(logits, y)
            
            pred = logits.argmax(dim=1)
            total_loss += loss.item() * len(y)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)
    
    def fit(self, train_loader, val_loader):
        print(f"\n{'='*60}")
        print("Training FT-Transformer")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"{'='*60}\n")
        
        # Create scheduler
        num_training_steps = len(train_loader) * self.config.EPOCHS
        self._create_scheduler(num_training_steps)
        
        for epoch in range(self.config.EPOCHS):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_preds, val_labels = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            elapsed = time.time() - start_time
            
            print(f"Epoch {epoch+1:3d}/{self.config.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {elapsed:.1f}s")
            
            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                print(f"  └─ New best! (Val Acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        self.load_checkpoint('best_model.pt')
        print(f"\nBest validation accuracy: {self.best_val_acc:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filename):
        path = Path(self.config.OUTPUT_DIR) / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc
        }, path)
    
    def load_checkpoint(self, filename):
        path = Path(self.config.OUTPUT_DIR) / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    @torch.no_grad()
    def predict(self, data_loader):
        self.model.eval()
        all_preds = []
        all_probs = []
        
        for batch in data_loader:
            x_num, x_cat = batch[0], batch[1]
            x_num = x_num.to(self.device)
            x_cat = x_cat.to(self.device) if x_cat.numel() > 0 else None
            
            logits, _ = self.model(x_num, x_cat)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)


# ============================================================
# MAIN
# ============================================================
def main():
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FT-Transformer for Network Intrusion Detection")
    print("="*60)
    
    # -------------------- Data Loading --------------------
    print("\n[1/6] Loading Data...")
    df = pd.read_csv(config.TRAIN_DATA_PATH)
    print(f"  Loaded: {df.shape}")
    
    # -------------------- Feature Engineering --------------------
    print("\n[2/6] Feature Engineering...")
    fe = FeatureEngineer()
    X_num, X_cat, y = fe.fit_transform(df, config.LABEL_COL)
    
    print(f"  Numerical features: {X_num.shape[1]}")
    print(f"  Categorical features: {X_cat.shape[1] if len(X_cat.shape) > 1 else 0}")
    
    class_names = fe.label_encoders['target'].classes_
    n_classes = len(class_names)
    print(f"  Classes: {n_classes}")
    
    for i, name in enumerate(class_names):
        count = (y == i).sum()
        print(f"    {i}: {name} ({count:,})")
    
    # Save
    fe.save(output_dir / 'feature_engineer.pkl')
    class_mapping = {int(i): str(c) for i, c in enumerate(class_names)}
    with open(output_dir / 'class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # -------------------- Train/Val Split --------------------
    print("\n[3/6] Splitting Data...")
    X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
        X_num, X_cat, y,
        test_size=config.VAL_RATIO,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    print(f"  Train: {len(y_train)}")
    print(f"  Val: {len(y_val)}")
    
    # -------------------- Class Weights --------------------
    print("\n[4/6] Computing Class Weights...")
    class_weights = None
    sampler = None
    
    if config.USE_CLASS_WEIGHT or config.USE_WEIGHTED_SAMPLER:
        class_counts = np.bincount(y_train)
        
        # Smoothed inverse frequency
        # smoothing: 0 = pure inverse, 1 = uniform
        smooth = config.CLASS_WEIGHT_SMOOTHING
        inv_freq = 1.0 / (class_counts + 1)
        uniform = np.ones_like(inv_freq) / n_classes
        class_weights = (1 - smooth) * inv_freq + smooth * uniform
        class_weights = class_weights / class_weights.sum() * n_classes
        
        print(f"  Class weights (smoothing={smooth}):")
        for i, name in enumerate(class_names):
            print(f"    {name}: {class_weights[i]:.4f}")
    
    if config.USE_WEIGHTED_SAMPLER:
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(y_train),
            replacement=True
        )
        class_weights = None  # sampler 쓰면 loss weight 안 씀
    
    # -------------------- DataLoaders --------------------
    print("\n[5/6] Creating DataLoaders...")
    
    train_dataset = TensorDataset(
        torch.tensor(X_num_train, dtype=torch.float32),
        torch.tensor(X_cat_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_num_val, dtype=torch.float32),
        torch.tensor(X_cat_val, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # -------------------- Model --------------------
    print("\n[6/6] Building Model...")
    
    cat_cardinalities = list(fe.cat_cardinalities.values())
    
    model = FTTransformer(
        num_features=fe.num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        d_ff_factor=config.D_FF_FACTOR,
        dropout=config.DROPOUT,
        attention_dropout=config.ATTENTION_DROPOUT,
        ffn_dropout=config.FFN_DROPOUT
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # -------------------- Training --------------------
    trainer = Trainer(model, config, class_weights=class_weights)
    history = trainer.fit(train_loader, val_loader)
    
    # -------------------- Final Evaluation --------------------
    print("\n" + "="*60)
    print("Final Evaluation on Validation Set")
    print("="*60)
    
    val_loss, val_acc, val_preds, val_labels = trainer.evaluate(val_loader)
    print(f"\nValidation Accuracy: {val_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=class_names, digits=4))
    
    print("\nPer-Class Accuracy:")
    for i, name in enumerate(class_names):
        mask = val_labels == i
        if mask.sum() > 0:
            acc = (val_preds[mask] == val_labels[mask]).mean()
            print(f"  {name}: {acc:.4f} ({mask.sum()} samples)")
    
    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_dir / 'confusion_matrix.csv')
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    
    # -------------------- Summary --------------------
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nOutput files in {output_dir}:")
    print(f"  - best_model.pt")
    print(f"  - final_model.pt")
    print(f"  - feature_engineer.pkl")
    print(f"  - class_mapping.json")
    print(f"  - confusion_matrix.csv")
    print(f"  - training_history.json")
    
    return trainer, fe


if __name__ == "__main__":
    main()