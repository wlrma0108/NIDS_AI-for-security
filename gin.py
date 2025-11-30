
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import pickle
import json
from pathlib import Path
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
class Config:
    # 경로 설정
    TRAIN_DATA_PATH = "./train.csv"
    TEST_DATA_PATH = "./test.csv"  # 최종 테스트용 (없으면 무시)
    OUTPUT_DIR = "./gin_output"
    
    # 데이터 설정
    LABEL_COL = "attack_cat"
    VAL_RATIO = 0.15
    RANDOM_STATE = 42
    
    # 그래프 설정
    K_NEIGHBORS = 20          # k-NN의 k
    SIMILARITY_METRIC = 'cosine'  # 'cosine' or 'euclidean'
    
    # 모델 설정
    HIDDEN_DIM = 256
    NUM_LAYERS = 3
    DROPOUT = 0.3
    EPS = 0.1                 # GIN의 epsilon (learnable하게 할 수도 있음)
    USE_BATCH_NORM = True
    USE_SKIP_CONNECTION = True
    
    # 학습 설정
    BATCH_SIZE = 2048
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 15
    
    # Focal Loss 설정
    FOCAL_GAMMA = 1.0
    USE_CLASS_WEIGHT = True
    
    # 기타
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0
    
config = Config()

# ============================================================
# FEATURE ENGINEER (최소한의 전처리)
# ============================================================
class FeatureEngineer:
    """필수적인 전처리만 수행"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.categorical_cols = ['proto', 'service', 'state']
        self.drop_cols = ['id', 'attack_cat', 'label']
        
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
        
        # 카테고리 인코딩
        df = self._encode_categorical(df, fit=True)
        
        # 불필요 컬럼 제거
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors='ignore')
        
        self.feature_names = df.columns.tolist()
        
        # 스케일링
        X = self.scaler.fit_transform(df.values)
        return X, y_encoded
    
    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df = self._preprocess(df)
        df = self._encode_categorical(df, fit=False)
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors='ignore')
        
        # 컬럼 맞추기
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
    
    def _encode_categorical(self, df, fit):
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if fit:
                self.label_encoders[col] = LabelEncoder()
                vals = list(df[col].unique()) + ['<UNK>']
                self.label_encoders[col].fit(vals)
            if col in self.label_encoders:
                known = set(self.label_encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known else '<UNK>')
                df[col] = self.label_encoders[col].transform(df[col])
        return df
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
    
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
# GRAPH CONSTRUCTOR
# ============================================================
class GraphConstructor:
    """k-NN 기반 그래프 구성"""
    
    def __init__(self, k: int = 10, metric: str = 'cosine'):
        self.k = k
        self.metric = metric
        self.nn_model = None
        
    def build_knn_graph(self, X: np.ndarray, fit: bool = True):
        """
        k-NN 그래프 구성
        Returns: edge_index (2, num_edges) - PyG 형식
        """
        n_samples = X.shape[0]
        
        if fit or self.nn_model is None:
            # fit
            if self.metric == 'cosine':
                # cosine similarity를 위해 normalize
                X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                self.nn_model = NearestNeighbors(n_neighbors=self.k + 1, metric='cosine')
                self.nn_model.fit(X_norm)
                distances, indices = self.nn_model.kneighbors(X_norm)
            else:
                self.nn_model = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean')
                self.nn_model.fit(X)
                distances, indices = self.nn_model.kneighbors(X)
        else:
            # transform only
            if self.metric == 'cosine':
                X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                distances, indices = self.nn_model.kneighbors(X_norm)
            else:
                distances, indices = self.nn_model.kneighbors(X)
        
        # edge_index 구성 (self-loop 제외)
        src_nodes = []
        dst_nodes = []
        
        for i in range(n_samples):
            for j in indices[i]:
                if i != j:  # self-loop 제외
                    src_nodes.append(i)
                    dst_nodes.append(j)
        
        edge_index = np.array([src_nodes, dst_nodes], dtype=np.int64)
        
        return edge_index
    
    def build_batch_graph(self, X_batch: np.ndarray, batch_indices: np.ndarray):
        """
        배치 내에서 k-NN 그래프 구성 (학습 시 사용)
        """
        n_batch = X_batch.shape[0]
        k_actual = min(self.k, n_batch - 1)
        
        if k_actual < 1:
            # 배치가 너무 작으면 self-loop만
            return np.array([[i, i] for i in range(n_batch)]).T
        
        if self.metric == 'cosine':
            X_norm = X_batch / (np.linalg.norm(X_batch, axis=1, keepdims=True) + 1e-8)
            nn = NearestNeighbors(n_neighbors=k_actual + 1, metric='cosine')
            nn.fit(X_norm)
            _, indices = nn.kneighbors(X_norm)
        else:
            nn = NearestNeighbors(n_neighbors=k_actual + 1, metric='euclidean')
            nn.fit(X_batch)
            _, indices = nn.kneighbors(X_batch)
        
        src_nodes = []
        dst_nodes = []
        
        for i in range(n_batch):
            for j in indices[i]:
                if i != j:
                    src_nodes.append(i)
                    dst_nodes.append(j)
        
        edge_index = np.array([src_nodes, dst_nodes], dtype=np.int64)
        return edge_index


# ============================================================
# GIN MODEL
# ============================================================
class GINLayer(nn.Module):
    """Single GIN Layer"""
    
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
        """
        x: (N, in_dim)
        edge_index: (2, E)
        """
        src, dst = edge_index
        
        # Aggregate: sum of neighbors
        N = x.size(0)
        aggr = torch.zeros_like(x)
        aggr.index_add_(0, dst, x[src])
        
        # GIN update: (1 + eps) * x + aggr
        out = (1 + self.eps) * x + aggr
        out = self.mlp(out)
        
        return out


class GINModel(nn.Module):
    """Graph Isomorphism Network for NIDS"""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, 
                 dropout=0.3, eps=0.1, use_batch_norm=True, use_skip=True):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_skip = use_skip
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GIN layers
        self.gin_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gin_layers.append(GINLayer(hidden_dim, hidden_dim, eps=eps))
        
        # Batch norms and dropouts
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
            )
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output layers (using all layer representations)
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
        """
        x: (N, in_dim) node features
        edge_index: (2, E) edge indices
        """
        # Input projection
        h = self.input_proj(x)
        
        if self.use_skip:
            layer_outputs = [h]
        
        # GIN layers
        for i in range(self.num_layers):
            h = self.gin_layers[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = self.dropouts[i](h)
            
            if self.use_skip:
                layer_outputs.append(h)
        
        # Combine all layers (skip connection / JK-style)
        if self.use_skip:
            h = torch.cat(layer_outputs, dim=-1)
        
        # Output
        out = self.output(h)
        
        return out


# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
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
    """학습 및 평가"""
    
    def __init__(self, model, config, class_weights=None):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE
        
        # Loss
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        self.criterion = FocalLoss(gamma=config.FOCAL_GAMMA, weight=class_weights)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        
        # Graph constructor
        self.graph_constructor = GraphConstructor(
            k=config.K_NEIGHBORS, 
            metric=config.SIMILARITY_METRIC
        )
        
        # Tracking
        self.best_val_acc = 0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, X, y):
        self.model.train()
        
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        n_batches = (n_samples + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
        
        for batch_idx in range(n_batches):
            start = batch_idx * self.config.BATCH_SIZE
            end = min(start + self.config.BATCH_SIZE, n_samples)
            batch_indices = indices[start:end]
            
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Build graph for this batch
            edge_index = self.graph_constructor.build_batch_graph(X_batch, batch_indices)
            
            # To tensor
            X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            edge_tensor = torch.tensor(edge_index, dtype=torch.long).to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            out = self.model(X_tensor, edge_tensor)
            loss = self.criterion(out, y_tensor)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            pred = out.argmax(dim=1)
            total_loss += loss.item() * len(batch_indices)
            total_correct += (pred == y_tensor).sum().item()
            total_samples += len(batch_indices)
        
        self.scheduler.step()
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def evaluate(self, X, y):
        self.model.eval()
        
        n_samples = X.shape[0]
        total_loss = 0
        all_preds = []
        all_labels = []
        
        n_batches = (n_samples + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
        
        for batch_idx in range(n_batches):
            start = batch_idx * self.config.BATCH_SIZE
            end = min(start + self.config.BATCH_SIZE, n_samples)
            
            X_batch = X[start:end]
            y_batch = y[start:end]
            
            # Build graph
            edge_index = self.graph_constructor.build_batch_graph(X_batch, np.arange(len(X_batch)))
            
            # To tensor
            X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            edge_tensor = torch.tensor(edge_index, dtype=torch.long).to(self.device)
            
            # Forward
            out = self.model(X_tensor, edge_tensor)
            loss = self.criterion(out, y_tensor)
            
            pred = out.argmax(dim=1)
            total_loss += loss.item() * len(X_batch)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch)
        
        avg_loss = total_loss / n_samples
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)
    
    def fit(self, X_train, y_train, X_val, y_val):
        print(f"\n{'='*60}")
        print("Training GIN Model")
        print(f"{'='*60}")
        print(f"Train samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.EPOCHS):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(X_train, y_train)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.evaluate(X_val, y_val)
            
            # History
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            elapsed = time.time() - start_time
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{self.config.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {elapsed:.1f}s")
            
            # Early stopping check
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                print(f"  └─ New best model saved! (Val Acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model
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
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        
        n_samples = X.shape[0]
        all_preds = []
        all_probs = []
        
        n_batches = (n_samples + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
        
        for batch_idx in range(n_batches):
            start = batch_idx * self.config.BATCH_SIZE
            end = min(start + self.config.BATCH_SIZE, n_samples)
            
            X_batch = X[start:end]
            edge_index = self.graph_constructor.build_batch_graph(X_batch, np.arange(len(X_batch)))
            
            X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(self.device)
            edge_tensor = torch.tensor(edge_index, dtype=torch.long).to(self.device)
            
            out = self.model(X_tensor, edge_tensor)
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            
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
    print("GIN-based Network Intrusion Detection System")
    print("="*60)
    
    # -------------------- Data Loading --------------------
    print("\n[1/5] Loading Data...")
    df = pd.read_csv(config.TRAIN_DATA_PATH)
    print(f"  Loaded: {df.shape}")
    
    # -------------------- Feature Engineering --------------------
    print("\n[2/5] Feature Engineering...")
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df, config.LABEL_COL)
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    
    # Class mapping
    class_names = fe.label_encoders['target'].classes_
    n_classes = len(class_names)
    print(f"  Classes: {n_classes}")
    for i, name in enumerate(class_names):
        count = (y == i).sum()
        print(f"    {i}: {name} ({count:,})")
    
    # Save feature engineer
    fe.save(output_dir / 'feature_engineer.pkl')
    
    # Class mapping 저장
    class_mapping = {int(i): str(c) for i, c in enumerate(class_names)}
    with open(output_dir / 'class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # -------------------- Train/Val Split --------------------
    print("\n[3/5] Splitting Data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.VAL_RATIO, 
        random_state=config.RANDOM_STATE, 
        stratify=y
    )
    print(f"  Train: {X_train.shape[0]}")
    print(f"  Val: {X_val.shape[0]}")
    
    # -------------------- Class Weights --------------------
    class_weights = None
    if config.USE_CLASS_WEIGHT:
        class_counts = np.bincount(y_train)
        # Inverse frequency with smoothing
        class_weights = 1.0 / np.sqrt(class_counts + 1)

        class_weights = class_weights / class_weights.sum() * n_classes
        print(f"\n  Class weights: {dict(zip(class_names, class_weights.round(3)))}")
    
    # -------------------- Model --------------------
    print("\n[4/5] Building Model...")
    model = GINModel(
        in_dim=X.shape[1],
        hidden_dim=config.HIDDEN_DIM,
        out_dim=n_classes,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        eps=config.EPS,
        use_batch_norm=config.USE_BATCH_NORM,
        use_skip=config.USE_SKIP_CONNECTION
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # -------------------- Training --------------------
    print("\n[5/5] Training...")
    trainer = Trainer(model, config, class_weights=class_weights)
    history = trainer.fit(X_train, y_train, X_val, y_val)
    
    # -------------------- Final Evaluation --------------------
    print("\n" + "="*60)
    print("Final Evaluation on Validation Set")
    print("="*60)
    
    val_loss, val_acc, val_preds, val_labels = trainer.evaluate(X_val, y_val)
    print(f"\nValidation Accuracy: {val_acc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=class_names, digits=4))
    
    # Per-class accuracy
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
    print(f"\nConfusion matrix saved to {output_dir / 'confusion_matrix.csv'}")
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    
    # -------------------- Summary --------------------
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nOutput files in {output_dir}:")
    print(f"  - best_model.pt (best validation checkpoint)")
    print(f"  - final_model.pt (final model)")
    print(f"  - feature_engineer.pkl")
    print(f"  - class_mapping.json")
    print(f"  - confusion_matrix.csv")
    print(f"  - training_history.json")
    
    return trainer, fe


if __name__ == "__main__":
    main()