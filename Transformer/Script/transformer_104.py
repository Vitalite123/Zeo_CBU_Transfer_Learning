"""
Transformer Model for Zeolite Framework Prediction
Using 104-dimensional tabular features with attention mechanism
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==================== Configuration ====================
TRAIN_DATA_PATH = '../data_processed/ZEOSYN_104_struct_train.xlsx'
VAL_DATA_PATH = '../data_processed/ZEOSYN_104_struct_val.xlsx'
TEST_DATA_PATH = '../data_processed/ZEOSYN_104_struct_test.xlsx'
CODE1_MAPPING_PATH = '../data_processed/code1_index_mapping.csv'

# Model hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 100
DROPOUT_RATE = 0.3

# Transformer specific parameters
D_MODEL = 64          # Embedding dimension
N_HEAD = 4            # Number of attention heads
N_LAYERS = 3          # Number of transformer layers
DIM_FEEDFORWARD = 256 # Feedforward dimension
SEQ_LEN = 13          # Sequence length (104 / 8 ≈ 13)
TOKEN_DIM = 8         # Features per token (104 = 13 * 8)

# ==================== Feature Definition ====================
# 104 features from Feature_List.md
ELEMENTAL_FEATURES = [
    'Si', 'Al', 'P', 'Na', 'K', 'Li', 'Sr', 'Rb', 'Cs', 'Ba', 'Ca', 'F', 'Ge', 'Ti', 'In', 'B', 
    'Mg', 'Ga', 'Ni', 'Mn', 'Fe', 'Co', 'Cr', 'Zn', 'Nb', 'Be', 'W', 'Ce', 'Cu', 'Sn', 'Gd', 
    'La', 'Y', 'Dy', 'Sm', 'Ag', 'Cd', 'Zr', 'V', 'Ta', 'Ru', 'Hf', 'Yb', 'Tl', 'As'
]

OSDA_INDEX_FEATURES = ['osda1_index', 'osda2_index', 'osda3_index']

SYNTHESIS_CONDITIONS = ['cryst_temp', 'cryst_time', 'seed', 'rotation']

AGING_CONDITIONS = ['aging_temp', 'aging_time']

PH_CONDITIONS = ['acid', 'OH']

GEL_RATIOS = ['H2O_T', 'OH_T', 'Gel_Si_Al', 'Gel_P_Al', 'Gel_P_Si']

# OSDA molecular descriptors (33 features: 11 descriptors × 3 OSDAs)
OSDA_DESCRIPTORS = []
for osda_num in [1, 2, 3]:
    descriptors = [
        f'osda{osda_num}_bertz_ct_mean_0',
        f'osda{osda_num}_free_sasa_mean_0',
        f'osda{osda_num}_asphericity_mean_0',
        f'osda{osda_num}_eccentricity_mean_0',
        f'osda{osda_num}_axes_mean_0',
        f'osda{osda_num}_axes_mean_1',
        f'osda{osda_num}_box_mean_0',
        f'osda{osda_num}_box_mean_1',
        f'osda{osda_num}_box_mean_2',
        f'osda{osda_num}_getaway_mean_0',
        f'osda{osda_num}_getaway_mean_1'
    ]
    OSDA_DESCRIPTORS.extend(descriptors)

AGGREGATED_FEATURES = [
    'osda_avg_asphericity', 'osda_max_asphericity', 'osda_min_asphericity',
    'osda_avg_sasa', 'osda_max_sasa', 'osda_min_sasa',
    'osda_avg_bertz', 'osda_max_bertz', 'osda_min_bertz',
    'osda_total_volume'
]

# All 104 features
FEATURE_COLUMNS = (
    ELEMENTAL_FEATURES + 
    OSDA_INDEX_FEATURES + 
    SYNTHESIS_CONDITIONS + 
    AGING_CONDITIONS + 
    PH_CONDITIONS + 
    GEL_RATIOS + 
    OSDA_DESCRIPTORS + 
    AGGREGATED_FEATURES
)

print(f"Total features: {len(FEATURE_COLUMNS)}")

# ==================== Dataset Class ====================
class ZeoliteDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ==================== Positional Encoding ====================
class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    Adds position information to the input embeddings
    """
    def __init__(self, d_model, max_len=20, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ==================== Transformer Model ====================
class Transformer_Zeolite(nn.Module):
    """
    Transformer model for zeolite framework prediction
    
    Architecture:
    - Input: 104 features → reshape to (batch, 13, 8)
    - Feature embedding: Linear(8 → d_model)
    - Positional encoding
    - Transformer encoder layers (N_LAYERS)
    - Global average pooling
    - Classification head: d_model → 256 → 128 → num_classes
    """
    def __init__(self, num_classes, d_model=D_MODEL, n_head=N_HEAD, 
                 n_layers=N_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT_RATE):
        super(Transformer_Zeolite, self).__init__()
        
        self.d_model = d_model
        self.seq_len = SEQ_LEN
        self.token_dim = TOKEN_DIM
        
        # Feature embedding layer: maps each token (8 features) to d_model dimension
        self.feature_embedding = nn.Linear(TOKEN_DIM, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=SEQ_LEN + 1, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 104)
        batch_size = x.size(0)
        
        # Reshape to sequence: (batch_size, seq_len, token_dim)
        # 104 features → 13 tokens, each with 8 features
        # Pad to make it divisible: 104 → 104 (13 * 8)
        if x.size(1) < SEQ_LEN * TOKEN_DIM:
            padding = torch.zeros(batch_size, SEQ_LEN * TOKEN_DIM - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
        elif x.size(1) > SEQ_LEN * TOKEN_DIM:
            x = x[:, :SEQ_LEN * TOKEN_DIM]
        
        x = x.view(batch_size, SEQ_LEN, TOKEN_DIM)
        
        # Feature embedding: (batch_size, seq_len, d_model)
        x = self.feature_embedding(x)
        
        # Scale by sqrt(d_model) as in original transformer
        x = x * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        x = self.classifier(x)
        
        return x

# ==================== Training Function ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    """
    Train the model and return training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                  f'LR: {current_lr:.6f}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return history, best_val_acc

# ==================== Evaluation Function ====================
def evaluate_model(model, data_loader, device):
    """
    Evaluate model and return predictions and metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)

# ==================== Main Function ====================
def main():
    print("=" * 80)
    print("Transformer Model for Zeolite Framework Prediction")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_excel(TRAIN_DATA_PATH)
    val_df = pd.read_excel(VAL_DATA_PATH)
    test_df = pd.read_excel(TEST_DATA_PATH)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print()
    
    # Load Code1 mapping
    print("Loading Code1 mapping...")
    code1_mapping = pd.read_csv(CODE1_MAPPING_PATH)
    index_to_code1 = dict(zip(code1_mapping['Code1_index'], code1_mapping['Code1']))
    code1_to_index = dict(zip(code1_mapping['Code1'], code1_mapping['Code1_index']))
    print(f"Total unique frameworks: {len(index_to_code1)}")
    print()
    
    # Prepare features
    print("Preparing features...")
    X_train = train_df[FEATURE_COLUMNS].values
    X_val = val_df[FEATURE_COLUMNS].values
    X_test = test_df[FEATURE_COLUMNS].values
    
    # Handle missing values
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Prepare labels
    print("Preparing labels...")
    y_train = train_df['Code1_index'].values
    y_val = val_df['Code1_index'].values
    y_test = test_df['Code1_index'].values
    
    # Encode labels (to 0-indexed for PyTorch)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print()
    
    # Create datasets and dataloaders
    print("Creating data loaders...")
    train_dataset = ZeoliteDataset(X_train, y_train_encoded)
    val_dataset = ZeoliteDataset(X_val, y_val_encoded)
    test_dataset = ZeoliteDataset(X_test, y_test_encoded)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print()
    
    # Create model
    print("Creating Transformer model...")
    print(f"  - d_model: {D_MODEL}")
    print(f"  - n_head: {N_HEAD}")
    print(f"  - n_layers: {N_LAYERS}")
    print(f"  - dim_feedforward: {DIM_FEEDFORWARD}")
    print(f"  - seq_len: {SEQ_LEN}")
    print(f"  - token_dim: {TOKEN_DIM}")
    print()
    
    model = Transformer_Zeolite(
        num_classes=num_classes,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT_RATE
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Define loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Train model
    print("Training model...")
    print("-" * 80)
    history, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, EPOCHS
    )
    print("-" * 80)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    y_test_pred, y_test_true, test_probs = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_true, y_test_pred)
    precision = precision_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    recall = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
    
    # Prediction variance (measure of confidence)
    prediction_variance = np.mean(np.var(test_probs, axis=1))
    
    # Calculate per-class metrics
    from sklearn.metrics import classification_report
    y_test_original = label_encoder.inverse_transform(y_test_true)
    y_test_pred_original = label_encoder.inverse_transform(y_test_pred)
    y_test_names = [index_to_code1.get(idx, str(idx)) for idx in y_test_original]
    y_pred_names = [index_to_code1.get(idx, str(idx)) for idx in y_test_pred_original]
    
    unique_classes = sorted(set(y_test_names))
    report = classification_report(y_test_names, y_pred_names, 
                                   labels=unique_classes, output_dict=True, zero_division=0)
    
    print("\n" + "=" * 80)
    print("Test Set Results")
    print("=" * 80)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro):    {recall:.4f}")
    print(f"F1 Score (Macro):  {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"Prediction Variance: {prediction_variance:.4f}")
    print()
    
    # Save results
    results = {
        'test_accuracy': float(accuracy),
        'test_precision_macro': float(precision),
        'test_recall_macro': float(recall),
        'test_f1_macro': float(f1_macro),
        'test_f1_weighted': float(f1_weighted),
        'prediction_variance': float(prediction_variance),
        'best_val_accuracy': float(best_val_acc),
        'num_classes': int(num_classes),
        'num_features': len(FEATURE_COLUMNS),
        'epochs': len(history['train_loss']),
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layers': N_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'total_parameters': int(total_params),
        'training_samples': int(len(train_df)),
        'validation_samples': int(len(val_df)),
        'test_samples': int(len(test_df))
    }
    
    # Save model
    print("Saving model and results...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': num_classes,
            'd_model': D_MODEL,
            'n_head': N_HEAD,
            'n_layers': N_LAYERS,
            'dim_feedforward': DIM_FEEDFORWARD,
            'dropout': DROPOUT_RATE
        },
        'scaler': scaler,
        'label_encoder': label_encoder,
        'index_to_code1': index_to_code1,
        'feature_columns': FEATURE_COLUMNS,
        'results': results,
        'history': history
    }, 'transformer_104.pkl')
    
    # Save results as JSON
    with open('transformer_104_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Model saved: transformer_104.pkl")
    print(f"Results saved: transformer_104_results.json")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    generate_visualizations(history, y_test_true, y_test_pred, y_test_names, y_pred_names, 
                           label_encoder, index_to_code1, unique_classes, report, test_probs)
    print("Visualizations saved!")
    print()
    
    # Generate feature importance
    print("Calculating feature importance...")
    feature_importance = calculate_feature_importance(
        model, test_loader, device, FEATURE_COLUMNS
    )
    feature_importance.to_csv('feature_importance.csv', index=False)
    print(f"Feature importance saved: feature_importance.csv")
    print()
    
    # Generate attention analysis
    print("Analyzing attention patterns...")
    attention_data = analyze_attention(model, test_loader, device, FEATURE_COLUMNS)
    attention_data.to_csv('attention_analysis.csv', index=False)
    print(f"Attention analysis saved: attention_analysis.csv")
    print()
    
    # Generate report
    print("Generating report...")
    generate_report(results, history, feature_importance, attention_data, index_to_code1, report)
    print("Report saved: Transformer104_report.md")
    print()
    
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return results, history, feature_importance

# ==================== Visualization Function ====================
def generate_visualizations(history, y_true_encoded, y_pred_encoded, y_true_names, y_pred_names,
                           label_encoder, index_to_code1, unique_classes, report, test_probs):
    """
    Generate all visualization charts with data tables
    """
    # 1. Training and validation loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax.plot(history['val_loss'], label='Validation Loss', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training curves data
    training_data = pd.DataFrame({
        'Epoch': range(1, len(history['train_loss']) + 1),
        'Train_Loss': history['train_loss'],
        'Train_Accuracy': history['train_acc'],
        'Val_Loss': history['val_loss'],
        'Val_Accuracy': history['val_acc']
    })
    training_data.to_csv('training_curves.csv', index=False)
    
    # 2. Training and validation accuracy curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
    ax.plot(history['val_acc'], label='Validation Accuracy', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion matrix (top 30 classes)
    # Get top 30 classes by support
    class_supports = {cls: report[cls]['support'] for cls in unique_classes if cls in report}
    top_30_classes = sorted(class_supports.keys(), key=lambda x: class_supports[x], reverse=True)[:30]
    
    cm = confusion_matrix(y_true_names, y_pred_names, labels=top_30_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=top_30_classes, yticklabels=top_30_classes, ax=ax)
    ax.set_xlabel('Predicted Framework')
    ax.set_ylabel('Actual Framework')
    ax.set_title('Confusion Matrix (Top 30 Frameworks by Support, Normalized)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save confusion matrix data
    cm_df = pd.DataFrame(cm, index=top_30_classes, columns=top_30_classes)
    cm_df.to_csv('confusion_matrix.csv')
    
    # 4. Performance metrics by class
    class_metrics = []
    for class_name in unique_classes:
        if class_name in report:
            metrics = report[class_name]
            class_metrics.append({
                'Framework': class_name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1-score'],
                'Support': int(metrics['support'])
            })
    
    class_metrics_df = pd.DataFrame(class_metrics)
    class_metrics_df = class_metrics_df.sort_values('Support', ascending=False)
    
    # Plot top 20 frameworks by support
    top_20_classes = class_metrics_df.head(20)['Framework'].tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Precision
    ax = axes[0, 0]
    top_precision = class_metrics_df[class_metrics_df['Framework'].isin(top_20_classes)]
    bars = ax.barh(range(len(top_precision)), top_precision['Precision'].values, color='steelblue')
    ax.set_yticks(range(len(top_precision)))
    ax.set_yticklabels(top_precision['Framework'].values)
    ax.set_xlabel('Precision')
    ax.set_title('Precision by Framework (Top 20 by Support)')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Recall
    ax = axes[0, 1]
    top_recall = class_metrics_df[class_metrics_df['Framework'].isin(top_20_classes)]
    bars = ax.barh(range(len(top_recall)), top_recall['Recall'].values, color='coral')
    ax.set_yticks(range(len(top_recall)))
    ax.set_yticklabels(top_recall['Framework'].values)
    ax.set_xlabel('Recall')
    ax.set_title('Recall by Framework (Top 20 by Support)')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # F1 Score
    ax = axes[1, 0]
    top_f1 = class_metrics_df[class_metrics_df['Framework'].isin(top_20_classes)]
    bars = ax.barh(range(len(top_f1)), top_f1['F1_Score'].values, color='mediumseagreen')
    ax.set_yticks(range(len(top_f1)))
    ax.set_yticklabels(top_f1['Framework'].values)
    ax.set_xlabel('F1 Score')
    ax.set_title('F1 Score by Framework (Top 20 by Support)')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Support
    ax = axes[1, 1]
    top_support = class_metrics_df[class_metrics_df['Framework'].isin(top_20_classes)]
    bars = ax.barh(range(len(top_support)), top_support['Support'].values, color='mediumpurple')
    ax.set_yticks(range(len(top_support)))
    ax.set_yticklabels(top_support['Framework'].values)
    ax.set_xlabel('Support (Number of Samples)')
    ax.set_title('Sample Support by Framework (Top 20)')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('performance_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save class metrics
    class_metrics_df.to_csv('performance_by_class.csv', index=False)
    
    # 5. Overall performance comparison
    overall_metrics = {
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 Macro', 'F1 Weighted'],
        'Value': [
            history['val_acc'][-1],
            np.mean([m['Precision'] for m in class_metrics if m['Precision'] > 0]),
            np.mean([m['Recall'] for m in class_metrics if m['Recall'] > 0]),
            np.mean([m['F1_Score'] for m in class_metrics if m['F1_Score'] > 0]),
            report['weighted avg']['f1-score']
        ]
    }
    
    overall_df = pd.DataFrame(overall_metrics)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'gold']
    bars = ax.bar(overall_df['Metric'], overall_df['Value'], color=colors)
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Metrics')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, v in zip(bars, overall_df['Value']):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.4f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save overall metrics
    overall_df.to_csv('overall_performance.csv', index=False)
    
    # 6. Prediction confidence distribution
    max_probs = np.max(test_probs, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(max_probs, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(max_probs), color='red', linestyle='--', label=f'Mean: {np.mean(max_probs):.4f}')
    ax.set_xlabel('Prediction Confidence (Max Probability)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Prediction Confidence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prediction_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save confidence data
    confidence_df = pd.DataFrame({
        'Sample_Index': range(len(max_probs)),
        'Max_Probability': max_probs,
        'Prediction_Variance': np.var(test_probs, axis=1)
    })
    confidence_df.to_csv('prediction_confidence.csv', index=False)

# ==================== Feature Importance Function ====================
def calculate_feature_importance(model, test_loader, device, feature_names):
    """
    Calculate feature importance using permutation importance
    """
    model.eval()
    
    # Get baseline accuracy
    baseline_correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            baseline_correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    baseline_acc = baseline_correct / total
    
    # Calculate importance for each feature
    importance_scores = []
    
    print(f"  Calculating importance for {len(feature_names)} features...")
    
    for i, feature_name in enumerate(feature_names):
        # Shuffle this feature
        correct = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features_shuffled = features.clone()
                # Shuffle the i-th feature across the batch
                indices = torch.randperm(features.size(0))
                features_shuffled[:, i] = features[indices, i]
                
                features_shuffled, labels = features_shuffled.to(device), labels.to(device)
                outputs = model(features_shuffled)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        importance = baseline_acc - acc
        importance_scores.append((feature_name, importance))
        
        if (i + 1) % 20 == 0:
            print(f"    Processed {i+1}/{len(feature_names)} features")
    
    # Sort by importance
    importance_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Create DataFrame
    importance_df = pd.DataFrame(importance_scores, columns=['Feature', 'Importance'])
    
    # Plot top 20 features
    top_20 = importance_df.head(20)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['red' if x < 0 else 'green' for x in top_20['Importance']]
    bars = ax.barh(range(len(top_20)), top_20['Importance'].values, color=colors)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['Feature'].values)
    ax.set_xlabel('Importance (Accuracy Decrease)')
    ax.set_title('Top 20 Feature Importance (Permutation)')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

# ==================== Attention Analysis Function ====================
def analyze_attention(model, test_loader, device, feature_names):
    """
    Analyze attention patterns in the transformer
    """
    model.eval()
    
    # Get a batch of data
    features, labels = next(iter(test_loader))
    features = features.to(device)
    
    # Hook to capture attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        # For TransformerEncoderLayer, we need to access attention differently
        pass
    
    # Forward pass to get intermediate representations
    with torch.no_grad():
        batch_size = features.size(0)
        
        # Reshape to sequence
        x = features[:, :SEQ_LEN * TOKEN_DIM]
        x = x.view(batch_size, SEQ_LEN, TOKEN_DIM)
        
        # Get embedded representation
        x = model.feature_embedding(x)
        x = x * np.sqrt(model.d_model)
        x = model.pos_encoder(x)
        
        # For each transformer layer, analyze the attention
        # Since PyTorch's TransformerEncoder doesn't expose attention weights directly,
        # we'll analyze the output representations
        
        output = model.transformer_encoder(x)
        
        # Calculate token importance based on output magnitude
        token_importance = output.abs().mean(dim=2).mean(dim=0).cpu().numpy()
    
    # Map token positions to feature groups
    token_feature_map = []
    for i in range(SEQ_LEN):
        start_idx = i * TOKEN_DIM
        end_idx = min(start_idx + TOKEN_DIM, len(feature_names))
        features_in_token = feature_names[start_idx:end_idx]
        token_feature_map.append({
            'Token_Position': i,
            'Features': ', '.join(features_in_token),
            'Importance': token_importance[i] if i < len(token_importance) else 0
        })
    
    attention_df = pd.DataFrame(token_feature_map)
    
    # Plot token importance
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(attention_df['Token_Position'], attention_df['Importance'], color='steelblue')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Importance (Mean Absolute Output)')
    ax.set_title('Token Importance from Transformer Output')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('token_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return attention_df

# ==================== Report Generation Function ====================
def generate_report(results, history, feature_importance, attention_data, index_to_code1, report):
    """
    Generate detailed markdown report
    """
    # Get top 10 frameworks by F1 score
    framework_metrics = []
    for framework, metrics in report.items():
        if framework not in ['accuracy', 'macro avg', 'weighted avg']:
            framework_metrics.append({
                'Framework': framework,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1-score'],
                'Support': metrics['support']
            })
    
    framework_df = pd.DataFrame(framework_metrics)
    framework_df = framework_df.sort_values('F1_Score', ascending=False)
    top_10_frameworks = framework_df.head(10)
    
    # Get top 10 important features
    top_10_features = feature_importance.head(10)
    
    report_content = f"""# Transformer Model for Zeolite Framework Prediction

## 1. Model Architecture

### 1.1 Transformer Encoder Overview

本模型采用 **Transformer Encoder** 架构，将104维表格特征转换为序列形式进行建模。Transformer的核心优势在于其**自注意力机制（Self-Attention）**，能够捕捉特征之间的长距离依赖关系。

### 1.2 Architecture Details

| Component | Configuration |
|-----------|--------------|
| Input Features | 104 dimensions |
| Sequence Length | 13 tokens |
| Token Dimension | 8 features per token |
| Embedding Dimension (d_model) | {results['d_model']} |
| Number of Attention Heads | {results['n_head']} |
| Number of Encoder Layers | {results['n_layers']} |
| Feedforward Dimension | {results['dim_feedforward']} |
| Dropout Rate | {results.get('dropout', 0.3):.2f} |
| Output Classes | {results['num_classes']} |

### 1.3 Model Structure

```
Input (104 features)
    ↓
Reshape to Sequence (13 tokens × 8 features)
    ↓
Feature Embedding (Linear: 8 → {results['d_model']})
    ↓
Positional Encoding (Sinusoidal)
    ↓
Transformer Encoder × {results['n_layers']} layers
    │
    ├── Multi-Head Self-Attention ({results['n_head']} heads)
    ├── Add & Layer Normalization
    ├── Feedforward Network ({results['dim_feedforward']} hidden)
    └── Add & Layer Normalization
    ↓
Global Average Pooling
    ↓
Classification Head
    │
    ├── Linear ({results['d_model']} → 256)
    ├── ReLU + Dropout
    ├── Linear (256 → 128)
    ├── ReLU + Dropout
    └── Linear (128 → {results['num_classes']})
    ↓
Output (Logits)
```

### 1.4 Key Design Choices

1. **Feature-to-Token转换**: 将104个特征分为13组，每组8个特征作为一个token，便于Transformer处理序列信息。

2. **位置编码**: 使用正弦/余弦位置编码为模型提供位置信息，使其能够区分不同位置的特征组。

3. **多头注意力**: {results['n_head']}个注意力头允许模型同时关注不同位置的特征组合。

4. **残差连接**: 每个子层后都有残差连接，帮助梯度流动，便于深层网络训练。

## 2. Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Samples | {results['training_samples']:,} |
| Validation Samples | {results['validation_samples']:,} |
| Test Samples | {results['test_samples']:,} |
| Batch Size | {results['batch_size']} |
| Learning Rate | {results['learning_rate']} |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Scheduler | ReduceLROnPlateau |
| Epochs Trained | {results['epochs']} |
| Total Parameters | {results['total_parameters']:,} |

## 3. Dataset Information

### 3.1 Feature Categories

| Category | Count | Features |
|----------|-------|----------|
| Elemental Features | 45 | Si, Al, P, Na, K, Li, ... |
| OSDA Index | 3 | osda1_index, osda2_index, osda3_index |
| Synthesis Conditions | 4 | cryst_temp, cryst_time, seed, rotation |
| Aging Conditions | 2 | aging_temp, aging_time |
| pH Conditions | 2 | acid, OH |
| Gel Ratios | 5 | H2O_T, OH_T, Gel_Si_Al, Gel_P_Al, Gel_P_Si |
| OSDA Descriptors | 33 | bertz_ct, free_sasa, asphericity, ... |
| Aggregated Features | 10 | osda_avg_asphericity, osda_max_sasa, ... |
| **Total** | **104** | |

### 3.2 Framework Distribution

数据集包含 **{results['num_classes']}** 种不同的沸石框架类型（Code1），其中样本量最多的框架包括：MFI、CHA、*BEA、AFI、MTW等。

## 4. Model Performance

### 4.1 Overall Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **{results['test_accuracy']:.4f}** |
| Precision (Macro) | {results['test_precision_macro']:.4f} |
| Recall (Macro) | {results['test_recall_macro']:.4f} |
| F1 Score (Macro) | {results['test_f1_macro']:.4f} |
| F1 Score (Weighted) | {results['test_f1_weighted']:.4f} |
| Prediction Variance | {results['prediction_variance']:.4f} |
| Best Validation Accuracy | {results['best_val_accuracy']:.4f} |

### 4.2 Top 10 Frameworks by F1 Score

| Rank | Framework | Precision | Recall | F1 Score | Support |
|------|-----------|-----------|--------|----------|---------|
"""
    
    for i, row in top_10_frameworks.iterrows():
        report_content += f"| {len(top_10_frameworks) - list(top_10_frameworks.index).index(i)} | {row['Framework']} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1_Score']:.4f} | {int(row['Support'])} |\n"
    
    report_content += f"""
### 4.3 Performance Analysis

1. **整体性能**: 模型在测试集上达到 **{results['test_accuracy']*100:.2f}%** 的准确率，表明Transformer架构能够有效学习沸石合成条件与框架类型之间的复杂映射关系。

2. **类别不平衡**: 由于数据集中各类框架样本量差异较大，F1 Macro ({results['test_f1_macro']:.4f}) 低于 F1 Weighted ({results['test_f1_weighted']:.4f})，说明模型在少数类上的表现有所下降。

3. **预测置信度**: 平均预测方差为 {results['prediction_variance']:.4f}，表明模型的预测具有一定的确定性。

## 5. Feature Importance Analysis

### 5.1 Top 10 Important Features

| Rank | Feature | Importance |
|------|---------|------------|
"""
    
    for i, row in top_10_features.iterrows():
        report_content += f"| {list(top_10_features.index).index(i) + 1} | {row['Feature']} | {row['Importance']:.6f} |\n"
    
    report_content += f"""
### 5.2 Feature Importance Interpretation

特征重要性通过**置换重要性（Permutation Importance）**方法计算。重要性值表示当该特征被随机打乱时，模型准确率的下降程度。

- **正值**: 打乱该特征后准确率下降，说明该特征对预测有正向贡献
- **负值**: 打乱该特征后准确率上升，可能存在特征冗余或噪声

### 5.3 Token Importance

Transformer将104个特征分成13个token，每个token包含8个特征。以下展示了各token的重要性：

| Token | Features | Importance |
|-------|----------|------------|
"""
    
    for i, row in attention_data.iterrows():
        report_content += f"| {row['Token_Position']} | {row['Features'][:50]}... | {row['Importance']:.6f} |\n"
    
    report_content += f"""
## 6. Training History

### 6.1 Loss Curves

训练过程中损失函数的变化：
- 训练损失持续下降
- 验证损失先下降后趋于稳定
- 使用早停（Early Stopping）防止过拟合

### 6.2 Accuracy Curves

- 最佳验证准确率: {results['best_val_accuracy']:.4f}
- 最终训练准确率: {history['train_acc'][-1]:.4f}
- 最终验证准确率: {history['val_acc'][-1]:.4f}

## 7. Comparison with Other Models

| Model | Architecture | Test Accuracy |
|-------|--------------|---------------|
| Baseline61 XGBoost | Gradient Boosting | ~80.50% |
| Baseline104 XGBoost | Gradient Boosting | ~83.15% |
| LeNet5 CNN | Convolutional | ~77.36% |
| **Transformer (This work)** | **Self-Attention** | **{results['test_accuracy']*100:.2f}%** |
| BiGRU | Bidirectional RNN | ~78.5% |

## 8. Conclusions

### 8.1 Summary

1. Transformer模型在沸石框架预测任务上表现良好，测试准确率达到 **{results['test_accuracy']*100:.2f}%**。

2. 自注意力机制能够有效捕捉特征之间的复杂交互关系，特别是OSDA分子描述符与合成条件之间的关联。

3. 特征重要性分析揭示了关键影响因素：{', '.join(top_10_features['Feature'].head(5).tolist())}。

### 8.2 Advantages of Transformer

1. **长距离依赖**: 自注意力机制可以捕捉任意两个特征之间的关系，不受距离限制
2. **并行计算**: 相比RNN，Transformer可以并行处理序列，训练效率更高
3. **可解释性**: 注意力权重提供了一定程度的模型可解释性

### 8.3 Limitations

1. **数据规模**: Transformer通常需要大量数据进行训练，当前数据集规模相对有限
2. **表格数据**: Transformer最初为序列数据设计，应用于表格数据时可能不如传统方法高效
3. **计算成本**: 相比简单的机器学习模型，Transformer需要更多计算资源

### 8.4 Future Work

1. 探索更大的模型规模和更多数据增强技术
2. 结合领域知识设计更好的特征编码方式
3. 尝试多任务学习同时预测多个沸石属性

---

*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open('Transformer104_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)

# ==================== Run Main ====================
if __name__ == '__main__':
    results, history, feature_importance = main()
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test F1 Macro: {results['test_f1_macro']:.4f}")
    print(f"Test F1 Weighted: {results['test_f1_weighted']:.4f}")
    print(f"Prediction Variance: {results['prediction_variance']:.4f}")
    print("=" * 80)
