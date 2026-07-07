"""
Bidirectional GRU Model for Zeolite Framework Prediction
Using 104-dimensional tabular features
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, top_k_accuracy_score
)
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
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 30
DROPOUT_RATE = 0.3
HIDDEN_SIZE = 128
NUM_LAYERS = 2

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


# ==================== Bidirectional GRU Model ====================
class BiGRU_Zeolite(nn.Module):
    """
    Bidirectional GRU for tabular data classification
    
    Architecture:
    - Input: 104 features → reshape to sequence (seq_len, input_size)
    - BiGRU layers: 2 layers with 256 hidden units each direction
    - Attention mechanism for sequence aggregation
    - Fully Connected: 512 -> 256 -> num_classes
    
    The model treats features as a sequence, allowing the GRU to
    capture dependencies between different feature groups.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(BiGRU_Zeolite, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Feature embedding layer
        # Reshape 104 features into sequence: 26 timesteps × 4 features
        self.seq_len = 26
        self.feature_per_step = 4
        
        self.feature_embedding = nn.Linear(self.feature_per_step, hidden_size // 2)
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(256, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, 104)
        batch_size = x.size(0)
        
        # Reshape to sequence: (batch_size, 26, 4)
        x = x.view(batch_size, self.seq_len, self.feature_per_step)
        
        # Feature embedding
        x = self.feature_embedding(x)  # (batch_size, 26, hidden_size//2)
        
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (batch_size, 26, hidden_size*2)
        
        # Attention mechanism
        attn_weights = self.attention(gru_out)  # (batch_size, 26, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(attn_weights * gru_out, dim=1)  # (batch_size, hidden_size*2)
        
        # Fully connected layers
        out = self.fc1(context)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        
        return out
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability"""
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.feature_per_step)
        x = self.feature_embedding(x)
        gru_out, _ = self.gru(x)
        attn_weights = self.attention(gru_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        return attn_weights.squeeze(-1)  # (batch_size, 26)


# ==================== Training Function ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    """
    Train the model with early stopping
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
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return history, best_val_acc


# ==================== Evaluation Function ====================
def evaluate_model(model, data_loader, device, num_classes):
    """
    Evaluate model and return predictions, probabilities and metrics
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
    print("Bidirectional GRU for Zeolite Framework Prediction")
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
    
    # Encode labels
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
    print("Creating BiGRU model...")
    model = BiGRU_Zeolite(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT_RATE
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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
    y_test_pred, y_test_true, test_probs = evaluate_model(model, test_loader, device, num_classes)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_true, y_test_pred)
    precision = precision_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    recall = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
    
    # Top-K accuracy
    top3_acc = top_k_accuracy_score(y_test_true, test_probs, k=3, labels=range(num_classes))
    top5_acc = top_k_accuracy_score(y_test_true, test_probs, k=5, labels=range(num_classes))
    top10_acc = top_k_accuracy_score(y_test_true, test_probs, k=10, labels=range(num_classes))
    
    # Prediction variance (measure of confidence)
    prediction_variance = np.mean(np.var(test_probs, axis=1))
    
    # Entropy of predictions
    prediction_entropy = -np.mean(np.sum(test_probs * np.log(test_probs + 1e-10), axis=1))
    
    print("\n" + "=" * 80)
    print("Test Set Results")
    print("=" * 80)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro):    {recall:.4f}")
    print(f"F1 Score (Macro):  {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"Top-3 Accuracy:    {top3_acc:.4f}")
    print(f"Top-5 Accuracy:    {top5_acc:.4f}")
    print(f"Top-10 Accuracy:   {top10_acc:.4f}")
    print(f"Prediction Variance: {prediction_variance:.4f}")
    print(f"Prediction Entropy: {prediction_entropy:.4f}")
    print()
    
    # Save results
    results = {
        'test_accuracy': float(accuracy),
        'test_precision_macro': float(precision),
        'test_recall_macro': float(recall),
        'test_f1_macro': float(f1_macro),
        'test_f1_weighted': float(f1_weighted),
        'test_top3_accuracy': float(top3_acc),
        'test_top5_accuracy': float(top5_acc),
        'test_top10_accuracy': float(top10_acc),
        'prediction_variance': float(prediction_variance),
        'prediction_entropy': float(prediction_entropy),
        'best_val_accuracy': float(best_val_acc),
        'num_classes': int(num_classes),
        'num_features': len(FEATURE_COLUMNS),
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'epochs': len(history['train_loss']),
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'dropout_rate': DROPOUT_RATE,
        'training_samples': int(len(train_df)),
        'validation_samples': int(len(val_df)),
        'test_samples': int(len(test_df)),
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params)
    }
    
    # Save model
    print("Saving model and results...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoder': label_encoder,
        'index_to_code1': index_to_code1,
        'feature_columns': FEATURE_COLUMNS,
        'results': results,
        'history': history
    }, 'bigru_104.pkl')
    
    # Save results as JSON
    with open('bigru_104_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Model saved: bigru_104.pkl")
    print(f"Results saved: bigru_104_results.json")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    generate_visualizations(history, y_test_true, y_test_pred, test_probs, 
                           label_encoder, index_to_code1, num_classes)
    print("Visualizations saved!")
    print()
    
    # Calculate attention weights for interpretability
    print("Calculating attention weights...")
    attention_weights = get_attention_analysis(model, test_loader, device)
    save_attention_analysis(attention_weights)
    print("Attention analysis saved!")
    print()
    
    # Generate feature importance
    print("Calculating feature importance...")
    feature_importance = calculate_feature_importance(
        model, test_loader, device, FEATURE_COLUMNS
    )
    feature_importance.to_csv('feature_importance.csv', index=False)
    print(f"Feature importance saved: feature_importance.csv")
    print()
    
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return results, history, feature_importance, attention_weights


# ==================== Visualization Functions ====================
def generate_visualizations(history, y_true, y_pred, y_probs, label_encoder, index_to_code1, num_classes):
    """
    Generate all visualization charts with data tables
    """
    # Convert numeric labels back to original Code1 indices
    y_true_original = label_encoder.inverse_transform(y_true)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    
    # Convert to Code1 names
    y_true_names = [index_to_code1.get(idx, str(idx)) for idx in y_true_original]
    y_pred_names = [index_to_code1.get(idx, str(idx)) for idx in y_pred_original]
    
    # Get unique class names
    unique_classes = sorted(set(y_true_names))
    
    # 1. Training and validation loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax.plot(history['val_loss'], label='Validation Loss', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[1]
    ax.plot(history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
    ax.plot(history['val_acc'], label='Validation Accuracy', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
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
    
    # 2. Confusion matrix (top 30 classes)
    # Get top 30 classes by support
    class_counts = pd.Series(y_true_names).value_counts()
    top_30_classes = class_counts.head(30).index.tolist()
    
    # Filter predictions
    mask_true = [c in top_30_classes for c in y_true_names]
    mask_pred = [c in top_30_classes for c in y_pred_names]
    
    cm = confusion_matrix(y_true_names, y_pred_names, labels=top_30_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=top_30_classes, yticklabels=top_30_classes, ax=ax)
    ax.set_xlabel('Predicted Framework')
    ax.set_ylabel('Actual Framework')
    ax.set_title('Confusion Matrix (Normalized, Top 30 Classes)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save confusion matrix data
    cm_df = pd.DataFrame(cm, index=top_30_classes, columns=top_30_classes)
    cm_df.to_csv('confusion_matrix.csv')
    
    # 3. Performance metrics by class
    report = classification_report(y_true_names, y_pred_names, 
                                   labels=unique_classes, output_dict=True, zero_division=0)
    
    # Extract metrics for each class
    class_metrics = []
    for class_name in unique_classes:
        if class_name in report:
            metrics = report[class_name]
            class_metrics.append({
                'Framework': class_name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1-score'],
                'Support': metrics['support']
            })
    
    class_metrics_df = pd.DataFrame(class_metrics)
    class_metrics_df.to_csv('performance_by_class.csv', index=False)
    
    # Plot top 20 frameworks by support
    top_20_classes = class_metrics_df.nlargest(20, 'Support')['Framework'].tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Precision
    ax = axes[0, 0]
    top_precision = class_metrics_df[class_metrics_df['Framework'].isin(top_20_classes)]
    top_precision = top_precision.sort_values('Precision', ascending=True)
    ax.barh(top_precision['Framework'], top_precision['Precision'], color='steelblue')
    ax.set_xlabel('Precision')
    ax.set_title('Precision by Framework (Top 20 by Support)')
    ax.grid(axis='x', alpha=0.3)
    
    # Recall
    ax = axes[0, 1]
    top_recall = class_metrics_df[class_metrics_df['Framework'].isin(top_20_classes)]
    top_recall = top_recall.sort_values('Recall', ascending=True)
    ax.barh(top_recall['Framework'], top_recall['Recall'], color='coral')
    ax.set_xlabel('Recall')
    ax.set_title('Recall by Framework (Top 20 by Support)')
    ax.grid(axis='x', alpha=0.3)
    
    # F1 Score
    ax = axes[1, 0]
    top_f1 = class_metrics_df[class_metrics_df['Framework'].isin(top_20_classes)]
    top_f1 = top_f1.sort_values('F1_Score', ascending=True)
    ax.barh(top_f1['Framework'], top_f1['F1_Score'], color='mediumseagreen')
    ax.set_xlabel('F1 Score')
    ax.set_title('F1 Score by Framework (Top 20 by Support)')
    ax.grid(axis='x', alpha=0.3)
    
    # Support
    ax = axes[1, 1]
    top_support = class_metrics_df[class_metrics_df['Framework'].isin(top_20_classes)]
    top_support = top_support.sort_values('Support', ascending=True)
    ax.barh(top_support['Framework'], top_support['Support'], color='mediumpurple')
    ax.set_xlabel('Support (Number of Samples)')
    ax.set_title('Sample Support by Framework (Top 20)')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Overall performance comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate accuracy manually
    test_accuracy = accuracy_score(y_true, y_pred)
    
    metrics_names = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'Top-3 Acc', 'Top-5 Acc', 'Top-10 Acc']
    metrics_values = [
        test_accuracy,
        report['macro avg']['f1-score'],
        report['weighted avg']['f1-score'],
        top_k_accuracy_score(y_true, y_probs, k=3, labels=range(num_classes)),
        top_k_accuracy_score(y_true, y_probs, k=5, labels=range(num_classes)),
        top_k_accuracy_score(y_true, y_probs, k=10, labels=range(num_classes))
    ]
    
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'gold', 'pink']
    bars = ax.bar(metrics_names, metrics_values, color=colors)
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Metrics')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, v in zip(bars, metrics_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save overall metrics
    overall_df = pd.DataFrame({'Metric': metrics_names, 'Value': metrics_values})
    overall_df.to_csv('overall_performance.csv', index=False)
    
    # 5. Prediction confidence distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Max probability distribution
    ax = axes[0]
    max_probs = np.max(y_probs, axis=1)
    ax.hist(max_probs, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Maximum Prediction Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Prediction Confidence')
    ax.axvline(x=np.mean(max_probs), color='red', linestyle='--', label=f'Mean: {np.mean(max_probs):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Entropy distribution
    ax = axes[1]
    entropy = -np.sum(y_probs * np.log(y_probs + 1e-10), axis=1)
    ax.hist(entropy, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.set_xlabel('Prediction Entropy')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Prediction Entropy')
    ax.axvline(x=np.mean(entropy), color='red', linestyle='--', label=f'Mean: {np.mean(entropy):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save confidence data
    confidence_df = pd.DataFrame({
        'Sample_Index': range(len(max_probs)),
        'Max_Probability': max_probs,
        'Entropy': entropy,
        'Correct': y_true == y_pred
    })
    confidence_df.to_csv('prediction_confidence.csv', index=False)


def get_attention_analysis(model, test_loader, device):
    """
    Get attention weights for interpretability
    """
    model.eval()
    all_attention = []
    
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)
            attn_weights = model.get_attention_weights(features)
            all_attention.append(attn_weights.cpu().numpy())
    
    all_attention = np.concatenate(all_attention, axis=0)
    
    # Mean attention across all samples
    mean_attention = np.mean(all_attention, axis=0)
    
    # Map attention to feature groups
    # Each position corresponds to 4 features
    feature_groups = [
        'Elements (Si-Al)', 'Elements (P-Na)', 'Elements (K-Li)', 'Elements (Sr-Rb)',
        'Elements (Cs-Ba)', 'Elements (Ca-F)', 'Elements (Ge-Ti)', 'Elements (In-B)',
        'Elements (Mg-Ga)', 'Elements (Ni-Mn)', 'Elements (Fe-Co)', 'Elements (Cr-Zn)',
        'Elements (Nb-Be)', 'Elements (W-Ce)', 'Elements (Cu-Sn)', 'Elements (Gd-La)',
        'Elements (Y-Dy)', 'Elements (Sm-Ag)', 'Elements (Cd-Zr)', 'Elements (V-Ta)',
        'Elements (Ru-Hf)', 'Elements (Yb-Tl)', 'Elements (As)', 'OSDA_Indices',
        'Synthesis_Conditions', 'Aging_PH_Ratios'
    ]
    
    return {
        'mean_attention': mean_attention,
        'attention_by_sample': all_attention,
        'feature_groups': feature_groups[:len(mean_attention)]
    }


def save_attention_analysis(attention_data):
    """
    Save attention analysis results
    """
    # Save mean attention
    attn_df = pd.DataFrame({
        'Position': range(len(attention_data['mean_attention'])),
        'Feature_Group': attention_data['feature_groups'],
        'Mean_Attention': attention_data['mean_attention']
    })
    attn_df.to_csv('attention_weights.csv', index=False)
    
    # Plot attention weights
    fig, ax = plt.subplots(figsize=(14, 6))
    positions = range(len(attention_data['mean_attention']))
    ax.bar(positions, attention_data['mean_attention'], color='steelblue')
    ax.set_xlabel('Feature Position')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Mean Attention Weights Across Feature Positions')
    ax.set_xticks(positions)
    ax.set_xticklabels(attention_data['feature_groups'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('attention_weights.png', dpi=300, bbox_inches='tight')
    plt.close()


def calculate_feature_importance(model, test_loader, device, feature_names):
    """
    Calculate feature importance using permutation importance
    Optimized version - uses subset of test data for faster computation
    """
    model.eval()
    
    # Use a subset for faster computation
    max_samples = min(1000, len(test_loader.dataset))
    
    # Get baseline accuracy
    baseline_correct = 0
    total = 0
    sample_count = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            if sample_count >= max_samples:
                break
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            baseline_correct += (predicted == labels).sum().item()
            total += labels.size(0)
            sample_count += labels.size(0)
    
    baseline_acc = baseline_correct / total
    
    # Calculate importance for each feature
    importance_scores = []
    
    for i, feature_name in enumerate(feature_names):
        # Shuffle this feature
        correct = 0
        total_samples = 0
        sample_count = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                if sample_count >= max_samples:
                    break
                    
                features_shuffled = features.clone()
                # Shuffle the i-th feature across the batch
                indices = torch.randperm(features.size(0))
                features_shuffled[:, i] = features[indices, i]
                
                features_shuffled, labels = features_shuffled.to(device), labels.to(device)
                outputs = model(features_shuffled)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                sample_count += labels.size(0)
        
        acc = correct / total_samples
        importance = baseline_acc - acc
        importance_scores.append((feature_name, importance))
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Feature importance: {i+1}/{len(feature_names)}")
    
    # Sort by importance
    importance_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Create DataFrame
    importance_df = pd.DataFrame(importance_scores, columns=['Feature', 'Importance'])
    
    # Plot top 20 features
    top_20 = importance_df.head(20)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['red' if x < 0 else 'green' for x in top_20['Importance']]
    ax.barh(top_20['Feature'], top_20['Importance'], color=colors)
    ax.set_xlabel('Importance (Accuracy Decrease)')
    ax.set_title('Top 20 Feature Importance (Permutation)')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df


# ==================== Run Main ====================
if __name__ == '__main__':
    results, history, feature_importance, attention_weights = main()
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test F1 Macro: {results['test_f1_macro']:.4f}")
    print(f"Test F1 Weighted: {results['test_f1_weighted']:.4f}")
    print(f"Top-5 Accuracy: {results['test_top5_accuracy']:.4f}")
    print(f"Top-10 Accuracy: {results['test_top10_accuracy']:.4f}")
    print("=" * 80)
