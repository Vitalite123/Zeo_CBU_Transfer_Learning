"""
LeNet-5 CNN Model for Zeolite Framework Prediction
Adapted for 104-dimensional tabular features
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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
LEARNING_RATE = 0.001
EPOCHS = 50
DROPOUT_RATE = 0.5

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

# ==================== LeNet-5 Model ====================
class LeNet5_Zeolite(nn.Module):
    """
    LeNet-5 CNN adapted for 1D tabular data
    Architecture:
    - Input: 104 features
    - Conv1: 64 filters, kernel=5, stride=1, padding=2
    - MaxPool1: kernel=2
    - Conv2: 128 filters, kernel=5, stride=1, padding=2
    - MaxPool2: kernel=2
    - Conv3: 256 filters, kernel=3, stride=1, padding=1
    - Fully Connected: 512 -> 256 -> num_classes
    """
    def __init__(self, num_classes):
        super(LeNet5_Zeolite, self).__init__()
        
        # Convolutional layers for 1D data
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Calculate output size after convolutions
        # Input: 104 -> after pool1: 52 -> after pool2: 26
        self.conv_output_size = 26 * 256
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, 104)
        # Reshape to (batch_size, 1, 104) for Conv1d
        x = x.unsqueeze(1)
        
        # Conv1 + Pool
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Conv2 + Pool
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Conv3
        x = self.conv3(x)
        x = self.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

# ==================== Training Function ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
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
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
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
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

# ==================== Main Function ====================
def main():
    print("=" * 80)
    print("LeNet-5 CNN for Zeolite Framework Prediction")
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
    print("Creating LeNet-5 model...")
    model = LeNet5_Zeolite(num_classes=num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("Training model...")
    print("-" * 80)
    history, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, EPOCHS
    )
    print("-" * 80)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    y_test_pred, y_test_true = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_true, y_test_pred)
    precision = precision_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    recall = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
    
    # Variance of predictions (measure of confidence)
    test_probs = []
    model.eval()
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            test_probs.extend(probs.cpu().numpy())
    test_probs = np.array(test_probs)
    prediction_variance = np.mean(np.var(test_probs, axis=1))
    
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
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'training_samples': int(len(train_df)),
        'validation_samples': int(len(val_df)),
        'test_samples': int(len(test_df))
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
    }, 'lenet5_104.pkl')
    
    # Save results as JSON
    with open('lenet5_104_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Model saved: lenet5_104.pkl")
    print(f"Results saved: lenet5_104_results.json")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    generate_visualizations(history, y_test_true, y_test_pred, label_encoder, index_to_code1)
    print("Visualizations saved!")
    print()
    
    # Generate feature importance (using permutation importance)
    print("Calculating feature importance...")
    feature_importance = calculate_feature_importance(
        model, test_loader, device, FEATURE_COLUMNS
    )
    
    # Save feature importance
    feature_importance.to_csv('feature_importance_104.csv', index=False)
    print(f"Feature importance saved: feature_importance_104.csv")
    print()
    
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return results, history, feature_importance

# ==================== Visualization Function ====================
def generate_visualizations(history, y_true, y_pred, label_encoder, index_to_code1):
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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], label='Train Loss', marker='o')
    ax.plot(history['val_loss'], label='Validation Loss', marker='s')
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
    ax.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax.plot(history['val_acc'], label='Validation Accuracy', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion matrix
    cm = confusion_matrix(y_true_names, y_pred_names, labels=unique_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=unique_classes, yticklabels=unique_classes, ax=ax)
    ax.set_xlabel('Predicted Framework')
    ax.set_ylabel('Actual Framework')
    ax.set_title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save confusion matrix data
    cm_df = pd.DataFrame(cm, index=unique_classes, columns=unique_classes)
    cm_df.to_csv('confusion_matrix.csv')
    
    # 4. Performance metrics by class
    from sklearn.metrics import classification_report
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
    
    # Plot top 20 frameworks by support
    top_20_classes = class_metrics_df.nlargest(20, 'Support')['Framework'].tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Precision
    ax = axes[0, 0]
    top_precision = class_metrics_df[class_metrics_df['Framework'].isin(top_20_classes)]
    ax.barh(top_precision['Framework'], top_precision['Precision'], color='steelblue')
    ax.set_xlabel('Precision')
    ax.set_title('Precision by Framework (Top 20 by Support)')
    ax.grid(axis='x', alpha=0.3)
    
    # Recall
    ax = axes[0, 1]
    top_recall = class_metrics_df[class_metrics_df['Framework'].isin(top_20_classes)]
    ax.barh(top_recall['Framework'], top_recall['Recall'], color='coral')
    ax.set_xlabel('Recall')
    ax.set_title('Recall by Framework (Top 20 by Support)')
    ax.grid(axis='x', alpha=0.3)
    
    # F1 Score
    ax = axes[1, 0]
    top_f1 = class_metrics_df[class_metrics_df['Framework'].isin(top_20_classes)]
    ax.barh(top_f1['Framework'], top_f1['F1_Score'], color='mediumseagreen')
    ax.set_xlabel('F1 Score')
    ax.set_title('F1 Score by Framework (Top 20 by Support)')
    ax.grid(axis='x', alpha=0.3)
    
    # Support
    ax = axes[1, 1]
    top_support = class_metrics_df[class_metrics_df['Framework'].isin(top_20_classes)]
    ax.barh(top_support['Framework'], top_support['Support'], color='mediumpurple')
    ax.set_xlabel('Support (Number of Samples)')
    ax.set_title('Sample Support by Framework (Top 20)')
    ax.grid(axis='x', alpha=0.3)
    
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
            np.mean([m['Precision'] for m in class_metrics]),
            np.mean([m['Recall'] for m in class_metrics]),
            np.mean([m['F1_Score'] for m in class_metrics]),
            report['weighted avg']['f1-score']
        ]
    }
    
    overall_df = pd.DataFrame(overall_metrics)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'gold']
    ax.bar(overall_df['Metric'], overall_df['Value'], color=colors)
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Metrics')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(overall_df['Value']):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save overall metrics
    overall_df.to_csv('overall_performance.csv', index=False)

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
    results, history, feature_importance = main()
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test F1 Macro: {results['test_f1_macro']:.4f}")
    print(f"Test F1 Weighted: {results['test_f1_weighted']:.4f}")
    print(f"Prediction Variance: {results['prediction_variance']:.4f}")
    print("=" * 80)