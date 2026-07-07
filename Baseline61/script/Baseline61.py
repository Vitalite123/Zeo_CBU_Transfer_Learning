"""
Code1 Prediction Model - Predict zeolite framework structure using 61 features
Using three tree-based models: XGBoost, RandomForest, LightGBM

Optimized parameter configuration:
- XGBoost: n_estimators=300, max_depth=10, learning_rate=0.05
- RandomForest: n_estimators=400, max_depth=25, min_samples_split=2
- LightGBM: n_estimators=800, max_depth=15, learning_rate=0.05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Set up Chinese font support (kept for compatibility if needed)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Define list of 104 features (based on sections 1 and 2 in Feature_List.md)
FEATURES_104 = [
    # 1.1 Element composition (45 features)
    'Si', 'Al', 'P', 'Na', 'K', 'Li', 'Sr', 'Rb', 'Cs', 'Ba', 'Ca', 'F', 'Ge', 'Ti', 'In', 'B', 'Mg', 'Ga', 'Ni', 'Mn', 'Fe', 'Co', 'Cr', 'Zn', 'Nb', 'Be', 'W', 'Ce', 'Cu', 'Sn', 'Gd', 'La', 'Y', 'Dy', 'Sm', 'Ag', 'Cd', 'Zr', 'V', 'Ta', 'Ru', 'Hf', 'Yb', 'Tl', 'As',
    # 1.2 OSDA indices (3 features)
    'osda1_index', 'osda2_index', 'osda3_index',
    # 1.3 Synthesis conditions (4 features)
    'cryst_temp', 'cryst_time', 'seed', 'rotation',
    # 1.4 Aging conditions (2 features)
    'aging_temp', 'aging_time',
    # 1.5 pH conditions (2 features)
    'acid', 'OH',
    # 1.6 Gel ratio (5 features)
    'H2O_T', 'OH_T', 'Gel_Si_Al', 'Gel_P_Al', 'Gel_P_Si',
    # 2.1 OSDA molecular descriptors (33 features) - for each OSDA (osda1, osda2, osda3)
    'osda1_bertz_ct_mean_0', 'osda1_free_sasa_mean_0', 'osda1_asphericity_mean_0', 'osda1_eccentricity_mean_0',
    'osda1_axes_mean_0', 'osda1_axes_mean_1', 'osda1_box_mean_0', 'osda1_box_mean_1', 'osda1_box_mean_2',
    'osda1_getaway_mean_0', 'osda1_getaway_mean_1',
    'osda2_bertz_ct_mean_0', 'osda2_free_sasa_mean_0', 'osda2_asphericity_mean_0', 'osda2_eccentricity_mean_0',
    'osda2_axes_mean_0', 'osda2_axes_mean_1', 'osda2_box_mean_0', 'osda2_box_mean_1', 'osda2_box_mean_2',
    'osda2_getaway_mean_0', 'osda2_getaway_mean_1',
    'osda3_bertz_ct_mean_0', 'osda3_free_sasa_mean_0', 'osda3_asphericity_mean_0', 'osda3_eccentricity_mean_0',
    'osda3_axes_mean_0', 'osda3_axes_mean_1', 'osda3_box_mean_0', 'osda3_box_mean_1', 'osda3_box_mean_2',
    'osda3_getaway_mean_0', 'osda3_getaway_mean_1',
    # 2.2 Aggregated features (10 features)
    'osda_avg_asphericity', 'osda_max_asphericity', 'osda_min_asphericity',
    'osda_avg_sasa', 'osda_max_sasa', 'osda_min_sasa',
    'osda_avg_bertz', 'osda_max_bertz', 'osda_min_bertz', 'osda_total_volume'
]

print(f"Total defined features: {len(FEATURES_104)}")


class Code1Predictor:
    """Code1 Prediction Model Class"""
    
    def __init__(self):
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.feature_names = FEATURES_104
        
    def load_data(self, train_path, val_path, test_path):
        """Load training, validation and test data"""
        print("=" * 60)
        print("Loading data...")
        print("=" * 60)
        
        # Read data
        train_df = pd.read_excel(train_path)
        val_df = pd.read_excel(val_path)
        test_df = pd.read_excel(test_path)
        
        print(f"Training set size: {train_df.shape}")
        print(f"Validation set size: {val_df.shape}")
        print(f"Test set size: {test_df.shape}")
        
        # Check if Code1_index column exists
        if 'Code1_index' not in train_df.columns:
            print("Warning: Code1_index column does not exist, trying to use Code1 column")
            if 'Code1' in train_df.columns:
                train_df['Code1_index'] = train_df['Code1']
                val_df['Code1_index'] = val_df['Code1']
                test_df['Code1_index'] = test_df['Code1']
            else:
                raise ValueError("Code1_index or Code1 column not found")
        
        # Get available features
        available_features = [f for f in self.feature_names if f in train_df.columns]
        print(f"\nOut of 104 features, {len(available_features)} features are available in the data")
        
        if len(available_features) < len(self.feature_names):
            missing_features = set(self.feature_names) - set(available_features)
            print(f"Missing features: {missing_features}")
            print(f"Will train with {len(available_features)} available features")
        
        self.feature_names = available_features
        
        # Prepare data
        X_train = train_df[self.feature_names]
        y_train = train_df['Code1_index']
        X_val = val_df[self.feature_names]
        y_val = val_df['Code1_index']
        X_test = test_df[self.feature_names]
        y_test = test_df['Code1_index']
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Handle missing values
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        X_test = X_test.fillna(0)
        
        print(f"\nNumber of classes: {len(self.label_encoder.classes_)}")
        print(f"Class distribution (Training set):")
        class_counts = pd.Series(y_train_encoded).value_counts()
        for idx, count in class_counts.head(10).items():
            print(f"  Class {idx} ({self.label_encoder.classes_[idx]}): {count} samples")
        
        return (X_train, y_train_encoded), (X_val, y_val_encoded), (X_test, y_test_encoded)
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("\n" + "=" * 60)
        print("Training XGBoost model...")
        print("=" * 60)
        
        model = xgb.XGBClassifier(
            n_estimators=300,           # Increase number of trees
            max_depth=10,               # Increase max depth
            learning_rate=0.05,         # Decrease learning rate
            subsample=0.9,              # Increase sample sampling ratio
            colsample_bytree=0.9,       # Increase feature sampling ratio
            min_child_weight=3,         # Increase min child weight
            gamma=0.1,                  # Increase min split gain
            reg_alpha=0.1,              # L1 regularization
            reg_lambda=1.0,             # L2 regularization
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.models['XGBoost'] = model
        print("XGBoost model training completed")
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train RandomForest model"""
        print("\n" + "=" * 60)
        print("Training RandomForest model...")
        print("=" * 60)
        
        model = RandomForestClassifier(
            n_estimators=400,          # Increase number of trees
            max_depth=25,              # Increase max depth
            min_samples_split=2,       # Decrease min samples for split
            min_samples_leaf=1,        # Decrease min samples for leaf node
            max_features='sqrt',       # Use sqrt number of features
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        self.models['RandomForest'] = model
        print("RandomForest model training completed")
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        print("\n" + "=" * 60)
        print("Training LightGBM model...")
        print("=" * 60)
        
        model = lgb.LGBMClassifier(
            n_estimators=800,           # Significantly increase number of trees
            max_depth=15,               # Increase max depth
            learning_rate=0.05,         # Decrease learning rate, need more trees
            num_leaves=64,              # Increase number of leaf nodes
            subsample=0.9,              # Increase sample sampling ratio
            colsample_bytree=0.9,       # Increase feature sampling ratio
            min_child_samples=10,       # Decrease min child samples
            reg_alpha=0.1,              # L1 regularization
            reg_lambda=0.1,             # L2 regularization
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]  # Increase early stopping rounds
        )
        
        self.models['LightGBM'] = model
        print("LightGBM model training completed")
        return model
    
    def evaluate_model(self, model, X, y, dataset_name):
        """Evaluate model performance"""
        y_pred = model.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        f1_weighted = f1_score(y, y_pred, average='weighted')
        f1_macro = f1_score(y, y_pred, average='macro')
        
        # Calculate variance (prediction uncertainty)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            variance = np.var(proba, axis=1).mean()
        else:
            variance = 0.0
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_score': f1_weighted,  # Keep backward compatibility
            'variance': variance,
            'predictions': y_pred
        }
    
    def generate_report(self, train_data, val_data, test_data):
        """Generate detailed report"""
        print("\n" + "=" * 60)
        print("Generating detailed report...")
        print("=" * 60)
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        results = {}
        
        # Evaluate all models
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name} model...")
            
            # Training set
            train_results = self.evaluate_model(model, X_train, y_train, "Training set")
            
            # Validation set
            val_results = self.evaluate_model(model, X_val, y_val, "Validation set")
            
            # Test set
            test_results = self.evaluate_model(model, X_test, y_test, "Test set")
            
            results[model_name] = {
                'train': train_results,
                'val': val_results,
                'test': test_results
            }
        
        # Print results table
        print("\n" + "=" * 60)
        print("Model Performance Summary")
        print("=" * 60)
        
        print("\n" + "-" * 80)
        print(f"{'Model':<15} {'Dataset':<10} {'Accuracy':<12} {'F1 Weighted':<14} {'F1 Macro':<12} {'Variance':<12}")
        print("-" * 80)
        
        for model_name, model_results in results.items():
            for dataset, metrics in model_results.items():
                print(f"{model_name:<15} {dataset:<10} {metrics['accuracy']:<12.4f} {metrics['f1_weighted']:<14.4f} {metrics['f1_macro']:<12.4f} {metrics['variance']:<12.4f}")
        
        print("-" * 80)
        
        return results
    
    def plot_confusion_matrices(self, X_val, y_val, X_test, y_test, save_path='confusion_matrices.png'):
        """Plot confusion matrices"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Confusion Matrices - Validation and Test Sets', fontsize=16, fontweight='bold')
        
        datasets = [
            ('Validation Set', X_val, y_val),
            ('Test Set', X_test, y_test)
        ]
        
        for row, (dataset_name, X, y) in enumerate(datasets):
            for col, (model_name, model) in enumerate(self.models.items()):
                ax = axes[row, col]
                
                y_pred = model.predict(X)
                cm = confusion_matrix(y, y_pred)
                
                # Only show confusion matrix for first 20 classes
                n_classes = min(20, len(cm))
                cm_display = cm[:n_classes, :n_classes]
                
                sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                ax.set_title(f'{model_name} - {dataset_name}', fontweight='bold')
                ax.set_xlabel('Predicted Class')
                ax.set_ylabel('True Class')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrices saved to: {save_path}")
        plt.close()
    
    def plot_feature_importance(self, save_path='feature_importance.png'):
        """Plot feature importance"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        fig.suptitle('Feature Importance - Comparison of Three Models', fontsize=16, fontweight='bold')
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            ax = axes[idx]
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                importance = np.zeros(len(self.feature_names))
            
            # Get top 20 most important features
            indices = np.argsort(importance)[-20:]
            top_features = [self.feature_names[i] for i in indices]
            top_importance = importance[indices]
            
            # Horizontal bar chart
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_importance, color=plt.cm.Set3(idx))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features, fontsize=8)
            ax.set_xlabel('Importance', fontweight='bold')
            ax.set_title(model_name, fontweight='bold')
            ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
        plt.close()
    
    def plot_performance_comparison(self, results, save_path='performance_comparison.png'):
        """Plot performance comparison"""
        models = list(results.keys())
        metrics = ['accuracy', 'f1_weighted', 'f1_macro']
        datasets = ['train', 'val', 'test']
        dataset_names = {'train': 'Training Set', 'val': 'Validation Set', 'test': 'Test Set'}
        metric_labels = {'accuracy': 'Accuracy', 'f1_weighted': 'F1 Weighted', 'f1_macro': 'F1 Macro'}
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            x = np.arange(len(models))
            width = 0.25
            
            for i, dataset in enumerate(datasets):
                values = [results[model][dataset][metric] for model in models]
                offset = (i - 1) * width
                ax.bar(x + offset, values, width, label=dataset_names[dataset], 
                      color=plt.cm.Set1(i))
            
            ax.set_xlabel('Model', fontweight='bold')
            ax.set_ylabel(metric_labels[metric], fontweight='bold')
            ax.set_title(f'{metric_labels[metric]} Comparison', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison plot saved to: {save_path}")
        plt.close()
    
    def save_results_table(self, results, save_path='results_table.csv'):
        """Save results table"""
        data = []
        
        for model_name, model_results in results.items():
            for dataset, metrics in model_results.items():
                data.append({
                    'Model': model_name,
                    'Dataset': dataset,
                    'Accuracy': metrics['accuracy'],
                    'F1_Weighted': metrics['f1_weighted'],
                    'F1_Macro': metrics['f1_macro'],
                    'F1_Score': metrics['f1_score'],  # Keep backward compatibility
                    'Variance': metrics['variance']
                })
        
        df = pd.DataFrame(data)
        try:
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"Results table saved to: {save_path}")
        except PermissionError:
            # If file is occupied, try saving to temporary filename
            temp_path = save_path.replace('.csv', '_new.csv')
            df.to_csv(temp_path, index=False, encoding='utf-8-sig')
            print(f"Results table saved to: {temp_path} (original file occupied)")
        return df
    
    def save_models(self, save_path='baseline61.pkl'):
        """Save all trained models"""
        import joblib
        
        model_data = {
            'models': self.models,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, save_path)
        print(f"\nAll models saved to: {save_path}")
        return save_path


def main():
    """Main function"""
    print("=" * 60)
    print("Code1 Prediction Model - Zeolite Framework Structure Prediction")
    print("Using 104 features + three tree-based models")
    print("=" * 60)
    
    # Initialize predictor
    predictor = Code1Predictor()
    
    # Load data
    train_data, val_data, test_data = predictor.load_data(
        '../data_processed/ZEOSYN_train.xlsx',
        '../data_processed/ZEOSYN_val.xlsx',
        '../data_processed/ZEOSYN_test.xlsx'
    )
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Train three models
    predictor.train_xgboost(X_train, y_train, X_val, y_val)
    predictor.train_random_forest(X_train, y_train)
    predictor.train_lightgbm(X_train, y_train, X_val, y_val)
    
    # Generate report
    results = predictor.generate_report(train_data, val_data, test_data)
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating visualization charts...")
    print("=" * 60)
    
    predictor.plot_confusion_matrices(X_val, y_val, X_test, y_test)
    predictor.plot_feature_importance()
    predictor.plot_performance_comparison(results)
    
    # Save results table
    results_df = predictor.save_results_table(results)
    
    # Save models
    predictor.save_models('baseline61.pkl')
    
    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  1. baseline61.pkl - Trained models")
    print("  2. confusion_matrices.png - Confusion matrices")
    print("  3. feature_importance.png - Feature importance")
    print("  4. performance_comparison.png - Performance comparison")
    print("  5. results_table.csv - Results data table")
    
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()