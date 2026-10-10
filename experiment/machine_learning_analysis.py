#!/usr/bin/env python3
"""
Machine Learning Analysis of Crystallization Dataset
This script performs machine learning analysis including classification, regression,
and predictive modeling for the crystallization dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import os
import warnings
warnings.filterwarnings('ignore')

# Ensure paths are relative to this script's directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.style.use('seaborn-v0_8-whitegrid')

class CrystallizationMLAnalysis:
    def __init__(self):
        self.chemical_data = None
        self.cv_data = None
        self.combined_data = None
        self.elements = ['Calcium', 'Carbon', 'Chlorine', 'Magnesium', 'Oxygen', 'Sodium', 'Sulphur']
        
    def load_data(self):
        """Load chemical and computer vision data"""
        print("Loading data for ML analysis...")
        
        # Load chemical composition data
        chemical_features = {}
        for element in self.elements:
            try:
                element_data = pd.read_csv(os.path.join(_SCRIPT_DIR, "recipies", f"{element}.csv"), header=None)
                # Flatten and create labels
                for table_idx in range(element_data.shape[0]):
                    for sample_idx in range(element_data.shape[1]):
                        sample_id = f"Table_{(table_idx//5)+1}_Sample_{table_idx%5*5+sample_idx+1}"
                        if sample_id not in chemical_features:
                            chemical_features[sample_id] = {}
                        chemical_features[sample_id][f'{element}_concentration'] = element_data.iloc[table_idx, sample_idx]
                        
                        # Add experimental conditions
                        table_num = (table_idx // 5) + 1
                        nahco3_levels = [0.25, 0.5, 0.75, 1.0, 2.0]
                        cacl2_levels = [0.5, 0.75, 1.0, 1.5, 2.0]
                        mgso4_levels = [0.25, 0.5, 0.75, 1.0, 2.0]
                        
                        chemical_features[sample_id]['NaHCO3_mM'] = nahco3_levels[table_num-1]
                        chemical_features[sample_id]['CaCl2_mM'] = cacl2_levels[sample_idx]
                        chemical_features[sample_id]['MgSO4_mM'] = mgso4_levels[table_idx % 5]
                        chemical_features[sample_id]['Table'] = table_num
                        
            except Exception as e:
                print(f"Error loading {element}: {e}")
        
        self.chemical_data = pd.DataFrame.from_dict(chemical_features, orient='index')
        print(f"Loaded chemical data: {self.chemical_data.shape}")
        
        # Try to load computer vision data if available
        try:
            self.cv_data = pd.read_csv(os.path.join(_SCRIPT_DIR, 'computer_vision_features.csv'))
            print(f"Loaded CV data: {self.cv_data.shape}")
        except FileNotFoundError:
            print("Computer vision data not available")
            
    def prepare_classification_data(self):
        """Prepare data for classification tasks"""
        if self.chemical_data is None:
            return None, None
        
        # Create classification targets based on experimental conditions
        y_table = self.chemical_data['Table'] - 1  # 0-4 for 5 tables
        
        # Create concentration categories
        ca_median = self.chemical_data['CaCl2_mM'].median()
        mg_median = self.chemical_data['MgSO4_mM'].median()
        
        # Binary classification: high vs low Ca concentration
        y_ca_class = (self.chemical_data['CaCl2_mM'] > ca_median).astype(int)
        
        # Multi-class: combination of Ca and Mg levels
        y_combo = ((self.chemical_data['CaCl2_mM'] > ca_median).astype(int) * 2 + 
                  (self.chemical_data['MgSO4_mM'] > mg_median).astype(int))
        
        # Features: element concentrations
        feature_cols = [col for col in self.chemical_data.columns if 'concentration' in col]
        X = self.chemical_data[feature_cols]
        
        return X, {'table': y_table, 'ca_class': y_ca_class, 'combo': y_combo}
    
    def prepare_regression_data(self):
        """Prepare data for regression tasks"""
        if self.chemical_data is None:
            return None, None
        
        # Regression targets: predict experimental conditions from concentrations
        feature_cols = [col for col in self.chemical_data.columns if 'concentration' in col]
        X = self.chemical_data[feature_cols]
        
        y_targets = {
            'CaCl2_mM': self.chemical_data['CaCl2_mM'],
            'MgSO4_mM': self.chemical_data['MgSO4_mM'],
            'NaHCO3_mM': self.chemical_data['NaHCO3_mM']
        }
        
        return X, y_targets
    
    def perform_classification_analysis(self):
        """Perform classification analysis"""
        print("\nPerforming classification analysis...")
        
        X, y_dict = self.prepare_classification_data()
        if X is None:
            return
        
        # Remove NaN values
        X_clean = X.fillna(X.mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        # Define classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        results = {}
        
        # Test each classification task
        for task_name, y in y_dict.items():
            print(f"\nClassification task: {task_name}")
            results[task_name] = {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Test each classifier
            for clf_name, clf in classifiers.items():
                try:
                    # Train and evaluate
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    score = clf.score(X_test, y_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
                    
                    results[task_name][clf_name] = {
                        'test_accuracy': score,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    print(f"{clf_name}: Accuracy = {score:.3f}, CV = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                    
                except Exception as e:
                    print(f"Error with {clf_name}: {e}")
                    results[task_name][clf_name] = {'test_accuracy': 0, 'cv_mean': 0, 'cv_std': 0}
        
        # Visualize results
        self.plot_classification_results(results)
        
        # Save results
        results_df = pd.DataFrame({
            (task, metric): {clf: results[task][clf][metric] for clf in results[task]}
            for task in results for metric in ['test_accuracy', 'cv_mean']
        }).T
        results_df.to_csv(os.path.join(_SCRIPT_DIR, 'classification_results.csv'))
        
        return results
    
    def perform_regression_analysis(self):
        """Perform regression analysis"""
        print("\nPerforming regression analysis...")
        
        X, y_dict = self.prepare_regression_data()
        if X is None:
            return
        
        # Remove NaN values
        X_clean = X.fillna(X.mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        # Define regressors
        regressors = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        results = {}
        
        # Test each regression task
        for task_name, y in y_dict.items():
            print(f"\nRegression task: {task_name}")
            results[task_name] = {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
            
            # Test each regressor
            for reg_name, reg in regressors.items():
                try:
                    # Train and evaluate
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    results[task_name][reg_name] = {
                        'r2_score': r2,
                        'rmse': rmse
                    }
                    
                    print(f"{reg_name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
                    
                except Exception as e:
                    print(f"Error with {reg_name}: {e}")
                    results[task_name][reg_name] = {'r2_score': 0, 'rmse': float('inf')}
        
        # Visualize results
        self.plot_regression_results(results)
        
        # Save results
        results_df = pd.DataFrame({
            (task, metric): {reg: results[task][reg][metric] for reg in results[task]}
            for task in results for metric in ['r2_score', 'rmse']
        }).T
        results_df.to_csv(os.path.join(_SCRIPT_DIR, 'regression_results.csv'))
        
        return results
    
    def plot_classification_results(self, results):
        """Plot classification results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data for plotting
        tasks = list(results.keys())
        classifiers = list(results[tasks[0]].keys())
        
        # Test accuracy comparison
        accuracy_data = []
        for task in tasks:
            for clf in classifiers:
                accuracy_data.append({
                    'Task': task,
                    'Classifier': clf,
                    'Accuracy': results[task][clf]['test_accuracy']
                })
        
        accuracy_df = pd.DataFrame(accuracy_data)
        
        # Plot accuracy comparison
        sns.barplot(data=accuracy_df, x='Task', y='Accuracy', hue='Classifier', ax=axes[0])
        axes[0].set_title('Classification Accuracy by Task')
        axes[0].set_ylabel('Test Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Cross-validation scores
        cv_data = []
        for task in tasks:
            for clf in classifiers:
                cv_data.append({
                    'Task': task,
                    'Classifier': clf,
                    'CV_Mean': results[task][clf]['cv_mean'],
                    'CV_Std': results[task][clf]['cv_std']
                })
        
        cv_df = pd.DataFrame(cv_data)
        
        # Plot CV scores
        sns.barplot(data=cv_df, x='Task', y='CV_Mean', hue='Classifier', ax=axes[1])
        axes[1].set_title('Cross-Validation Accuracy')
        axes[1].set_ylabel('CV Accuracy')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(_SCRIPT_DIR, 'classification_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_regression_results(self, results):
        """Plot regression results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data for plotting
        tasks = list(results.keys())
        regressors = list(results[tasks[0]].keys())
        
        # R² scores
        r2_data = []
        for task in tasks:
            for reg in regressors:
                r2_data.append({
                    'Task': task,
                    'Regressor': reg,
                    'R2_Score': results[task][reg]['r2_score']
                })
        
        r2_df = pd.DataFrame(r2_data)
        
        # Plot R² scores
        sns.barplot(data=r2_df, x='Task', y='R2_Score', hue='Regressor', ax=axes[0])
        axes[0].set_title('R² Scores by Task')
        axes[0].set_ylabel('R² Score')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # RMSE scores
        rmse_data = []
        for task in tasks:
            for reg in regressors:
                rmse_data.append({
                    'Task': task,
                    'Regressor': reg,
                    'RMSE': results[task][reg]['rmse']
                })
        
        rmse_df = pd.DataFrame(rmse_data)
        
        # Plot RMSE scores
        sns.barplot(data=rmse_df, x='Task', y='RMSE', hue='Regressor', ax=axes[1])
        axes[1].set_title('RMSE by Task')
        axes[1].set_ylabel('RMSE')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(_SCRIPT_DIR, 'regression_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance_analysis(self):
        """Analyze feature importance using Random Forest"""
        print("\nPerforming feature importance analysis...")
        
        X, y_dict = self.prepare_classification_data()
        if X is None:
            return
        
        X_clean = X.fillna(X.mean())
        
        # Use Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Analyze importance for table classification
        y_table = y_dict['table']
        rf.fit(X_clean, y_table)
        
        # Get feature importance
        importance = rf.feature_importances_
        features = X_clean.columns
        
        # Create feature importance plot
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')
        plt.title('Top 10 Feature Importance for Table Classification')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(_SCRIPT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save feature importance
        importance_df.to_csv(os.path.join(_SCRIPT_DIR, 'feature_importance.csv'), index=False)
        
        return importance_df
    
    def generate_ml_report(self):
        """Generate comprehensive ML analysis report"""
        print("COMPREHENSIVE MACHINE LEARNING ANALYSIS")
        print("="*50)
        
        # Load data
        self.load_data()
        
        # Perform analyses
        classification_results = self.perform_classification_analysis()
        regression_results = self.perform_regression_analysis()
        feature_importance = self.feature_importance_analysis()
        
        print("\nMachine learning analysis complete!")
        print("Generated files:")
        print("- classification_results.csv")
        print("- regression_results.csv")
        print("- feature_importance.csv")
        print("- classification_results.png")
        print("- regression_results.png")
        print("- feature_importance.png")

if __name__ == "__main__":
    ml_analyzer = CrystallizationMLAnalysis()
    ml_analyzer.generate_ml_report() 