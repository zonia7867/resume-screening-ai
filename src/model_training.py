"""
Model Training Module for Resume Screening
This module handles training classification models for resume categorization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import pickle
import os
from datetime import datetime


class ResumeClassifier:
   
    
    def __init__(self, model_type='random_forest'):
        
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.label_encoder = None
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the classification model"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='linear',
                C=1.0,
                probability=True,
                random_state=42
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
       
        print(f"Training {self.model_type} model...")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print(f"✅ Model trained successfully!")
    
    def predict(self, X_test):
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_test)
        else:
            raise ValueError(f"{self.model_type} does not support probability predictions")
    
    def evaluate(self, X_test, y_test, label_encoder=None):
      
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Get class names if label encoder provided
        target_names = None
        if label_encoder is not None:
            target_names = label_encoder.classes_
        
        # Classification report
        print("\n" + "="*80)
        print(f"MODEL EVALUATION - {self.model_type.upper()}")
        print("="*80)
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
            
        Returns:
            dict: Cross-validation scores
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid=None):
   
        if param_grid is None:
            # Default parameter grids for different models
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'naive_bayes':
                param_grid = {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf']
                }
            elif self.model_type == 'logistic':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2']
                }
        
        print(f"\nPerforming hyperparameter tuning for {self.model_type}...")
        print(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def get_feature_importance(self, feature_names, top_n=20):
        
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        if self.model_type != 'random_forest':
            print(f"Feature importance not available for {self.model_type}")
            return None
        
        # Get feature importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        })
        
        return importance_df
    
    def save_model(self, filepath):
       
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
       
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"✅ Model loaded from {filepath}")
        print(f"   Model type: {self.model_type}")
        print(f"   Trained on: {model_data.get('timestamp', 'Unknown')}")


def train_and_evaluate_models(X_train, X_test, y_train, y_test, 
                               feature_names, label_encoder):
   
    models = ['random_forest', 'naive_bayes', 'logistic']
    results = {}
    
    print("\n" + "="*80)
    print("TRAINING AND EVALUATING MULTIPLE MODELS")
    print("="*80)
    
    for model_type in models:
        print(f"\n{'='*80}")
        print(f"Training {model_type.upper()} model...")
        print(f"{'='*80}")
        
        # Initialize and train
        classifier = ResumeClassifier(model_type=model_type)
        classifier.train(X_train, y_train)
        
        # Evaluate
        metrics = classifier.evaluate(X_test, y_test, label_encoder)
        
        # Save results
        results[model_type] = {
            'classifier': classifier,
            'metrics': metrics
        }
        
        # Save model
        model_path = f'models/{model_type}_model.pkl'
        classifier.save_model(model_path)
    
    # Compare models
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        model: {
            'Accuracy': results[model]['metrics']['accuracy'],
            'Precision': results[model]['metrics']['precision'],
            'Recall': results[model]['metrics']['recall'],
            'F1-Score': results[model]['metrics']['f1_score']
        }
        for model in results
    }).T
    
    print(comparison_df)
    
    # Find best model
    best_model = comparison_df['Accuracy'].idxmax()
    print(f"\n🏆 Best Model: {best_model.upper()}")
    print(f"   Accuracy: {comparison_df.loc[best_model, 'Accuracy']:.4f}")
    
    return results


if __name__ == "__main__":
   
    
    # This would typically be called after feature extraction
    print("Model Training Module")
    print("="*80)
    print("\nThis module provides classification models for resume categorization.")
    print("\nUsage:")
    print("1. Extract features using FeatureExtractor")
    print("2. Split data into train/test sets")
    print("3. Train models using ResumeClassifier")
    print("4. Evaluate and save the best model")
    print("\nSee Jupyter notebook for complete workflow.")