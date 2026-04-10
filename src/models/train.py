"""
Model Training Module
=====================

This module handles training of multiple machine learning models
for Customer Lifetime Value prediction.

Author: CLV Predictor Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import yaml
import joblib
from loguru import logger
from datetime import datetime
import sys

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")


class ModelTrainer:
    """
    A class to handle training of multiple ML models for CLV prediction.
    
    This class supports training Linear Regression, Random Forest, and XGBoost
    models with configurable hyperparameters and cross-validation.
    
    Attributes:
        config (dict): Configuration dictionary
        models (dict): Dictionary of trained models
        best_model (object): Best performing model
        best_model_name (str): Name of the best model
        scaler (StandardScaler): Feature scaler
        
    Example:
        >>> trainer = ModelTrainer()
        >>> results = trainer.train_all_models(X_train, y_train)
        >>> best_model = trainer.get_best_model()
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ModelTrainer with configuration settings.
        
        Args:
            config_path (Optional[str]): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.training_results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        logger.info("ModelTrainer initialized successfully")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'models': {
                'linear_regression': {'fit_intercept': True},
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': 42
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'training': {
                'test_size': 0.2,
                'random_state': 42,
                'cross_validation_folds': 5
            }
        }
    
    def _get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get model parameters from configuration.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Model parameters
        """
        return self.config.get('models', {}).get(model_name, {})
    
    def _create_linear_regression(self) -> LinearRegression:
        """
        Create a Linear Regression model.
        
        Returns:
            LinearRegression: Configured Linear Regression model
        """
        params = self._get_model_params('linear_regression')
        return LinearRegression(
            fit_intercept=params.get('fit_intercept', True)
        )
    
    def _create_random_forest(self) -> RandomForestRegressor:
        """
        Create a Random Forest Regressor model.
        
        Returns:
            RandomForestRegressor: Configured Random Forest model
        """
        params = self._get_model_params('random_forest')
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 5),
            min_samples_leaf=params.get('min_samples_leaf', 2),
            random_state=params.get('random_state', 42),
            n_jobs=params.get('n_jobs', -1)
        )
    
    def _create_xgboost(self) -> Any:
        """
        Create an XGBoost Regressor model.
        
        Returns:
            XGBRegressor: Configured XGBoost model
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, using GradientBoosting instead")
            params = self._get_model_params('xgboost')
            return GradientBoostingRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=params.get('random_state', 42)
            )
        
        params = self._get_model_params('xgboost')
        return xgb.XGBRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            random_state=params.get('random_state', 42),
            n_jobs=params.get('n_jobs', -1),
            objective=params.get('objective', 'reg:squarederror')
        )
    
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Train a single model.
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (Optional[pd.DataFrame]): Validation features
            y_val (Optional[pd.Series]): Validation target
            
        Returns:
            Dict[str, Any]: Training results including model and metrics
        """
        logger.info(f"Training {model_name}...")
        
        # Create model
        if model_name == 'linear_regression':
            model = self._create_linear_regression()
        elif model_name == 'random_forest':
            model = self._create_random_forest()
        elif model_name == 'xgboost':
            model = self._create_xgboost()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate training score
        train_score = model.score(X_train, y_train)
        
        # Calculate validation score if validation set provided
        val_score = None
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
        
        # Store model
        self.models[model_name] = model
        
        # Prepare results
        results = {
            'model': model,
            'model_name': model_name,
            'train_score': train_score,
            'val_score': val_score,
            'training_time': training_time,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = dict(zip(
                self.feature_names,
                model.feature_importances_
            ))
        elif hasattr(model, 'coef_'):
            results['feature_importance'] = dict(zip(
                self.feature_names,
                np.abs(model.coef_)
            ))
        
        self.training_results[model_name] = results
        
        val_score_str = f"{val_score:.4f}" if val_score is not None else "N/A"
        logger.info(f"{model_name} - Train R²: {train_score:.4f}, "
                   f"Val R²: {val_score_str}, "
                   f"Time: {training_time:.2f}s")
        
        return results
    
    def train_all_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        scale_features: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all available models.
        
        Args:
            X (pd.DataFrame): Features DataFrame
            y (pd.Series): Target Series
            test_size (float): Proportion of data for validation
            scale_features (bool): Whether to scale features
            
        Returns:
            Dict[str, Dict[str, Any]]: Results for all models
        """
        logger.info("Training all models...")
        
        # Split data
        random_state = self.config.get('training', {}).get('random_state', 42)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features if requested
        if scale_features:
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Train each model
        models_to_train = ['linear_regression', 'random_forest', 'xgboost']
        
        for model_name in models_to_train:
            try:
                self.train_model(
                    model_name,
                    X_train_scaled,
                    y_train,
                    X_val_scaled,
                    y_val
                )
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        # Determine best model
        self._select_best_model()
        
        return self.training_results
    
    def train_with_cross_validation(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train model with cross-validation.
        
        Args:
            model_name (str): Name of the model
            X (pd.DataFrame): Features DataFrame
            y (pd.Series): Target Series
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        logger.info(f"Training {model_name} with {cv_folds}-fold cross-validation...")
        
        # Create model
        if model_name == 'linear_regression':
            model = self._create_linear_regression()
        elif model_name == 'random_forest':
            model = self._create_random_forest()
        elif model_name == 'xgboost':
            model = self._create_xgboost()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')
        
        results = {
            'model_name': model_name,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_folds': cv_folds
        }
        
        logger.info(f"{model_name} CV Results - Mean R²: {results['cv_mean']:.4f} "
                   f"(+/- {results['cv_std']:.4f})")
        
        return results
    
    def _select_best_model(self) -> None:
        """Select the best model based on validation scores."""
        if not self.training_results:
            logger.warning("No models trained yet")
            return
        
        best_score = -np.inf
        best_name = None
        
        for name, results in self.training_results.items():
            score = results.get('val_score') or results.get('train_score', 0)
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_name:
            self.best_model = self.models[best_name]
            self.best_model_name = best_name
            logger.info(f"Best model: {best_name} with score: {best_score:.4f}")
    
    def get_best_model(self) -> Tuple[Any, str]:
        """
        Get the best performing model.
        
        Returns:
            Tuple[Any, str]: (Best model object, model name)
        """
        if self.best_model is None:
            self._select_best_model()
        
        return self.best_model, self.best_model_name
    
    def hyperparameter_tuning(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, List]] = None,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_name (str): Name of the model to tune
            X (pd.DataFrame): Features DataFrame
            y (pd.Series): Target Series
            param_grid (Optional[Dict[str, List]]): Parameter grid for search
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Best parameters and scores
        """
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        # Create base model
        if model_name == 'linear_regression':
            model = Ridge()
            if param_grid is None:
                param_grid = {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
        elif model_name == 'random_forest':
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
        elif model_name == 'xgboost':
            if XGBOOST_AVAILABLE:
                model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
                if param_grid is None:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
            else:
                model = GradientBoostingRegressor(random_state=42)
                if param_grid is None:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2]
                    }
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        
        # Store best model
        self.models[f"{model_name}_tuned"] = grid_search.best_estimator_
        
        results = {
            'model_name': model_name,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': pd.DataFrame(grid_search.cv_results_).to_dict()
        }
        
        logger.info(f"Best params for {model_name}: {results['best_params']}")
        logger.info(f"Best CV score: {results['best_score']:.4f}")
        
        return results
    
    def save_model(
        self,
        model_name: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model_name (Optional[str]): Name of model to save. 
                                        Defaults to best model.
            output_path (Optional[str]): Path to save the model
            
        Returns:
            str: Path where the model was saved
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if output_path is None:
            output_path = self.config.get('models', {}).get(
                'save_path', 'models/'
            )
            output_path = Path(output_path) / f"{model_name}_model.pkl"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and scaler together
        model_data = {
            'model': self.models[model_name],
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': model_name,
            'training_results': self.training_results.get(model_name, {}),
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, output_path)
        logger.info(f"Model saved to {output_path}")
        
        return str(output_path)
    
    def save_all_models(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Save all trained models to disk.
        
        Args:
            output_dir (Optional[str]): Directory to save models
            
        Returns:
            List[str]: Paths where models were saved
        """
        saved_paths = []
        
        if output_dir is None:
            output_dir = self.config.get('models', {}).get('save_path', 'models/')
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name in self.models:
            path = self.save_model(model_name, output_dir / f"{model_name}_model.pkl")
            saved_paths.append(path)
        
        return saved_paths
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load a saved model from disk.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            Dict[str, Any]: Model data including model, scaler, and metadata
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        # Restore to trainer
        model_name = model_data['model_name']
        self.models[model_name] = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {model_path}")
        return model_data
    
    def get_feature_importance(
        self,
        model_name: Optional[str] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance from a trained model.
        
        Args:
            model_name (Optional[str]): Name of the model
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Get importance values
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            logger.warning(f"Model {model_name} doesn't have feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Normalize importance
        importance_df['Importance_Normalized'] = (
            importance_df['Importance'] / importance_df['Importance'].sum() * 100
        )
        
        return importance_df.head(top_n)
    
    def get_training_summary(self) -> pd.DataFrame:
        """
        Get a summary of all training results.
        
        Returns:
            pd.DataFrame: Summary of training results
        """
        if not self.training_results:
            return pd.DataFrame()
        
        summary_data = []
        for name, results in self.training_results.items():
            summary_data.append({
                'Model': name,
                'Train R²': results.get('train_score', 'N/A'),
                'Validation R²': results.get('val_score', 'N/A'),
                'Training Time (s)': results.get('training_time', 'N/A'),
                'N Features': results.get('n_features', 'N/A'),
                'N Samples': results.get('n_samples', 'N/A')
            })
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data.data_loader import generate_sample_data
    from data.data_preprocessing import DataPreprocessor
    from features.feature_builder import FeatureBuilder
    
    # Generate and prepare data
    sample_df = generate_sample_data(n_records=5000)
    preprocessor = DataPreprocessor()
    cleaned_df = preprocessor.clean_data(sample_df)
    
    builder = FeatureBuilder()
    features_df = builder.build_features(cleaned_df)
    
    X, y = builder.get_feature_importance_ready_data(features_df, target_col='Monetary')
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(X, y)
    
    print("\nTraining Summary:")
    print(trainer.get_training_summary())
    
    print("\nFeature Importance (Best Model):")
    print(trainer.get_feature_importance(top_n=10))
    
    # Save best model
    trainer.save_model()