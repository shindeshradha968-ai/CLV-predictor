"""
Model Evaluation Module
=======================

This module handles comprehensive evaluation of trained models
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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error
)
from sklearn.model_selection import cross_val_score, learning_curve

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")


class ModelEvaluator:
    """
    A class to evaluate machine learning models for CLV prediction.
    
    This class provides comprehensive evaluation metrics, visualizations,
    and diagnostic tools for assessing model performance.
    
    Attributes:
        config (dict): Configuration dictionary
        evaluation_results (dict): Stored evaluation results
        
    Example:
        >>> evaluator = ModelEvaluator()
        >>> metrics = evaluator.evaluate_model(model, X_test, y_test)
        >>> print(metrics)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            config_path (Optional[str]): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.evaluation_results = {}
        
        logger.info("ModelEvaluator initialized successfully")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return {'evaluation': {'metrics': ['mae', 'mse', 'rmse', 'r2', 'mape']}}
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate a model on test data.
        
        Args:
            model: Trained model object
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model for logging
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Store results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'y_test': y_test,
            'y_pred': y_pred,
            'evaluated_at': datetime.now().isoformat()
        }
        
        logger.info(f"{model_name} Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        # Ensure numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Handle edge cases
        y_pred = np.nan_to_num(y_pred, nan=0, posinf=0, neginf=0)
        
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'Explained_Variance': explained_variance_score(y_true, y_pred),
            'Max_Error': max_error(y_true, y_pred)
        }
        
        # Calculate MAPE carefully to avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['MAPE'] = np.nan
        
        # Additional metrics
        metrics['Mean_Residual'] = np.mean(y_true - y_pred)
        metrics['Std_Residual'] = np.std(y_true - y_pred)
        
        # Median Absolute Error
        metrics['MedAE'] = np.median(np.abs(y_true - y_pred))
        
        return metrics
    
    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same test set.
        
        Args:
            models (Dict[str, Any]): Dictionary of model_name: model pairs
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            pd.DataFrame: Comparison table of all models
        """
        logger.info("Comparing models...")
        
        comparison_data = []
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            metrics['Model'] = model_name
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Reorder columns
        cols = ['Model'] + [col for col in comparison_df.columns if col != 'Model']
        comparison_df = comparison_df[cols]
        
        # Sort by R2 score
        comparison_df = comparison_df.sort_values('R2', ascending=False)
        
        logger.info("Model comparison complete")
        return comparison_df
    
    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        scoring: str = 'r2'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation for a model.
        
        Args:
            model: Model object
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv_folds (int): Number of CV folds
            scoring (str): Scoring metric
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
        
        results = {
            'scores': scores.tolist(),
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max(),
            'cv_folds': cv_folds,
            'scoring': scoring
        }
        
        logger.info(f"CV Mean {scoring}: {results['mean']:.4f} (+/- {results['std']:.4f})")
        
        return results
    
    def analyze_residuals(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Analyze residuals for a evaluated model.
        
        Args:
            model_name (str): Name of the evaluated model
            
        Returns:
            Dict[str, Any]: Residual analysis results
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        results = self.evaluation_results[model_name]
        y_true = np.array(results['y_test'])
        y_pred = np.array(results['y_pred'])
        residuals = y_true - y_pred
        
        analysis = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'median': np.median(residuals),
            'skewness': pd.Series(residuals).skew(),
            'kurtosis': pd.Series(residuals).kurtosis(),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'percentile_5': np.percentile(residuals, 5),
            'percentile_95': np.percentile(residuals, 95)
        }
        
        logger.info(f"Residual analysis for {model_name}:")
        logger.info(f"  Mean: {analysis['mean']:.4f}, Std: {analysis['std']:.4f}")
        
        return analysis
    
    def plot_predictions_vs_actual(
        self,
        model_name: str,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a scatter plot of predicted vs actual values.
        
        Args:
            model_name (str): Name of the evaluated model
            figsize (Tuple[int, int]): Figure size
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        results = self.evaluation_results[model_name]
        y_true = np.array(results['y_test'])
        y_pred = np.array(results['y_pred'])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='none', s=50)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Labels and title
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'Predicted vs Actual - {model_name}', fontsize=14)
        
        # Add R² annotation
        r2 = results['metrics']['R2']
        ax.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_residuals(
        self,
        model_name: str,
        figsize: Tuple[int, int] = (14, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create residual diagnostic plots.
        
        Args:
            model_name (str): Name of the evaluated model
            figsize (Tuple[int, int]): Figure size
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        results = self.evaluation_results[model_name]
        y_true = np.array(results['y_test'])
        y_pred = np.array(results['y_pred'])
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        
        # Plot 2: Histogram of residuals
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        
        # Plot 3: Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot')
        
        plt.suptitle(f'Residual Diagnostics - {model_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names (List[str]): List of feature names
            top_n (int): Number of top features to show
            figsize (Tuple[int, int]): Figure size
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            raise ValueError("Model doesn't have feature importance attributes")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True).tail(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
        
        # Color bars by importance
        colors = plt.cm.RdYlGn(importance_df['Importance'] / importance_df['Importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_learning_curve(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot learning curve for a model.
        
        Args:
            model: Model object
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv_folds (int): Number of CV folds
            figsize (Tuple[int, int]): Figure size
            save_path (Optional[str]): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y,
            cv=cv_folds,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                       alpha=0.1, color='blue')
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                       alpha=0.1, color='orange')
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        ax.plot(train_sizes, test_mean, 'o-', color='orange', label='Validation Score')
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Learning Curve', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def generate_report(
        self,
        model_name: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            model_name (str): Name of the evaluated model
            output_path (Optional[str]): Path to save the report
            
        Returns:
            str: Report content as string
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        results = self.evaluation_results[model_name]
        metrics = results['metrics']
        residual_analysis = self.analyze_residuals(model_name)
        
        report = f"""
================================================================================
                    MODEL EVALUATION REPORT
================================================================================

Model Name: {model_name}
Evaluation Date: {results['evaluated_at']}

--------------------------------------------------------------------------------
                         PERFORMANCE METRICS
--------------------------------------------------------------------------------
Mean Absolute Error (MAE):        {metrics['MAE']:.4f}
Mean Squared Error (MSE):         {metrics['MSE']:.4f}
Root Mean Squared Error (RMSE):   {metrics['RMSE']:.4f}
R-Squared (R²):                   {metrics['R2']:.4f}
Explained Variance:               {metrics['Explained_Variance']:.4f}
Mean Absolute Percentage Error:   {metrics['MAPE']:.2f}%
Median Absolute Error:            {metrics['MedAE']:.4f}
Maximum Error:                    {metrics['Max_Error']:.4f}

--------------------------------------------------------------------------------
                         RESIDUAL ANALYSIS
--------------------------------------------------------------------------------
Mean Residual:                    {residual_analysis['mean']:.4f}
Standard Deviation:               {residual_analysis['std']:.4f}
Median Residual:                  {residual_analysis['median']:.4f}
Skewness:                         {residual_analysis['skewness']:.4f}
Kurtosis:                         {residual_analysis['kurtosis']:.4f}
5th Percentile:                   {residual_analysis['percentile_5']:.4f}
95th Percentile:                  {residual_analysis['percentile_95']:.4f}

--------------------------------------------------------------------------------
                            INTERPRETATION
--------------------------------------------------------------------------------
"""
        # Add interpretation based on metrics
        if metrics['R2'] > 0.9:
            report += "• Excellent model fit (R² > 0.9)\n"
        elif metrics['R2'] > 0.7:
            report += "• Good model fit (R² > 0.7)\n"
        elif metrics['R2'] > 0.5:
            report += "• Moderate model fit (R² > 0.5)\n"
        else:
            report += "• Poor model fit (R² < 0.5) - consider improving features or model\n"
        
        if abs(residual_analysis['skewness']) < 0.5:
            report += "• Residuals are approximately normally distributed\n"
        else:
            report += "• Residuals show skewness - model may have systematic bias\n"
        
        report += """
================================================================================
                              END OF REPORT
================================================================================
"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(model, X_test, y_test, "RandomForest")
    
    # Generate report
    report = evaluator.generate_report("RandomForest")
    print(report)