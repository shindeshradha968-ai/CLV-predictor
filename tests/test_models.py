"""
Unit Tests for Models Module
============================

Tests for model training, evaluation, and prediction functionality.

Author: CLV Predictor Team
Version: 1.0.0
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loader import generate_sample_data
from data.data_preprocessing import DataPreprocessor
from features.feature_builder import FeatureBuilder
from models.train import ModelTrainer
from models.evaluate import ModelEvaluator
from models.predict import CLVPredictor


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    @pytest.fixture
    def trainer(self):
        """Create a ModelTrainer instance."""
        return ModelTrainer()
    
    @pytest.fixture
    def training_data(self):
        """Generate training data."""
        sample_df = generate_sample_data(n_records=1000)
        preprocessor = DataPreprocessor()
        cleaned_df = preprocessor.clean_data(sample_df)
        
        builder = FeatureBuilder()
        features_df = builder.build_features(cleaned_df)
        
        X, y = builder.get_feature_importance_ready_data(features_df)
        return X, y
    
    def test_init(self, trainer):
        """Test ModelTrainer initialization."""
        assert trainer is not None
        assert hasattr(trainer, 'config')
        assert hasattr(trainer, 'models')
    
    def test_train_linear_regression(self, trainer, training_data):
        """Test Linear Regression training."""
        X, y = training_data
        results = trainer.train_model('linear_regression', X, y)
        
        assert results is not None
        assert 'model' in results
        assert 'train_score' in results
        assert results['train_score'] is not None
    
    def test_train_random_forest(self, trainer, training_data):
        """Test Random Forest training."""
        X, y = training_data
        results = trainer.train_model('random_forest', X, y)
        
        assert results is not None
        assert 'model' in results
        assert 'feature_importance' in results
    
    def test_train_xgboost(self, trainer, training_data):
        """Test XGBoost training."""
        X, y = training_data
        results = trainer.train_model('xgboost', X, y)
        
        assert results is not None
        assert 'model' in results
    
    def test_train_all_models(self, trainer, training_data):
        """Test training all models."""
        X, y = training_data
        results = trainer.train_all_models(X, y, test_size=0.2)
        
        assert len(results) >= 1
        assert trainer.best_model is not None
        assert trainer.best_model_name is not None
    
    def test_get_best_model(self, trainer, training_data):
        """Test best model selection."""
        X, y = training_data
        trainer.train_all_models(X, y)
        
        best_model, best_name = trainer.get_best_model()
        
        assert best_model is not None
        assert best_name is not None
    
    def test_save_and_load_model(self, trainer, training_data):
        """Test model saving and loading."""
        X, y = training_data
        trainer.train_all_models(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            save_path = trainer.save_model(output_path=f.name)
            
            # Load model
            loaded_data = trainer.load_model(save_path)
            
            assert loaded_data is not None
            assert 'model' in loaded_data
            assert 'scaler' in loaded_data
    
    def test_get_feature_importance(self, trainer, training_data):
        """Test feature importance extraction."""
        X, y = training_data
        trainer.train_all_models(X, y)
        
        importance_df = trainer.get_feature_importance(top_n=10)
        
        assert importance_df is not None
        assert len(importance_df) <= 10
        assert 'Feature' in importance_df.columns
        assert 'Importance' in importance_df.columns
    
    def test_get_training_summary(self, trainer, training_data):
        """Test training summary generation."""
        X, y = training_data
        trainer.train_all_models(X, y)
        
        summary = trainer.get_training_summary()
        
        assert summary is not None
        assert isinstance(summary, pd.DataFrame)
        assert 'Model' in summary.columns


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a ModelEvaluator instance."""
        return ModelEvaluator()
    
    @pytest.fixture
    def trained_model(self):
        """Train a model for testing."""
        sample_df = generate_sample_data(n_records=1000)
        preprocessor = DataPreprocessor()
        cleaned_df = preprocessor.clean_data(sample_df)
        
        builder = FeatureBuilder()
        features_df = builder.build_features(cleaned_df)
        X, y = builder.get_feature_importance_ready_data(features_df)
        
        trainer = ModelTrainer()
        trainer.train_all_models(X, y)
        
        return trainer, X, y
    
    def test_init(self, evaluator):
        """Test ModelEvaluator initialization."""
        assert evaluator is not None
        assert hasattr(evaluator, 'config')
    
    def test_evaluate_model(self, evaluator, trained_model):
        """Test model evaluation."""
        trainer, X, y = trained_model
        
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2)
        
        X_test_scaled = pd.DataFrame(
            trainer.scaler.transform(X_test),
            columns=X_test.columns
        )
        
        metrics = evaluator.evaluate_model(
            trainer.best_model,
            X_test_scaled,
            y_test,
            "test_model"
        )
        
        assert metrics is not None
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'R2' in metrics
    
    def test_analyze_residuals(self, evaluator, trained_model):
        """Test residual analysis."""
        trainer, X, y = trained_model
        
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2)
        
        X_test_scaled = pd.DataFrame(
            trainer.scaler.transform(X_test),
            columns=X_test.columns
        )
        
        evaluator.evaluate_model(trainer.best_model, X_test_scaled, y_test, "test_model")
        analysis = evaluator.analyze_residuals("test_model")
        
        assert analysis is not None
        assert 'mean' in analysis
        assert 'std' in analysis
        assert 'skewness' in analysis


class TestCLVPredictor:
    """Tests for CLVPredictor class."""
    
    @pytest.fixture
    def predictor_with_model(self):
        """Create a CLVPredictor with a trained model."""
        sample_df = generate_sample_data(n_records=500)
        preprocessor = DataPreprocessor()
        cleaned_df = preprocessor.clean_data(sample_df)
        
        builder = FeatureBuilder()
        features_df = builder.build_features(cleaned_df)
        X, y = builder.get_feature_importance_ready_data(features_df)
        
        trainer = ModelTrainer()
        trainer.train_all_models(X, y)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            trainer.save_model(output_path=f.name)
            predictor = CLVPredictor(f.name)
        
        return predictor, X
    
    def test_init(self):
        """Test CLVPredictor initialization."""
        predictor = CLVPredictor()
        assert predictor is not None
        assert predictor.model is None  # No model loaded
    
    def test_load_model(self, predictor_with_model):
        """Test model loading."""
        predictor, _ = predictor_with_model
        
        assert predictor.model is not None
        assert predictor.feature_names is not None
    
    def test_predict(self, predictor_with_model):
        """Test prediction functionality."""
        predictor, X = predictor_with_model
        
        predictions = predictor.predict(X.head(10))
        
        assert predictions is not None
        assert len(predictions) == 10
        assert all(p >= 0 for p in predictions)  # CLV should be non-negative
    
    def test_predict_with_confidence(self, predictor_with_model):
        """Test prediction with confidence intervals."""
        predictor, X = predictor_with_model
        
        results = predictor.predict_with_confidence(X.head(10))
        
        assert 'Predicted_CLV' in results.columns
        assert 'Lower_Bound' in results.columns
        assert 'Upper_Bound' in results.columns
    
    def test_segment_customers(self, predictor_with_model):
        """Test customer segmentation."""
        predictor, X = predictor_with_model
        
        predictions = predictor.predict(X)
        segments_df = predictor.segment_customers(predictions)
        
        assert 'CLV_Segment' in segments_df.columns
        assert 'Predicted_CLV' in segments_df.columns
    
    def test_get_model_info(self, predictor_with_model):
        """Test model info retrieval."""
        predictor, _ = predictor_with_model
        
        info = predictor.get_model_info()
        
        assert info is not None
        assert 'model_name' in info
        assert 'n_features' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])