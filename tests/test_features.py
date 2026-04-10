"""
Unit Tests for Features Module
==============================

Tests for feature engineering functionality.

Author: CLV Predictor Team
Version: 1.0.0
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loader import generate_sample_data
from data.data_preprocessing import DataPreprocessor
from features.feature_builder import FeatureBuilder


class TestFeatureBuilder:
    """Tests for FeatureBuilder class."""
    
    @pytest.fixture
    def feature_builder(self):
        """Create a FeatureBuilder instance."""
        return FeatureBuilder()
    
    @pytest.fixture
    def preprocessed_data(self):
        """Generate and preprocess sample data."""
        sample_df = generate_sample_data(n_records=1000)
        preprocessor = DataPreprocessor()
        return preprocessor.clean_data(sample_df)
    
    def test_init(self, feature_builder):
        """Test FeatureBuilder initialization."""
        assert feature_builder is not None
        assert hasattr(feature_builder, 'config')
        assert hasattr(feature_builder, 'scaler')
    
    def test_build_features(self, feature_builder, preprocessed_data):
        """Test feature building functionality."""
        features_df = feature_builder.build_features(preprocessed_data)
        
        assert features_df is not None
        assert isinstance(features_df, pd.DataFrame)
        assert 'CustomerID' in features_df.columns
    
    def test_rfm_features(self, feature_builder, preprocessed_data):
        """Test RFM feature generation."""
        features_df = feature_builder.build_features(
            preprocessed_data,
            include_rfm=True,
            include_behavioral=False,
            include_time=False,
            include_cohort=False
        )
        
        assert 'Recency' in features_df.columns
        assert 'Frequency' in features_df.columns
        assert 'Monetary' in features_df.columns
        assert 'RFMScore' in features_df.columns
    
    def test_behavioral_features(self, feature_builder, preprocessed_data):
        """Test behavioral feature generation."""
        features_df = feature_builder.build_features(
            preprocessed_data,
            include_rfm=False,
            include_behavioral=True,
            include_time=False,
            include_cohort=False
        )
        
        assert 'AvgTransactionValue' in features_df.columns
        assert 'UniqueProducts' in features_df.columns
        assert 'TotalTransactions' in features_df.columns
    
    def test_time_features(self, feature_builder, preprocessed_data):
        """Test time-based feature generation."""
        features_df = feature_builder.build_features(
            preprocessed_data,
            include_rfm=False,
            include_behavioral=False,
            include_time=True,
            include_cohort=False
        )
        
        assert 'CustomerTenure' in features_df.columns
        assert 'DaysSinceLastPurchase' in features_df.columns
    
    def test_no_missing_values(self, feature_builder, preprocessed_data):
        """Test that features have no missing values."""
        features_df = feature_builder.build_features(preprocessed_data)
        
        # Check numeric columns for NaN
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        assert not features_df[numeric_cols].isnull().any().any()
    
    def test_scale_features(self, feature_builder, preprocessed_data):
        """Test feature scaling."""
        features_df = feature_builder.build_features(preprocessed_data)
        scaled_df = feature_builder.scale_features(features_df, method='standard')
        
        assert scaled_df is not None
        assert len(scaled_df) == len(features_df)
    
    def test_get_feature_importance_ready_data(self, feature_builder, preprocessed_data):
        """Test preparation for feature importance analysis."""
        features_df = feature_builder.build_features(preprocessed_data)
        X, y = feature_builder.get_feature_importance_ready_data(features_df)
        
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert 'CustomerID' not in X.columns
        assert 'Monetary' not in X.columns  # Target column excluded


class TestRFMScoring:
    """Tests for RFM scoring functionality."""
    
    @pytest.fixture
    def feature_builder(self):
        return FeatureBuilder()
    
    @pytest.fixture
    def preprocessed_data(self):
        sample_df = generate_sample_data(n_records=500)
        preprocessor = DataPreprocessor()
        return preprocessor.clean_data(sample_df)
    
    def test_rfm_scores_range(self, feature_builder, preprocessed_data):
        """Test that RFM scores are in valid range."""
        features_df = feature_builder.build_features(
            preprocessed_data,
            include_rfm=True,
            include_behavioral=False,
            include_time=False,
            include_cohort=False
        )
        
        for score_col in ['RecencyScore', 'FrequencyScore', 'MonetaryScore']:
            if score_col in features_df.columns:
                assert features_df[score_col].min() >= 1
                assert features_df[score_col].max() <= 5
    
    def test_customer_segments(self, feature_builder, preprocessed_data):
        """Test customer segment assignment."""
        features_df = feature_builder.build_features(
            preprocessed_data,
            include_rfm=True,
            include_behavioral=False,
            include_time=False,
            include_cohort=False
        )
        
        if 'CustomerSegment' in features_df.columns:
            valid_segments = [
                'Champions', 'Loyal Customers', 'New Customers',
                'At Risk', 'Cant Lose', 'Lost', 'Potential Loyalists'
            ]
            assert features_df['CustomerSegment'].isin(valid_segments).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])