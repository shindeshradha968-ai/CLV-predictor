"""
Unit Tests for Data Module
==========================

Tests for data loading and preprocessing functionality.

Author: CLV Predictor Team
Version: 1.0.0
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loader import DataLoader, generate_sample_data
from data.data_preprocessing import DataPreprocessor


class TestDataLoader:
    """Tests for DataLoader class."""
    
    @pytest.fixture
    def data_loader(self):
        """Create a DataLoader instance."""
        return DataLoader()
    
    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file for testing."""
        df = generate_sample_data(n_records=100)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            return f.name
    
    def test_init(self, data_loader):
        """Test DataLoader initialization."""
        assert data_loader is not None
        assert hasattr(data_loader, 'config')
        assert hasattr(data_loader, 'required_columns')
    
    def test_load_csv(self, data_loader, sample_csv):
        """Test CSV loading functionality."""
        df = data_loader.load_csv(sample_csv)
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
    
    def test_load_csv_file_not_found(self, data_loader):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            data_loader.load_csv("nonexistent_file.csv")
    
    def test_validate_data(self, data_loader, sample_csv):
        """Test data validation."""
        df = data_loader.load_csv(sample_csv)
        validation_results = data_loader.validate_data(df)
        
        assert 'is_valid' in validation_results
        assert 'missing_columns' in validation_results
        assert 'data_quality' in validation_results
    
    def test_get_data_summary(self, data_loader, sample_csv):
        """Test data summary generation."""
        df = data_loader.load_csv(sample_csv)
        summary = data_loader.get_data_summary(df)
        
        assert 'shape' in summary
        assert 'columns' in summary
        assert 'dtypes' in summary
        assert summary['shape'] == (100, df.shape[1])


class TestGenerateSampleData:
    """Tests for sample data generation."""
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        df = generate_sample_data(n_records=500)
        
        assert df is not None
        assert len(df) == 500
        assert 'InvoiceNo' in df.columns
        assert 'CustomerID' in df.columns
        assert 'TotalAmount' not in df.columns  # Not added until preprocessing
    
    def test_generate_sample_data_columns(self):
        """Test that all required columns are present."""
        df = generate_sample_data(n_records=100)
        
        required_cols = [
            'InvoiceNo', 'StockCode', 'Description', 'Quantity',
            'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'
        ]
        
        for col in required_cols:
            assert col in df.columns


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a DataPreprocessor instance."""
        return DataPreprocessor()
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        return generate_sample_data(n_records=500)
    
    def test_init(self, preprocessor):
        """Test DataPreprocessor initialization."""
        assert preprocessor is not None
        assert hasattr(preprocessor, 'config')
    
    def test_clean_data(self, preprocessor, sample_data):
        """Test data cleaning functionality."""
        cleaned_df = preprocessor.clean_data(sample_data)
        
        assert cleaned_df is not None
        assert isinstance(cleaned_df, pd.DataFrame)
        assert len(cleaned_df) <= len(sample_data)
        
        # Check that TotalAmount column was added
        assert 'TotalAmount' in cleaned_df.columns
    
    def test_clean_data_removes_negative_quantities(self, preprocessor, sample_data):
        """Test that negative quantities are removed."""
        cleaned_df = preprocessor.clean_data(sample_data)
        
        assert (cleaned_df['Quantity'] > 0).all()
    
    def test_clean_data_removes_missing_customers(self, preprocessor, sample_data):
        """Test that rows with missing CustomerID are removed."""
        cleaned_df = preprocessor.clean_data(sample_data)
        
        assert cleaned_df['CustomerID'].notna().all()
    
    def test_aggregate_by_customer(self, preprocessor, sample_data):
        """Test customer-level aggregation."""
        cleaned_df = preprocessor.clean_data(sample_data)
        customer_df = preprocessor.aggregate_by_customer(cleaned_df)
        
        assert customer_df is not None
        assert 'CustomerID' in customer_df.columns
        assert 'Recency' in customer_df.columns
        assert 'TotalRevenue' in customer_df.columns
        
        # Each customer should appear only once
        assert customer_df['CustomerID'].nunique() == len(customer_df)
    
    def test_prepare_for_modeling(self, preprocessor, sample_data):
        """Test data preparation for modeling."""
        cleaned_df = preprocessor.clean_data(sample_data)
        customer_df = preprocessor.aggregate_by_customer(cleaned_df)
        
        X, y = preprocessor.prepare_for_modeling(customer_df)
        
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert 'CustomerID' not in X.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])