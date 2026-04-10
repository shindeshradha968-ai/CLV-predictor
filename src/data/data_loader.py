"""
Data Loader Module
==================

This module provides functionality for loading data from various sources
including CSV files, databases, and APIs.

Author: CLV Predictor Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
import yaml
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")


class DataLoader:
    """
    A class to handle data loading operations for the CLV Predictor application.
    
    This class provides methods to load data from CSV files, validate the data
    structure, and perform initial data quality checks.
    
    Attributes:
        config (dict): Configuration dictionary loaded from config.yaml
        data_path (Path): Path to the data directory
        
    Example:
        >>> loader = DataLoader()
        >>> df = loader.load_csv("data/raw/online_retail.csv")
        >>> print(df.shape)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the DataLoader with configuration settings.
        
        Args:
            config_path (Optional[str]): Path to the configuration file.
                                         Defaults to 'src/config/config.yaml'
        """
        self.config = self._load_config(config_path)
        self.data_path = Path(self.config.get('data', {}).get('raw_data_path', 'data/raw'))
        self.required_columns = self._get_required_columns()
        
        logger.info("DataLoader initialized successfully")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (Optional[str]): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if config_path is None:
            # Try to find config file relative to this module
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration if config file is not found.
        
        Returns:
            Dict[str, Any]: Default configuration dictionary
        """
        return {
            'data': {
                'columns': {
                    'invoice_no': 'InvoiceNo',
                    'stock_code': 'StockCode',
                    'description': 'Description',
                    'quantity': 'Quantity',
                    'invoice_date': 'InvoiceDate',
                    'unit_price': 'UnitPrice',
                    'customer_id': 'CustomerID',
                    'country': 'Country'
                },
                'encoding': 'utf-8'
            }
        }
    
    def _get_required_columns(self) -> list:
        """
        Get the list of required columns from configuration.
        
        Returns:
            list: List of required column names
        """
        columns_config = self.config.get('data', {}).get('columns', {})
        return list(columns_config.values())
    
    def load_csv(
        self, 
        file_path: Union[str, Path],
        encoding: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path (Union[str, Path]): Path to the CSV file
            encoding (Optional[str]): File encoding. Defaults to config setting
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            pd.DataFrame: Loaded DataFrame
            
        Raises:
            FileNotFoundError: If the specified file does not exist
            ValueError: If the file cannot be parsed as CSV
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"The file {file_path} does not exist")
        
        # Use encoding from config if not specified
        if encoding is None:
            encoding = self.config.get('data', {}).get('encoding', 'utf-8')
        
        try:
            # Try different encodings if the default fails
            encodings_to_try = [encoding, 'latin1', 'iso-8859-1', 'cp1252']
            
            for enc in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=enc, **kwargs)
                    logger.info(f"Successfully loaded {file_path} with encoding {enc}")
                    logger.info(f"DataFrame shape: {df.shape}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Could not decode file with any encoding: {encodings_to_try}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the loaded DataFrame against expected schema.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            Dict[str, Any]: Validation results including:
                - is_valid (bool): Whether validation passed
                - missing_columns (list): List of missing required columns
                - data_quality (dict): Data quality metrics
        """
        validation_results = {
            'is_valid': True,
            'missing_columns': [],
            'extra_columns': [],
            'data_quality': {}
        }
        
        # Check for missing columns
        existing_columns = set(df.columns)
        required_columns = set(self.required_columns)
        
        missing = required_columns - existing_columns
        extra = existing_columns - required_columns
        
        if missing:
            validation_results['is_valid'] = False
            validation_results['missing_columns'] = list(missing)
            logger.warning(f"Missing required columns: {missing}")
        
        validation_results['extra_columns'] = list(extra)
        
        # Calculate data quality metrics
        validation_results['data_quality'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        logger.info(f"Validation complete. Valid: {validation_results['is_valid']}")
        return validation_results
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to summarize
            
        Returns:
            Dict[str, Any]: Summary statistics and information
        """
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'null_counts': df.isnull().sum().to_dict(),
            'unique_counts': {col: df[col].nunique() for col in df.columns},
            'numeric_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        logger.info("Data summary generated successfully")
        return summary
    
    def load_and_validate(
        self, 
        file_path: Union[str, Path],
        **kwargs
    ) -> tuple:
        """
        Load data and validate it in one operation.
        
        Args:
            file_path (Union[str, Path]): Path to the CSV file
            **kwargs: Additional arguments to pass to load_csv
            
        Returns:
            tuple: (DataFrame, validation_results)
        """
        df = self.load_csv(file_path, **kwargs)
        validation_results = self.validate_data(df)
        
        return df, validation_results


def generate_sample_data(n_records: int = 10000, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate sample retail transaction data for testing and demonstration.
    
    Args:
        n_records (int): Number of records to generate
        output_path (Optional[str]): Path to save the generated data
        
    Returns:
        pd.DataFrame: Generated sample data
    """
    np.random.seed(42)
    
    # Generate customer IDs (fewer customers than transactions)
    n_customers = n_records // 10
    customer_ids = np.random.randint(10000, 20000, n_customers)
    
    # Generate sample data
    data = {
        'InvoiceNo': [f'INV{str(i).zfill(6)}' for i in range(n_records)],
        'StockCode': [f'STK{np.random.randint(1000, 9999)}' for _ in range(n_records)],
        'Description': np.random.choice([
            'WHITE HANGING HEART T-LIGHT HOLDER',
            'WHITE METAL LANTERN',
            'CREAM CUPID HEARTS COAT HANGER',
            'KNITTED UNION FLAG HOT WATER BOTTLE',
            'RED WOOLLY HOTTIE WHITE HEART',
            'SET 7 BABUSHKA NESTING BOXES',
            'GLASS STAR FROSTED T-LIGHT HOLDER',
            'HAND WARMER UNION JACK',
            'HAND WARMER RED POLKA DOT',
            'ASSORTED COLOUR BIRD ORNAMENT'
        ], n_records),
        'Quantity': np.random.randint(1, 25, n_records),
        'InvoiceDate': pd.date_range(
            start='2020-01-01', 
            periods=n_records, 
            freq='10min'
        ).strftime('%Y-%m-%d %H:%M:%S'),
        'UnitPrice': np.round(np.random.uniform(0.5, 50.0, n_records), 2),
        'CustomerID': np.random.choice(customer_ids, n_records),
        'Country': np.random.choice([
            'United Kingdom', 'France', 'Germany', 'Spain', 
            'Italy', 'Netherlands', 'Belgium', 'Switzerland'
        ], n_records, p=[0.7, 0.08, 0.06, 0.04, 0.04, 0.03, 0.03, 0.02])
    }
    
    df = pd.DataFrame(data)
    
    # Add some cancelled orders (negative quantities)
    cancel_indices = np.random.choice(n_records, size=int(n_records * 0.05), replace=False)
    df.loc[cancel_indices, 'InvoiceNo'] = 'C' + df.loc[cancel_indices, 'InvoiceNo']
    df.loc[cancel_indices, 'Quantity'] = -abs(df.loc[cancel_indices, 'Quantity'])
    
    # Add some missing CustomerIDs
    missing_indices = np.random.choice(n_records, size=int(n_records * 0.02), replace=False)
    df.loc[missing_indices, 'CustomerID'] = np.nan
    
    logger.info(f"Generated {n_records} sample records")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Sample data saved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Generate sample data
    sample_df = generate_sample_data(n_records=5000, output_path="data/raw/sample_retail.csv")
    
    # Load and validate
    df, validation = loader.load_and_validate("data/raw/sample_retail.csv")
    
    print("\nValidation Results:")
    print(f"Is Valid: {validation['is_valid']}")
    print(f"Total Rows: {validation['data_quality']['total_rows']}")
    print(f"Duplicate Rows: {validation['data_quality']['duplicate_rows']}")