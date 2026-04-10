"""
Data Preprocessing Module
=========================

This module handles all data cleaning and preprocessing operations
for the CLV Predictor application.

Author: CLV Predictor Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import yaml
from loguru import logger
from datetime import datetime
import sys

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")


class DataPreprocessor:
    """
    A class to handle data preprocessing operations for CLV prediction.
    
    This class provides methods for cleaning, transforming, and preparing
    retail transaction data for feature engineering and model training.
    
    Attributes:
        config (dict): Configuration dictionary
        column_mapping (dict): Mapping of column names from config
        
    Example:
        >>> preprocessor = DataPreprocessor()
        >>> cleaned_df = preprocessor.clean_data(raw_df)
        >>> print(cleaned_df.shape)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the DataPreprocessor with configuration settings.
        
        Args:
            config_path (Optional[str]): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.column_mapping = self.config.get('data', {}).get('columns', {})
        
        # Define column names
        self.col_invoice = self.column_mapping.get('invoice_no', 'InvoiceNo')
        self.col_stock = self.column_mapping.get('stock_code', 'StockCode')
        self.col_desc = self.column_mapping.get('description', 'Description')
        self.col_quantity = self.column_mapping.get('quantity', 'Quantity')
        self.col_date = self.column_mapping.get('invoice_date', 'InvoiceDate')
        self.col_price = self.column_mapping.get('unit_price', 'UnitPrice')
        self.col_customer = self.column_mapping.get('customer_id', 'CustomerID')
        self.col_country = self.column_mapping.get('country', 'Country')
        
        logger.info("DataPreprocessor initialized successfully")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (Optional[str]): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
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
                }
            }
        }
    
    def clean_data(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning on the raw DataFrame.
        
        This method:
        - Removes duplicates
        - Handles missing values
        - Removes cancelled transactions
        - Filters out invalid quantities and prices
        - Converts data types
        
        Args:
            df (pd.DataFrame): Raw DataFrame to clean
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if not inplace:
            df = df.copy()
        
        initial_rows = len(df)
        logger.info(f"Starting data cleaning. Initial rows: {initial_rows}")
        
        # Step 1: Remove duplicates
        df = self._remove_duplicates(df)
        
        # Step 2: Handle missing CustomerIDs
        df = self._handle_missing_customers(df)
        
        # Step 3: Remove cancelled transactions
        df = self._remove_cancelled_transactions(df)
        
        # Step 4: Filter invalid quantities
        df = self._filter_invalid_quantities(df)
        
        # Step 5: Filter invalid prices
        df = self._filter_invalid_prices(df)
        
        # Step 6: Convert data types
        df = self._convert_data_types(df)
        
        # Step 7: Add derived columns
        df = self._add_derived_columns(df)
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        logger.info(f"Data cleaning complete. Final rows: {final_rows}, Removed: {removed_rows} ({removed_rows/initial_rows*100:.2f}%)")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        
        if before - after > 0:
            logger.info(f"Removed {before - after} duplicate rows")
        
        return df
    
    def _handle_missing_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle rows with missing CustomerID values.
        
        For CLV prediction, we need valid customer identifiers,
        so rows without CustomerID are removed.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with missing customers handled
        """
        before = len(df)
        
        # Remove rows with missing CustomerID
        df = df.dropna(subset=[self.col_customer])
        
        after = len(df)
        if before - after > 0:
            logger.info(f"Removed {before - after} rows with missing CustomerID")
        
        return df
    
    def _remove_cancelled_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove cancelled transactions (invoices starting with 'C').
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with cancelled transactions removed
        """
        before = len(df)
        
        # Identify cancelled transactions
        cancelled_mask = df[self.col_invoice].astype(str).str.startswith('C')
        df = df[~cancelled_mask]
        
        after = len(df)
        if before - after > 0:
            logger.info(f"Removed {before - after} cancelled transactions")
        
        return df
    
    def _filter_invalid_quantities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows with invalid (negative or zero) quantities.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with valid quantities only
        """
        before = len(df)
        
        df = df[df[self.col_quantity] > 0]
        
        after = len(df)
        if before - after > 0:
            logger.info(f"Removed {before - after} rows with invalid quantities")
        
        return df
    
    def _filter_invalid_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows with invalid (negative or zero) prices.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with valid prices only
        """
        before = len(df)
        
        df = df[df[self.col_price] > 0]
        
        after = len(df)
        if before - after > 0:
            logger.info(f"Removed {before - after} rows with invalid prices")
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns to appropriate data types.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with corrected data types
        """
        # Convert InvoiceDate to datetime
        if self.col_date in df.columns:
            df[self.col_date] = pd.to_datetime(df[self.col_date], errors='coerce')
            
            # Remove rows where date conversion failed
            df = df.dropna(subset=[self.col_date])
        
        # Convert CustomerID to integer (then to string for consistency)
        if self.col_customer in df.columns:
            df[self.col_customer] = df[self.col_customer].astype(float).astype(int).astype(str)
        
        # Ensure numeric types for quantity and price
        if self.col_quantity in df.columns:
            df[self.col_quantity] = pd.to_numeric(df[self.col_quantity], errors='coerce')
        
        if self.col_price in df.columns:
            df[self.col_price] = pd.to_numeric(df[self.col_price], errors='coerce')
        
        logger.info("Data types converted successfully")
        return df
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived columns for analysis.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with derived columns
        """
        # Calculate total amount per transaction line
        df['TotalAmount'] = df[self.col_quantity] * df[self.col_price]
        
        # Extract date components
        if self.col_date in df.columns and pd.api.types.is_datetime64_any_dtype(df[self.col_date]):
            df['Year'] = df[self.col_date].dt.year
            df['Month'] = df[self.col_date].dt.month
            df['DayOfWeek'] = df[self.col_date].dt.dayofweek
            df['Hour'] = df[self.col_date].dt.hour
            df['Date'] = df[self.col_date].dt.date
        
        logger.info("Derived columns added successfully")
        return df
    
    def aggregate_by_customer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate transaction data at the customer level.
        
        This creates a summary of each customer's purchasing behavior.
        
        Args:
            df (pd.DataFrame): Cleaned transaction DataFrame
            
        Returns:
            pd.DataFrame: Customer-level aggregated data
        """
        # Get the reference date (most recent transaction date)
        reference_date = df[self.col_date].max()
        
        # Aggregate by customer
        customer_df = df.groupby(self.col_customer).agg({
            self.col_invoice: 'nunique',  # Number of unique transactions
            self.col_date: ['min', 'max'],  # First and last purchase dates
            'TotalAmount': ['sum', 'mean', 'std'],  # Monetary values
            self.col_quantity: ['sum', 'mean'],  # Quantity metrics
            self.col_stock: 'nunique',  # Number of unique products
            self.col_country: 'first'  # Customer's country (first occurrence)
        }).reset_index()
        
        # Flatten column names
        customer_df.columns = [
            'CustomerID', 'TransactionCount', 'FirstPurchaseDate', 'LastPurchaseDate',
            'TotalRevenue', 'AvgOrderValue', 'StdOrderValue',
            'TotalQuantity', 'AvgQuantity', 'UniqueProducts', 'Country'
        ]
        
        # Calculate recency (days since last purchase)
        customer_df['Recency'] = (reference_date - customer_df['LastPurchaseDate']).dt.days
        
        # Calculate customer tenure (days since first purchase)
        customer_df['Tenure'] = (reference_date - customer_df['FirstPurchaseDate']).dt.days
        
        # Fill NaN in StdOrderValue with 0 (for customers with single transaction)
        customer_df['StdOrderValue'] = customer_df['StdOrderValue'].fillna(0)
        
        logger.info(f"Aggregated data for {len(customer_df)} customers")
        return customer_df
    
    def prepare_for_modeling(
        self, 
        df: pd.DataFrame,
        target_column: str = 'TotalRevenue'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training by separating features and target.
        
        Args:
            df (pd.DataFrame): Preprocessed customer-level DataFrame
            target_column (str): Name of the target variable column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (Features DataFrame, Target Series)
        """
        # Define columns to exclude from features
        exclude_columns = [
            'CustomerID', 'FirstPurchaseDate', 'LastPurchaseDate', 
            'Country', target_column
        ]
        
        # Select feature columns
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_columns)} features")
        return X, y
    
    def save_processed_data(
        self, 
        df: pd.DataFrame, 
        output_path: Optional[str] = None
    ) -> str:
        """
        Save processed data to CSV file.
        
        Args:
            df (pd.DataFrame): Processed DataFrame to save
            output_path (Optional[str]): Output file path
            
        Returns:
            str: Path where the file was saved
        """
        if output_path is None:
            output_path = self.config.get('data', {}).get(
                'processed_data_path', 
                'data/processed/processed_data.csv'
            )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
        return str(output_path)
    
    def get_preprocessing_stats(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate statistics about the preprocessing transformation.
        
        Args:
            df_before (pd.DataFrame): DataFrame before preprocessing
            df_after (pd.DataFrame): DataFrame after preprocessing
            
        Returns:
            Dict[str, Any]: Preprocessing statistics
        """
        stats = {
            'rows_before': len(df_before),
            'rows_after': len(df_after),
            'rows_removed': len(df_before) - len(df_after),
            'removal_percentage': (len(df_before) - len(df_after)) / len(df_before) * 100,
            'columns_before': len(df_before.columns),
            'columns_after': len(df_after.columns),
            'unique_customers_before': df_before[self.col_customer].nunique() if self.col_customer in df_before.columns else None,
            'unique_customers_after': df_after[self.col_customer].nunique() if self.col_customer in df_after.columns else None,
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader, generate_sample_data
    
    # Generate sample data
    sample_df = generate_sample_data(n_records=5000)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Clean data
    cleaned_df = preprocessor.clean_data(sample_df)
    
    # Aggregate by customer
    customer_df = preprocessor.aggregate_by_customer(cleaned_df)
    
    # Prepare for modeling
    X, y = preprocessor.prepare_for_modeling(customer_df)
    
    print("\nPreprocessing Results:")
    print(f"Cleaned transactions: {len(cleaned_df)}")
    print(f"Unique customers: {len(customer_df)}")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")