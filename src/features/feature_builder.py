"""
Feature Builder Module
======================

This module handles all feature engineering operations for CLV prediction,
including RFM analysis, time-based features, and behavioral features.

Author: CLV Predictor Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import yaml
from loguru import logger
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import sys

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")


class FeatureBuilder:
    """
    A class to build features for Customer Lifetime Value prediction.
    
    This class implements various feature engineering techniques including:
    - RFM (Recency, Frequency, Monetary) analysis
    - Time-based features
    - Behavioral features
    - Cohort-based features
    
    Attributes:
        config (dict): Configuration dictionary
        scaler (StandardScaler): Scaler for numerical features
        encoders (dict): Dictionary of label encoders for categorical features
        
    Example:
        >>> builder = FeatureBuilder()
        >>> features_df = builder.build_features(transaction_df)
        >>> print(features_df.shape)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the FeatureBuilder with configuration settings.
        
        Args:
            config_path (Optional[str]): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.encoders = {}
        self.feature_names = []
        
        logger.info("FeatureBuilder initialized successfully")
    
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
            'features': {
                'rfm': {
                    'recency_weight': 0.25,
                    'frequency_weight': 0.35,
                    'monetary_weight': 0.40
                },
                'aggregation_periods': [30, 60, 90, 180, 365]
            }
        }
    
    def build_features(
        self, 
        df: pd.DataFrame,
        reference_date: Optional[datetime] = None,
        include_rfm: bool = True,
        include_behavioral: bool = True,
        include_time: bool = True,
        include_cohort: bool = True
    ) -> pd.DataFrame:
        """
        Build comprehensive feature set from transaction data.
        
        Args:
            df (pd.DataFrame): Transaction-level DataFrame
            reference_date (Optional[datetime]): Reference date for calculations
            include_rfm (bool): Whether to include RFM features
            include_behavioral (bool): Whether to include behavioral features
            include_time (bool): Whether to include time-based features
            include_cohort (bool): Whether to include cohort features
            
        Returns:
            pd.DataFrame: Customer-level DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Set reference date
        if reference_date is None:
            reference_date = df['InvoiceDate'].max()
        
        # Initialize customer DataFrame
        customer_df = pd.DataFrame({'CustomerID': df['CustomerID'].unique()})
        
        # Build RFM features
        if include_rfm:
            rfm_features = self._build_rfm_features(df, reference_date)
            customer_df = customer_df.merge(rfm_features, on='CustomerID', how='left')
            logger.info("RFM features added")
        
        # Build behavioral features
        if include_behavioral:
            behavioral_features = self._build_behavioral_features(df)
            customer_df = customer_df.merge(behavioral_features, on='CustomerID', how='left')
            logger.info("Behavioral features added")
        
        # Build time-based features
        if include_time:
            time_features = self._build_time_features(df, reference_date)
            customer_df = customer_df.merge(time_features, on='CustomerID', how='left')
            logger.info("Time-based features added")
        
        # Build cohort features
        if include_cohort:
            cohort_features = self._build_cohort_features(df)
            customer_df = customer_df.merge(cohort_features, on='CustomerID', how='left')
            logger.info("Cohort features added")
        
        # Fill any remaining NaN values
        customer_df = self._handle_missing_values(customer_df)

        # Store feature names
        self.feature_names = [col for col in customer_df.columns if col != 'CustomerID']

        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
        return customer_df

    def _safe_qcut(
        self,
        series: pd.Series,
        q: int = 5,
        ascending: bool = True
    ) -> pd.Series:
        """
        Safely apply pd.qcut handling cases where duplicates reduce the number of bins.

        When data has many duplicate values, pd.qcut with duplicates='drop' may create
        fewer bins than requested. This method handles that by dynamically generating
        labels based on the actual number of bins created.

        Args:
            series (pd.Series): The series to bin
            q (int): Number of quantiles to try to create
            ascending (bool): If True, lower values get lower scores (1-5).
                            If False, lower values get higher scores (5-1).

        Returns:
            pd.Series: Series with scores from 1 to 5 (or fewer if bins are dropped)
        """
        try:
            # First, try to create the bins without labels to see how many we get
            _, bins = pd.qcut(series, q=q, retbins=True, duplicates='drop')
            n_bins = len(bins) - 1  # Number of actual bins created

            if n_bins < 1:
                # If we can't create any bins, return a constant score
                return pd.Series([3.0] * len(series), index=series.index)

            # Generate labels based on actual number of bins
            if ascending:
                # For Frequency and Monetary: higher values = higher scores
                labels = list(range(1, n_bins + 1))
            else:
                # For Recency: lower values (more recent) = higher scores
                labels = list(range(n_bins, 0, -1))

            # Apply qcut with the correct number of labels
            result = pd.qcut(series, q=q, labels=labels, duplicates='drop')
            return result.astype(float)

        except ValueError:
            # If qcut fails entirely, use a fallback approach with pd.cut
            try:
                _, bins = pd.cut(series, bins=q, retbins=True, duplicates='drop')
                n_bins = len(bins) - 1

                if n_bins < 1:
                    return pd.Series([3.0] * len(series), index=series.index)

                if ascending:
                    labels = list(range(1, n_bins + 1))
                else:
                    labels = list(range(n_bins, 0, -1))

                result = pd.cut(series, bins=q, labels=labels, duplicates='drop')
                return result.astype(float)
            except ValueError:
                # Last resort: return median score
                return pd.Series([3.0] * len(series), index=series.index)

    def _build_rfm_features(
        self, 
        df: pd.DataFrame,
        reference_date: datetime
    ) -> pd.DataFrame:
        """
        Build RFM (Recency, Frequency, Monetary) features.
        
        Args:
            df (pd.DataFrame): Transaction DataFrame
            reference_date (datetime): Reference date for recency calculation
            
        Returns:
            pd.DataFrame: DataFrame with RFM features
        """
        # Calculate RFM metrics
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalAmount': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        # Calculate RFM scores (1-5 scale using quantiles)
        # Use a helper function to handle cases where duplicates reduce the number of bins
        rfm['RecencyScore'] = self._safe_qcut(rfm['Recency'], q=5, ascending=False)
        rfm['FrequencyScore'] = self._safe_qcut(rfm['Frequency'].rank(method='first'), q=5, ascending=True)
        rfm['MonetaryScore'] = self._safe_qcut(rfm['Monetary'].rank(method='first'), q=5, ascending=True)
        
        # Fill any NaN scores with median
        for col in ['RecencyScore', 'FrequencyScore', 'MonetaryScore']:
            rfm[col] = rfm[col].fillna(rfm[col].median())
        
        # Calculate weighted RFM score
        weights = self.config.get('features', {}).get('rfm', {})
        r_weight = weights.get('recency_weight', 0.25)
        f_weight = weights.get('frequency_weight', 0.35)
        m_weight = weights.get('monetary_weight', 0.40)
        
        rfm['RFMScore'] = (
            rfm['RecencyScore'] * r_weight +
            rfm['FrequencyScore'] * f_weight +
            rfm['MonetaryScore'] * m_weight
        )
        
        # Create RFM segments
        rfm['RFMSegment'] = (
            rfm['RecencyScore'].astype(int).astype(str) +
            rfm['FrequencyScore'].astype(int).astype(str) +
            rfm['MonetaryScore'].astype(int).astype(str)
        )
        
        # Create customer segments based on RFM
        rfm['CustomerSegment'] = rfm.apply(self._assign_customer_segment, axis=1)
        
        # Encode customer segment
        segment_encoder = LabelEncoder()
        rfm['CustomerSegmentEncoded'] = segment_encoder.fit_transform(rfm['CustomerSegment'])
        self.encoders['CustomerSegment'] = segment_encoder
        
        return rfm
    
    def _assign_customer_segment(self, row: pd.Series) -> str:
        """
        Assign customer segment based on RFM scores.
        
        Args:
            row (pd.Series): Row with RFM scores
            
        Returns:
            str: Customer segment name
        """
        r, f, m = row['RecencyScore'], row['FrequencyScore'], row['MonetaryScore']
        
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r <= 2 and f <= 2 and m >= 3:
            return 'Cant Lose'
        elif r <= 2 and f <= 2:
            return 'Lost'
        else:
            return 'Potential Loyalists'
    
    def _build_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build behavioral features from transaction patterns.
        
        Args:
            df (pd.DataFrame): Transaction DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with behavioral features
        """
        # Basic aggregations
        behavioral = df.groupby('CustomerID').agg({
            'TotalAmount': ['mean', 'std', 'min', 'max', 'median'],
            'Quantity': ['sum', 'mean', 'std', 'max'],
            'UnitPrice': ['mean', 'std', 'min', 'max'],
            'StockCode': 'nunique',
            'InvoiceNo': 'nunique',
            'Description': 'nunique'
        }).reset_index()
        
        # Flatten column names
        behavioral.columns = [
            'CustomerID',
            'AvgTransactionValue', 'StdTransactionValue', 'MinTransactionValue',
            'MaxTransactionValue', 'MedianTransactionValue',
            'TotalQuantity', 'AvgQuantity', 'StdQuantity', 'MaxQuantity',
            'AvgUnitPrice', 'StdUnitPrice', 'MinUnitPrice', 'MaxUnitPrice',
            'UniqueProducts', 'TotalTransactions', 'UniqueDescriptions'
        ]
        
        # Calculate derived metrics
        behavioral['TransactionValueRange'] = behavioral['MaxTransactionValue'] - behavioral['MinTransactionValue']
        behavioral['PriceRange'] = behavioral['MaxUnitPrice'] - behavioral['MinUnitPrice']
        behavioral['ProductDiversity'] = behavioral['UniqueProducts'] / behavioral['TotalTransactions']
        behavioral['AvgItemsPerTransaction'] = behavioral['TotalQuantity'] / behavioral['TotalTransactions']
        
        # Calculate coefficient of variation for transaction value
        behavioral['TransactionValueCV'] = np.where(
            behavioral['AvgTransactionValue'] != 0,
            behavioral['StdTransactionValue'] / behavioral['AvgTransactionValue'],
            0
        )
        
        return behavioral
    
    def _build_time_features(
        self, 
        df: pd.DataFrame,
        reference_date: datetime
    ) -> pd.DataFrame:
        """
        Build time-based features from transaction data.
        
        Args:
            df (pd.DataFrame): Transaction DataFrame
            reference_date (datetime): Reference date
            
        Returns:
            pd.DataFrame: DataFrame with time-based features
        """
        time_features = df.groupby('CustomerID').agg({
            'InvoiceDate': ['min', 'max', 'count']
        }).reset_index()
        
        time_features.columns = ['CustomerID', 'FirstPurchase', 'LastPurchase', 'PurchaseCount']
        
        # Calculate tenure and recency
        time_features['CustomerTenure'] = (reference_date - time_features['FirstPurchase']).dt.days
        time_features['DaysSinceLastPurchase'] = (reference_date - time_features['LastPurchase']).dt.days
        
        # Calculate average days between purchases
        purchase_intervals = df.groupby('CustomerID').apply(
            lambda x: x['InvoiceDate'].sort_values().diff().mean().days if len(x) > 1 else 0
        ).reset_index()
        purchase_intervals.columns = ['CustomerID', 'AvgDaysBetweenPurchases']
        
        time_features = time_features.merge(purchase_intervals, on='CustomerID', how='left')
        
        # Calculate purchase velocity (transactions per month of tenure)
        time_features['PurchaseVelocity'] = np.where(
            time_features['CustomerTenure'] > 0,
            time_features['PurchaseCount'] / (time_features['CustomerTenure'] / 30),
            time_features['PurchaseCount']
        )
        
        # Day of week preferences
        dow_features = self._build_dow_features(df)
        time_features = time_features.merge(dow_features, on='CustomerID', how='left')
        
        # Hour preferences
        hour_features = self._build_hour_features(df)
        time_features = time_features.merge(hour_features, on='CustomerID', how='left')
        
        # Drop datetime columns
        time_features = time_features.drop(['FirstPurchase', 'LastPurchase'], axis=1)
        
        return time_features
    
    def _build_dow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build day-of-week preference features.
        
        Args:
            df (pd.DataFrame): Transaction DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with day-of-week features
        """
        # Calculate transactions by day of week
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        
        dow_pivot = df.pivot_table(
            index='CustomerID',
            columns='DayOfWeek',
            values='InvoiceNo',
            aggfunc='count',
            fill_value=0
        ).reset_index()
        
        dow_pivot.columns = ['CustomerID'] + [f'DOW_{i}' for i in range(len(dow_pivot.columns) - 1)]
        
        # Calculate weekend vs weekday preference
        weekend_cols = [col for col in dow_pivot.columns if col in ['DOW_5', 'DOW_6']]
        weekday_cols = [col for col in dow_pivot.columns if col.startswith('DOW_') and col not in weekend_cols]
        
        if weekend_cols and weekday_cols:
            dow_pivot['WeekendRatio'] = (
                dow_pivot[weekend_cols].sum(axis=1) / 
                (dow_pivot[weekday_cols].sum(axis=1) + 1)
            )
        else:
            dow_pivot['WeekendRatio'] = 0
        
        # Get most frequent purchase day
        dow_cols = [col for col in dow_pivot.columns if col.startswith('DOW_')]
        if dow_cols:
            dow_pivot['MostFrequentDay'] = dow_pivot[dow_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(float)
        else:
            dow_pivot['MostFrequentDay'] = 0
        
        return dow_pivot
    
    def _build_hour_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build hour-of-day preference features.
        
        Args:
            df (pd.DataFrame): Transaction DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with hour-based features
        """
        df['Hour'] = df['InvoiceDate'].dt.hour
        
        hour_stats = df.groupby('CustomerID').agg({
            'Hour': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        hour_stats.columns = ['CustomerID', 'AvgPurchaseHour', 'StdPurchaseHour', 
                              'EarliestPurchaseHour', 'LatestPurchaseHour']
        
        # Create time period preferences
        df['TimePeriod'] = pd.cut(
            df['Hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        period_pivot = df.pivot_table(
            index='CustomerID',
            columns='TimePeriod',
            values='InvoiceNo',
            aggfunc='count',
            fill_value=0
        ).reset_index()
        
        # Rename columns
        new_cols = ['CustomerID']
        for col in period_pivot.columns[1:]:
            new_cols.append(f'Period_{col}')
        period_pivot.columns = new_cols
        
        hour_features = hour_stats.merge(period_pivot, on='CustomerID', how='left')
        
        return hour_features
    
    def _build_cohort_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build cohort-based features.
        
        Args:
            df (pd.DataFrame): Transaction DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with cohort features
        """
        # Determine first purchase month for each customer
        first_purchase = df.groupby('CustomerID')['InvoiceDate'].min().reset_index()
        first_purchase.columns = ['CustomerID', 'FirstPurchaseDate']
        first_purchase['CohortMonth'] = first_purchase['FirstPurchaseDate'].dt.to_period('M')
        
        # Calculate cohort statistics
        cohort_stats = df.merge(first_purchase[['CustomerID', 'CohortMonth']], on='CustomerID')
        
        cohort_agg = cohort_stats.groupby('CohortMonth').agg({
            'CustomerID': 'nunique',
            'TotalAmount': 'mean'
        }).reset_index()
        cohort_agg.columns = ['CohortMonth', 'CohortSize', 'CohortAvgSpend']
        
        # Merge cohort features back to customers
        cohort_features = first_purchase[['CustomerID', 'CohortMonth']].merge(
            cohort_agg, on='CohortMonth', how='left'
        )
        
        # Encode cohort month
        cohort_features['CohortMonthEncoded'] = cohort_features['CohortMonth'].astype(str).rank(method='dense').astype(int)
        
        # Drop CohortMonth as it's a Period object
        cohort_features = cohort_features.drop('CohortMonth', axis=1)
        
        return cohort_features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the feature DataFrame.
        
        Args:
            df (pd.DataFrame): Feature DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'CustomerID':
                df[col] = df[col].fillna(df[col].median())
        
        # Fill remaining NaN with 0
        df = df.fillna(0)
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def scale_features(
        self, 
        df: pd.DataFrame,
        method: str = 'standard',
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df (pd.DataFrame): Feature DataFrame
            method (str): Scaling method ('standard' or 'minmax')
            exclude_cols (Optional[List[str]]): Columns to exclude from scaling
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        if exclude_cols is None:
            exclude_cols = ['CustomerID']
        
        # Select columns to scale
        scale_cols = [col for col in df.columns 
                     if col not in exclude_cols and 
                     df[col].dtype in ['int64', 'float64']]
        
        # Choose scaler
        scaler = self.scaler if method == 'standard' else self.minmax_scaler
        
        # Fit and transform
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        
        logger.info(f"Scaled {len(scale_cols)} features using {method} scaling")
        return df
    
    def get_feature_importance_ready_data(
        self,
        features_df: pd.DataFrame,
        target_col: str = 'Monetary'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.
        
        Args:
            features_df (pd.DataFrame): DataFrame with all features
            target_col (str): Name of the target column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X, y) ready for modeling
        """
        # Separate target
        y = features_df[target_col].copy()
        
        # Select feature columns
        exclude_cols = ['CustomerID', target_col, 'RFMSegment', 'CustomerSegment']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols].copy()
        
        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        return X, y
    
    def save_features(
        self, 
        df: pd.DataFrame, 
        output_path: Optional[str] = None
    ) -> str:
        """
        Save engineered features to CSV file.
        
        Args:
            df (pd.DataFrame): Feature DataFrame
            output_path (Optional[str]): Output file path
            
        Returns:
            str: Path where features were saved
        """
        if output_path is None:
            output_path = self.config.get('data', {}).get(
                'features_path',
                'data/processed/features.csv'
            )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")
        
        return str(output_path)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data.data_loader import generate_sample_data
    from data.data_preprocessing import DataPreprocessor
    
    # Generate and preprocess sample data
    sample_df = generate_sample_data(n_records=5000)
    preprocessor = DataPreprocessor()
    cleaned_df = preprocessor.clean_data(sample_df)
    
    # Build features
    builder = FeatureBuilder()
    features_df = builder.build_features(cleaned_df)
    
    print("\nFeature Engineering Results:")
    print(f"Number of customers: {len(features_df)}")
    print(f"Number of features: {len(features_df.columns) - 1}")
    print(f"\nFeature columns: {features_df.columns.tolist()}")