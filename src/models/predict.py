"""
Prediction Module
=================

This module handles predictions using trained CLV models.

Author: CLV Predictor Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import yaml
import joblib
from loguru import logger
from datetime import datetime
import sys

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")


class CLVPredictor:
    """
    A class to make Customer Lifetime Value predictions using trained models.
    
    This class provides methods for making predictions on new data,
    generating customer segments, and providing actionable insights.
    
    Attributes:
        model: Trained model object
        scaler: Feature scaler
        feature_names (List[str]): Expected feature names
        model_name (str): Name of the loaded model
        
    Example:
        >>> predictor = CLVPredictor()
        >>> predictor.load_model("models/best_model.pkl")
        >>> predictions = predictor.predict(new_data)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the CLVPredictor.
        
        Args:
            model_path (Optional[str]): Path to saved model file
        """
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_name = None
        self.model_metadata = {}
        
        if model_path:
            self.load_model(model_path)
        
        logger.info("CLVPredictor initialized")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data.get('feature_names', [])
        self.model_name = model_data.get('model_name', 'unknown')
        self.model_metadata = model_data.get('training_results', {})
        
        logger.info(f"Model '{self.model_name}' loaded from {model_path}")
    
    def predict(
        self,
        X: pd.DataFrame,
        scale: bool = True
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Features DataFrame
            scale (bool): Whether to scale features using loaded scaler
            
        Returns:
            np.ndarray: Predicted CLV values
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Ensure correct feature order
        X = self._prepare_features(X)
        
        # Scale features if scaler is available
        if scale and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Ensure non-negative predictions (CLV can't be negative)
        predictions = np.maximum(predictions, 0)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features by ensuring correct columns and order.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Prepared features
        """
        if not self.feature_names:
            logger.warning("No feature names stored. Using input features as-is.")
            return X
        
        # Check for missing features
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Filling with 0.")
            for feature in missing_features:
                X[feature] = 0
        
        # Reorder columns to match training
        X = X[self.feature_names]
        
        return X
    
    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Make predictions with confidence intervals.
        
        Note: This is an approximation based on training error.
        For true confidence intervals, use ensemble methods.
        
        Args:
            X (pd.DataFrame): Features DataFrame
            confidence_level (float): Confidence level (e.g., 0.95)
            
        Returns:
            pd.DataFrame: Predictions with confidence intervals
        """
        predictions = self.predict(X)
        
        # Get error metrics from training if available
        train_metrics = self.model_metadata.get('metrics', {})
        std_error = train_metrics.get('rmse', predictions.std() * 0.1)
        
        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower_bound = np.maximum(predictions - z_score * std_error, 0)
        upper_bound = predictions + z_score * std_error
        
        results = pd.DataFrame({
            'Predicted_CLV': predictions,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Confidence_Level': confidence_level
        })
        
        return results
    
    def segment_customers(
        self,
        predictions: np.ndarray,
        customer_ids: Optional[List[str]] = None,
        n_segments: int = 5
    ) -> pd.DataFrame:
        """
        Segment customers based on predicted CLV.
        
        Args:
            predictions (np.ndarray): Predicted CLV values
            customer_ids (Optional[List[str]]): Customer identifiers
            n_segments (int): Number of segments to create
            
        Returns:
            pd.DataFrame: Customer segments
        """
        if customer_ids is None:
            customer_ids = [f"Customer_{i}" for i in range(len(predictions))]
        
        # Create DataFrame
        df = pd.DataFrame({
            'CustomerID': customer_ids,
            'Predicted_CLV': predictions
        })
        
        # Define segment labels
        segment_labels = ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
        if n_segments != 5:
            segment_labels = [f'Segment_{i+1}' for i in range(n_segments)]
        
        # Create segments using quantiles
        df['CLV_Segment'] = pd.qcut(
            df['Predicted_CLV'],
            q=n_segments,
            labels=segment_labels[:n_segments],
            duplicates='drop'
        )
        
        # Add segment statistics
        segment_stats = df.groupby('CLV_Segment')['Predicted_CLV'].agg(['mean', 'min', 'max', 'count'])
        df = df.merge(
            segment_stats.reset_index().rename(columns={
                'mean': 'Segment_Mean_CLV',
                'min': 'Segment_Min_CLV',
                'max': 'Segment_Max_CLV',
                'count': 'Segment_Size'
            }),
            on='CLV_Segment',
            how='left'
        )
        
        # Add percentile rank
        df['CLV_Percentile'] = df['Predicted_CLV'].rank(pct=True) * 100
        
        logger.info(f"Created {n_segments} customer segments")
        return df
    
    def get_recommendations(
        self,
        segment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate actionable recommendations based on customer segments.
        
        Args:
            segment_df (pd.DataFrame): Customer segment DataFrame
            
        Returns:
            pd.DataFrame: Recommendations for each segment
        """
        recommendations = {
            'High': {
                'Strategy': 'Retention & Loyalty',
                'Actions': [
                    'VIP program enrollment',
                    'Exclusive offers and early access',
                    'Personal account manager',
                    'Premium customer service'
                ],
                'Priority': 'Critical'
            },
            'Medium-High': {
                'Strategy': 'Upselling & Cross-selling',
                'Actions': [
                    'Personalized product recommendations',
                    'Loyalty program incentives',
                    'Bundle offers',
                    'Referral program'
                ],
                'Priority': 'High'
            },
            'Medium': {
                'Strategy': 'Engagement & Growth',
                'Actions': [
                    'Targeted email campaigns',
                    'Seasonal promotions',
                    'Product education',
                    'Feedback collection'
                ],
                'Priority': 'Medium'
            },
            'Medium-Low': {
                'Strategy': 'Activation & Re-engagement',
                'Actions': [
                    'Win-back campaigns',
                    'Special discounts',
                    'New product introductions',
                    'Survey for feedback'
                ],
                'Priority': 'Medium'
            },
            'Low': {
                'Strategy': 'Cost-Effective Engagement',
                'Actions': [
                    'Automated email sequences',
                    'Self-service resources',
                    'Community building',
                    'Consider profitability analysis'
                ],
                'Priority': 'Low'
            }
        }
        
        # Add recommendations to segment data
        segment_df['Strategy'] = segment_df['CLV_Segment'].map(
            lambda x: recommendations.get(x, {}).get('Strategy', 'General Engagement')
        )
        segment_df['Priority'] = segment_df['CLV_Segment'].map(
            lambda x: recommendations.get(x, {}).get('Priority', 'Medium')
        )
        segment_df['Recommended_Actions'] = segment_df['CLV_Segment'].map(
            lambda x: ', '.join(recommendations.get(x, {}).get('Actions', ['Standard engagement']))
        )
        
        return segment_df
    
    def batch_predict(
        self,
        X: pd.DataFrame,
        batch_size: int = 1000
    ) -> np.ndarray:
        """
        Make predictions in batches for large datasets.
        
        Args:
            X (pd.DataFrame): Features DataFrame
            batch_size (int): Size of each batch
            
        Returns:
            np.ndarray: All predictions
        """
        n_samples = len(X)
        predictions = []
        
        for i in range(0, n_samples, batch_size):
            batch = X.iloc[i:i+batch_size]
            batch_predictions = self.predict(batch)
            predictions.extend(batch_predictions)
            
            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Processed {min(i + batch_size, n_samples)}/{n_samples} samples")
        
        return np.array(predictions)
    
    def predict_single_customer(
        self,
        customer_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make prediction for a single customer with detailed output.
        
        Args:
            customer_features (Dict[str, Any]): Customer feature dictionary
            
        Returns:
            Dict[str, Any]: Prediction results with insights
        """
        # Convert to DataFrame
        X = pd.DataFrame([customer_features])
        
        # Make prediction
        prediction = self.predict(X)[0]
        
        # Get confidence interval
        conf_df = self.predict_with_confidence(X)
        
        # Determine segment
        # Using approximate segment thresholds
        if prediction > 1000:
            segment = 'High'
        elif prediction > 500:
            segment = 'Medium-High'
        elif prediction > 200:
            segment = 'Medium'
        elif prediction > 50:
            segment = 'Medium-Low'
        else:
            segment = 'Low'
        
        result = {
            'predicted_clv': round(prediction, 2),
            'lower_bound': round(conf_df['Lower_Bound'].iloc[0], 2),
            'upper_bound': round(conf_df['Upper_Bound'].iloc[0], 2),
            'segment': segment,
            'recommendations': self._get_segment_recommendations(segment),
            'model_used': self.model_name,
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _get_segment_recommendations(self, segment: str) -> List[str]:
        """Get recommendations for a segment."""
        recommendations = {
            'High': [
                'Enroll in VIP loyalty program',
                'Assign dedicated account manager',
                'Offer exclusive products/services'
            ],
            'Medium-High': [
                'Target with cross-sell opportunities',
                'Offer loyalty rewards',
                'Invite to referral program'
            ],
            'Medium': [
                'Engage with personalized campaigns',
                'Offer seasonal promotions',
                'Gather feedback for improvement'
            ],
            'Medium-Low': [
                'Re-engagement campaign needed',
                'Offer incentives to increase purchase frequency',
                'Analyze reasons for low engagement'
            ],
            'Low': [
                'Cost-effective communication only',
                'Consider automated engagement',
                'Evaluate customer profitability'
            ]
        }
        return recommendations.get(segment, ['General engagement recommended'])
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        if self.model is None:
            return {'status': 'No model loaded'}
        
        info = {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'has_scaler': self.scaler is not None,
            'training_metrics': self.model_metadata
        }
        
        # Add model-specific info
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        if hasattr(self.model, 'max_depth'):
            info['max_depth'] = self.model.max_depth
        
        return info
    
    def save_predictions(
        self,
        predictions_df: pd.DataFrame,
        output_path: str
    ) -> str:
        """
        Save predictions to a CSV file.
        
        Args:
            predictions_df (pd.DataFrame): Predictions DataFrame
            output_path (str): Output file path
            
        Returns:
            str: Path where predictions were saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        return str(output_path)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    import tempfile
    
    # Create and train a sample model
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Save model
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'model_name': 'test_model',
        'training_results': {}
    }
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        joblib.dump(model_data, f.name)
        model_path = f.name
    
    # Test predictor
    predictor = CLVPredictor(model_path)
    
    # Make predictions
    new_data = pd.DataFrame(np.random.randn(5, 10), columns=[f'feature_{i}' for i in range(10)])
    predictions = predictor.predict(new_data)
    
    print("\nPredictions:")
    print(predictions)
    
    # Get model info
    print("\nModel Info:")
    print(predictor.get_model_info())