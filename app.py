"""
CLV Predictor - Streamlit Application
=====================================

A comprehensive web application for Customer Lifetime Value prediction
using machine learning models.

Author: CLV Predictor Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import tempfile
import joblib
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import custom modules
from src.data.data_loader import DataLoader, generate_sample_data
from src.data.data_preprocessing import DataPreprocessor
from src.features.feature_builder import FeatureBuilder
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.predict import CLVPredictor

# Page configuration
st.set_page_config(
    page_title="CLV Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'features_data' not in st.session_state:
        st.session_state.features_data = None
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'trainer' not in st.session_state:
        st.session_state.trainer = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">💰 Customer Lifetime Value Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict and analyze customer value using machine learning</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📊 Data Upload", "🔧 Data Processing", "📈 Feature Engineering", 
         "🤖 Model Training", "📋 Model Evaluation", "🎯 Predictions", "📑 Reports"]
    )
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Upload":
        show_data_upload_page()
    elif page == "🔧 Data Processing":
        show_data_processing_page()
    elif page == "📈 Feature Engineering":
        show_feature_engineering_page()
    elif page == "🤖 Model Training":
        show_model_training_page()
    elif page == "📋 Model Evaluation":
        show_model_evaluation_page()
    elif page == "🎯 Predictions":
        show_predictions_page()
    elif page == "📑 Reports":
        show_reports_page()


def show_home_page():
    """Display the home page."""
    st.header("Welcome to CLV Predictor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 What is CLV?
        Customer Lifetime Value (CLV) is the total worth of a customer to a business 
        over the whole period of their relationship. It's an important metric because 
        it costs less to keep existing customers than to acquire new ones.
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Why Predict CLV?
        - Identify high-value customers
        - Optimize marketing spend
        - Improve customer retention
        - Make data-driven decisions
        - Increase profitability
        """)
    
    with col3:
        st.markdown("""
        ### 🤖 Our Approach
        We use three powerful algorithms:
        - **Linear Regression** - Baseline model
        - **Random Forest** - Ensemble method
        - **XGBoost** - Gradient boosting
        """)
    
    st.markdown("---")
    
    # Quick start guide
    st.subheader("🚀 Quick Start Guide")
    
    steps = [
        ("1️⃣", "Upload Data", "Upload your transaction data or use sample data"),
        ("2️⃣", "Process Data", "Clean and preprocess your data"),
        ("3️⃣", "Engineer Features", "Create RFM and behavioral features"),
        ("4️⃣", "Train Models", "Train and compare ML models"),
        ("5️⃣", "Make Predictions", "Predict CLV for your customers")
    ]
    
    cols = st.columns(5)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(desc)
    
    st.markdown("---")
    
    # Sample data option
    st.subheader("📥 Try with Sample Data")
    if st.button("Generate Sample Dataset", type="primary"):
        with st.spinner("Generating sample data..."):
            sample_df = generate_sample_data(n_records=5000)
            st.session_state.raw_data = sample_df
            st.session_state.data_loaded = True
            st.success("✅ Sample data generated successfully! Go to 'Data Upload' to view it.")


def show_data_upload_page():
    """Display the data upload page."""
    st.header("📊 Data Upload")
    
    tab1, tab2 = st.tabs(["Upload CSV", "Use Sample Data"])
    
    with tab1:
        st.subheader("Upload Your Transaction Data")
        
        st.markdown("""
        **Required columns:**
        - `InvoiceNo` - Invoice/transaction identifier
        - `StockCode` - Product code
        - `Description` - Product description
        - `Quantity` - Quantity purchased
        - `InvoiceDate` - Date and time of transaction
        - `UnitPrice` - Price per unit
        - `CustomerID` - Customer identifier
        - `Country` - Customer's country
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.raw_data = df
                st.session_state.data_loaded = True
                st.success(f"✅ Data uploaded successfully! Shape: {df.shape}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with tab2:
        st.subheader("Generate Sample Data")
        
        n_records = st.slider("Number of records", 1000, 50000, 5000, step=1000)
        
        if st.button("Generate Sample Data"):
            with st.spinner("Generating data..."):
                sample_df = generate_sample_data(n_records=n_records)
                st.session_state.raw_data = sample_df
                st.session_state.data_loaded = True
                st.success(f"✅ Generated {n_records} sample records!")
    
    # Display data preview
    if st.session_state.data_loaded and st.session_state.raw_data is not None:
        st.markdown("---")
        st.subheader("Data Preview")
        
        df = st.session_state.raw_data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            if 'CustomerID' in df.columns:
                st.metric("Unique Customers", f"{df['CustomerID'].nunique():,}")
            else:
                st.metric("Unique Customers", "N/A")
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        st.dataframe(df.head(100), use_container_width=True)
        
        # Data quality overview
        st.subheader("Data Quality Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Missing Values**")
            missing = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Missing %': (missing.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        
        with col2:
            st.markdown("**Data Types**")
            dtypes_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.values.astype(str)
            })
            st.dataframe(dtypes_df, use_container_width=True)


def show_data_processing_page():
    """Display the data processing page."""
    st.header("🔧 Data Processing")
    
    if not st.session_state.data_loaded or st.session_state.raw_data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.raw_data
    
    st.subheader("Processing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
        remove_cancelled = st.checkbox("Remove cancelled transactions", value=True)
    
    with col2:
        remove_missing_customers = st.checkbox("Remove rows with missing CustomerID", value=True)
        filter_negative_quantities = st.checkbox("Filter out negative quantities", value=True)
    
    if st.button("Process Data", type="primary"):
        with st.spinner("Processing data..."):
            preprocessor = DataPreprocessor()
            
            # Store original stats
            original_rows = len(df)
            
            # Process data
            cleaned_df = preprocessor.clean_data(df)
            
            # Store in session state
            st.session_state.cleaned_data = cleaned_df
            
            # Show results
            st.success("✅ Data processing complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Rows", f"{original_rows:,}")
            with col2:
                st.metric("Cleaned Rows", f"{len(cleaned_df):,}")
            with col3:
                removed = original_rows - len(cleaned_df)
                st.metric("Rows Removed", f"{removed:,}", delta=f"-{removed/original_rows*100:.1f}%")
    
    # Show cleaned data if available
    if st.session_state.cleaned_data is not None:
        st.markdown("---")
        st.subheader("Cleaned Data Preview")
        st.dataframe(st.session_state.cleaned_data.head(100), use_container_width=True)
        
        # Visualizations
        st.subheader("Data Insights")
        
        cleaned_df = st.session_state.cleaned_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'TotalAmount' in cleaned_df.columns:
                fig = px.histogram(
                    cleaned_df,
                    x='TotalAmount',
                    nbins=50,
                    title='Distribution of Transaction Values'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Country' in cleaned_df.columns:
                country_counts = cleaned_df['Country'].value_counts().head(10)
                fig = px.bar(
                    x=country_counts.index,
                    y=country_counts.values,
                    title='Top 10 Countries by Transactions'
                )
                fig.update_layout(xaxis_title='Country', yaxis_title='Transactions')
                st.plotly_chart(fig, use_container_width=True)


def show_feature_engineering_page():
    """Display the feature engineering page."""
    st.header("📈 Feature Engineering")
    
    if st.session_state.cleaned_data is None:
        st.warning("⚠️ Please process your data first!")
        return
    
    st.subheader("Feature Engineering Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_rfm = st.checkbox("Include RFM Features", value=True)
        include_behavioral = st.checkbox("Include Behavioral Features", value=True)
    
    with col2:
        include_time = st.checkbox("Include Time-based Features", value=True)
        include_cohort = st.checkbox("Include Cohort Features", value=True)
    
    if st.button("Build Features", type="primary"):
        with st.spinner("Building features..."):
            builder = FeatureBuilder()
            
            features_df = builder.build_features(
                st.session_state.cleaned_data,
                include_rfm=include_rfm,
                include_behavioral=include_behavioral,
                include_time=include_time,
                include_cohort=include_cohort
            )
            
            st.session_state.features_data = features_df
            st.session_state.feature_builder = builder
            
            st.success(f"✅ Created {len(features_df.columns) - 1} features for {len(features_df)} customers!")
    
    # Display features
    if st.session_state.features_data is not None:
        st.markdown("---")
        st.subheader("Engineered Features")
        
        features_df = st.session_state.features_data
        
        # Feature statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Customers", f"{len(features_df):,}")
        with col2:
            st.metric("Number of Features", len(features_df.columns) - 1)
        with col3:
            if 'Monetary' in features_df.columns:
                st.metric("Avg CLV", f"${features_df['Monetary'].mean():,.2f}")
        
        st.dataframe(features_df.head(100), use_container_width=True)
        
        # Feature visualizations
        st.subheader("Feature Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Recency' in features_df.columns and 'Frequency' in features_df.columns:
                fig = px.scatter(
                    features_df,
                    x='Recency',
                    y='Frequency',
                    color='Monetary' if 'Monetary' in features_df.columns else None,
                    title='RFM Analysis: Recency vs Frequency'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'CustomerSegment' in features_df.columns:
                segment_counts = features_df['CustomerSegment'].value_counts()
                fig = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title='Customer Segments Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns[:15]
        if len(numeric_cols) > 1:
            corr_matrix = features_df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                title='Feature Correlation Matrix'
            )
            st.plotly_chart(fig, use_container_width=True)


def show_model_training_page():
    """Display the model training page."""
    st.header("🤖 Model Training")
    
    if st.session_state.features_data is None:
        st.warning("⚠️ Please build features first!")
        return
    
    features_df = st.session_state.features_data
    
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_col = st.selectbox(
            "Select Target Variable",
            ['Monetary', 'TotalRevenue'] if 'TotalRevenue' in features_df.columns else ['Monetary'],
            index=0
        )
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        st.markdown("**Models to Train:**")
        train_lr = st.checkbox("Linear Regression", value=True)
        train_rf = st.checkbox("Random Forest", value=True)
        train_xgb = st.checkbox("XGBoost", value=True)
    
    if st.button("Train Models", type="primary"):
        if not any([train_lr, train_rf, train_xgb]):
            st.error("Please select at least one model to train!")
            return
        
        with st.spinner("Training models... This may take a few minutes."):
            # Prepare data
            builder = st.session_state.get('feature_builder', FeatureBuilder())
            X, y = builder.get_feature_importance_ready_data(features_df, target_col=target_col)
            
            # Initialize trainer
            trainer = ModelTrainer()
            
            # Train models
            results = trainer.train_all_models(X, y, test_size=test_size)
            
            st.session_state.trainer = trainer
            st.session_state.models_trained = True
            st.session_state.X = X
            st.session_state.y = y
            
            st.success("✅ Model training complete!")
        
        # Display results
        st.markdown("---")
        st.subheader("Training Results")
        
        summary_df = trainer.get_training_summary()
        st.dataframe(summary_df, use_container_width=True)
        
        # Model comparison chart
        fig = go.Figure()
        
        for i, row in summary_df.iterrows():
            fig.add_trace(go.Bar(
                name=row['Model'],
                x=['Train R²', 'Validation R²'],
                y=[row['Train R²'], row['Validation R²']],
                text=[f"{row['Train R²']:.4f}", f"{row['Validation R²']:.4f}"],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            yaxis_title='R² Score'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance (Best Model)")
        importance_df = trainer.get_feature_importance(top_n=15)
        
        if not importance_df.empty:
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'Top 15 Features - {trainer.best_model_name}'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Save model option
        st.markdown("---")
        if st.button("Save Best Model"):
            model_path = trainer.save_model()
            st.success(f"✅ Model saved to {model_path}")


def show_model_evaluation_page():
    """Display the model evaluation page."""
    st.header("📋 Model Evaluation")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
        return
    
    trainer = st.session_state.trainer
    
    st.subheader("Detailed Evaluation")
    
    model_name = st.selectbox(
        "Select Model to Evaluate",
        list(trainer.models.keys())
    )
    
    if st.button("Evaluate Model"):
        with st.spinner("Evaluating model..."):
            from sklearn.model_selection import train_test_split
            
            X = st.session_state.X
            y = st.session_state.y
            
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale test data
            X_test_scaled = pd.DataFrame(
                trainer.scaler.transform(X_test),
                columns=X_test.columns
            )
            
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_model(
                trainer.models[model_name],
                X_test_scaled,
                y_test,
                model_name
            )
            
            # Display metrics
            st.subheader("Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{metrics['R2']:.4f}")
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.2f}")
            with col3:
                st.metric("MAE", f"{metrics['MAE']:.2f}")
            with col4:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            
            # Plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Predicted vs Actual
                y_pred = trainer.models[model_name].predict(X_test_scaled)
                
                fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title='Predicted vs Actual Values'
                )
                fig.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Residual Distribution
                residuals = y_test - y_pred
                
                fig = px.histogram(
                    x=residuals,
                    nbins=50,
                    title='Residual Distribution',
                    labels={'x': 'Residual', 'y': 'Count'}
                )
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            # Generate report
            report = evaluator.generate_report(model_name)
            
            with st.expander("📄 Full Evaluation Report"):
                st.text(report)


def show_predictions_page():
    """Display the predictions page."""
    st.header("🎯 Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
        return
    
    trainer = st.session_state.trainer
    
    tab1, tab2, tab3 = st.tabs(["Batch Predictions", "Single Customer", "Upload New Data"])
    
    with tab1:
        st.subheader("Batch Predictions")
        
        if st.session_state.features_data is not None:
            features_df = st.session_state.features_data
            
            model_name = st.selectbox(
                "Select Model",
                list(trainer.models.keys()),
                key="batch_model"
            )
            
            if st.button("Generate Predictions", key="batch_predict"):
                with st.spinner("Generating predictions..."):
                    # Prepare features
                    X = st.session_state.X
                    
                    # Scale features
                    X_scaled = trainer.scaler.transform(X)
                    
                    # Make predictions
                    predictions = trainer.models[model_name].predict(X_scaled)
                    predictions = np.maximum(predictions, 0)  # Ensure non-negative
                    
                    # Create results DataFrame
                    results_df = features_df[['CustomerID']].copy()
                    results_df['Predicted_CLV'] = predictions
                    
                    # Add segments
                    results_df['CLV_Segment'] = pd.qcut(
                        results_df['Predicted_CLV'],
                        q=5,
                        labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'],
                        duplicates='drop'
                    )
                    
                    results_df['CLV_Percentile'] = results_df['Predicted_CLV'].rank(pct=True) * 100
                    
                    st.session_state.predictions = results_df
                    
                    st.success(f"✅ Generated predictions for {len(results_df)} customers!")
            
            # Display predictions
            if st.session_state.predictions is not None:
                results_df = st.session_state.predictions
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Predicted CLV", f"${results_df['Predicted_CLV'].mean():,.2f}")
                with col2:
                    st.metric("Total Predicted Value", f"${results_df['Predicted_CLV'].sum():,.2f}")
                with col3:
                    st.metric("Highest CLV", f"${results_df['Predicted_CLV'].max():,.2f}")
                with col4:
                    st.metric("Median CLV", f"${results_df['Predicted_CLV'].median():,.2f}")
                
                # Segment distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    segment_counts = results_df['CLV_Segment'].value_counts()
                    fig = px.pie(
                        values=segment_counts.values,
                        names=segment_counts.index,
                        title='Customer Segment Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    segment_value = results_df.groupby('CLV_Segment')['Predicted_CLV'].sum()
                    fig = px.bar(
                        x=segment_value.index,
                        y=segment_value.values,
                        title='Total Predicted Value by Segment',
                        labels={'x': 'Segment', 'y': 'Total CLV'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # CLV Distribution
                fig = px.histogram(
                    results_df,
                    x='Predicted_CLV',
                    nbins=50,
                    title='Distribution of Predicted CLV',
                    color='CLV_Segment'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.subheader("Prediction Results")
                
                # Filter options
                segment_filter = st.multiselect(
                    "Filter by Segment",
                    results_df['CLV_Segment'].unique().tolist(),
                    default=results_df['CLV_Segment'].unique().tolist()
                )
                
                filtered_df = results_df[results_df['CLV_Segment'].isin(segment_filter)]
                
                st.dataframe(
                    filtered_df.sort_values('Predicted_CLV', ascending=False).head(100),
                    use_container_width=True
                )
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name=f"clv_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with tab2:
        st.subheader("Single Customer Prediction")
        
        st.markdown("Enter customer features to predict their CLV:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
            frequency = st.number_input("Frequency (number of transactions)", min_value=1, value=5)
            monetary = st.number_input("Monetary (average order value)", min_value=0.0, value=100.0)
        
        with col2:
            tenure = st.number_input("Customer Tenure (days)", min_value=0, value=365)
            total_quantity = st.number_input("Total Quantity Purchased", min_value=1, value=50)
            unique_products = st.number_input("Unique Products Purchased", min_value=1, value=10)
        
        with col3:
            avg_days_between = st.number_input("Avg Days Between Purchases", min_value=0.0, value=30.0)
            transaction_count = st.number_input("Total Transactions", min_value=1, value=5)
        
        model_name = st.selectbox(
            "Select Model",
            list(trainer.models.keys()),
            key="single_model"
        )
        
        if st.button("Predict CLV", key="single_predict"):
            with st.spinner("Calculating..."):
                # Create feature vector
                # Note: This is a simplified version - in production, ensure all required features are present
                feature_dict = {
                    'Recency': recency,
                    'Frequency': frequency,
                    'Monetary': monetary,
                    'CustomerTenure': tenure,
                    'TotalQuantity': total_quantity,
                    'UniqueProducts': unique_products,
                    'AvgDaysBetweenPurchases': avg_days_between,
                    'TotalTransactions': transaction_count
                }
                
                # Fill in missing features with median values from training data
                X_train = st.session_state.X
                for col in X_train.columns:
                    if col not in feature_dict:
                        feature_dict[col] = X_train[col].median()
                
                # Create DataFrame with correct column order
                X_new = pd.DataFrame([feature_dict])[X_train.columns]
                
                # Scale features
                X_scaled = trainer.scaler.transform(X_new)
                
                # Predict
                prediction = trainer.models[model_name].predict(X_scaled)[0]
                prediction = max(0, prediction)
                
                # Determine segment
                percentiles = st.session_state.y.quantile([0.2, 0.4, 0.6, 0.8]).values
                if prediction >= percentiles[3]:
                    segment = "High"
                    color = "🟢"
                elif prediction >= percentiles[2]:
                    segment = "Medium-High"
                    color = "🔵"
                elif prediction >= percentiles[1]:
                    segment = "Medium"
                    color = "🟡"
                elif prediction >= percentiles[0]:
                    segment = "Medium-Low"
                    color = "🟠"
                else:
                    segment = "Low"
                    color = "🔴"
                
                # Display result
                st.markdown("---")
                st.subheader("Prediction Result")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted CLV", f"${prediction:,.2f}")
                with col2:
                    st.metric("Customer Segment", f"{color} {segment}")
                with col3:
                    percentile = (st.session_state.y < prediction).mean() * 100
                    st.metric("Percentile Rank", f"{percentile:.1f}%")
                
                # Recommendations
                st.subheader("Recommendations")
                
                recommendations = {
                    'High': [
                        "🌟 Enroll in VIP loyalty program",
                        "🎁 Offer exclusive products and early access",
                        "👤 Assign dedicated account manager",
                        "📞 Schedule regular check-ins"
                    ],
                    'Medium-High': [
                        "📈 Target with upsell opportunities",
                        "🎯 Personalized product recommendations",
                        "💳 Offer premium loyalty tier",
                        "📧 Regular engagement campaigns"
                    ],
                    'Medium': [
                        "📬 Send targeted promotions",
                        "🎉 Seasonal campaign inclusion",
                        "📝 Gather feedback",
                        "🔄 Cross-sell related products"
                    ],
                    'Medium-Low': [
                        "🔔 Re-engagement campaigns",
                        "💰 Offer incentives to increase purchase frequency",
                        "❓ Survey for feedback",
                        "📱 Mobile app promotions"
                    ],
                    'Low': [
                        "📧 Automated email sequences",
                        "💡 Self-service resources",
                        "👥 Community building initiatives",
                        "📊 Monitor for churn risk"
                    ]
                }
                
                for rec in recommendations.get(segment, []):
                    st.markdown(f"- {rec}")
    
    with tab3:
        st.subheader("Upload New Data for Predictions")
        
        st.markdown("""
        Upload a CSV file with customer features to generate CLV predictions.
        The file should contain the same features used for training.
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="new_data")
        
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(new_df)} records")
                
                st.dataframe(new_df.head(10), use_container_width=True)
                
                model_name = st.selectbox(
                    "Select Model",
                    list(trainer.models.keys()),
                    key="new_data_model"
                )
                
                if st.button("Generate Predictions for New Data"):
                    with st.spinner("Processing..."):
                        # Ensure columns match
                        X_train = st.session_state.X
                        
                        # Add missing columns with median values
                        for col in X_train.columns:
                            if col not in new_df.columns:
                                new_df[col] = X_train[col].median()
                        
                        # Select and order columns
                        X_new = new_df[X_train.columns]
                        
                        # Scale
                        X_scaled = trainer.scaler.transform(X_new)
                        
                        # Predict
                        predictions = trainer.models[model_name].predict(X_scaled)
                        predictions = np.maximum(predictions, 0)
                        
                        # Add predictions to DataFrame
                        new_df['Predicted_CLV'] = predictions
                        new_df['CLV_Segment'] = pd.qcut(
                            predictions,
                            q=5,
                            labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'],
                            duplicates='drop'
                        )
                        
                        st.success("✅ Predictions generated!")
                        st.dataframe(new_df, use_container_width=True)
                        
                        # Download
                        csv = new_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="new_predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")


def show_reports_page():
    """Display the reports page."""
    st.header("📑 Reports & Insights")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
        return
    
    tab1, tab2, tab3 = st.tabs(["Executive Summary", "Customer Insights", "Model Comparison"])
    
    with tab1:
        st.subheader("Executive Summary")
        
        if st.session_state.predictions is not None:
            results_df = st.session_state.predictions
            features_df = st.session_state.features_data
            
            # Key metrics
            st.markdown("### 📊 Key Performance Indicators")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Customers",
                    f"{len(results_df):,}"
                )
            
            with col2:
                st.metric(
                    "Total Predicted Revenue",
                    f"${results_df['Predicted_CLV'].sum():,.0f}"
                )
            
            with col3:
                st.metric(
                    "Average CLV",
                    f"${results_df['Predicted_CLV'].mean():,.2f}"
                )
            
            with col4:
                high_value = (results_df['CLV_Segment'] == 'High').sum()
                st.metric(
                    "High-Value Customers",
                    f"{high_value:,}",
                    f"{high_value/len(results_df)*100:.1f}%"
                )
            
            # Segment analysis
            st.markdown("### 📈 Segment Analysis")
            
            segment_analysis = results_df.groupby('CLV_Segment').agg({
                'Predicted_CLV': ['count', 'sum', 'mean', 'min', 'max']
            }).round(2)
            segment_analysis.columns = ['Count', 'Total Value', 'Avg Value', 'Min Value', 'Max Value']
            segment_analysis['% of Customers'] = (segment_analysis['Count'] / segment_analysis['Count'].sum() * 100).round(1)
            segment_analysis['% of Value'] = (segment_analysis['Total Value'] / segment_analysis['Total Value'].sum() * 100).round(1)
            
            st.dataframe(segment_analysis, use_container_width=True)
            
            # Pareto analysis
            st.markdown("### 🎯 Pareto Analysis")
            
            sorted_clv = results_df.sort_values('Predicted_CLV', ascending=False)
            sorted_clv['Cumulative_Value'] = sorted_clv['Predicted_CLV'].cumsum()
            sorted_clv['Cumulative_Percentage'] = sorted_clv['Cumulative_Value'] / sorted_clv['Predicted_CLV'].sum() * 100
            sorted_clv['Customer_Percentage'] = np.arange(1, len(sorted_clv) + 1) / len(sorted_clv) * 100
            
            # Find 80/20 point
            pareto_point = sorted_clv[sorted_clv['Cumulative_Percentage'] >= 80].iloc[0]['Customer_Percentage']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sorted_clv['Customer_Percentage'],
                y=sorted_clv['Cumulative_Percentage'],
                mode='lines',
                name='Cumulative Value %',
                fill='tozeroy'
            ))
            fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80% of Value")
            fig.add_vline(x=pareto_point, line_dash="dash", line_color="red")
            fig.update_layout(
                title=f'Pareto Analysis: {pareto_point:.1f}% of customers generate 80% of value',
                xaxis_title='% of Customers',
                yaxis_title='Cumulative % of Value'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"📌 **Key Insight:** The top {pareto_point:.1f}% of customers are predicted to generate 80% of total customer value.")
        
        else:
            st.info("Generate predictions first to view the executive summary.")
    
    with tab2:
        st.subheader("Customer Insights")
        
        if st.session_state.features_data is not None and st.session_state.predictions is not None:
            features_df = st.session_state.features_data
            results_df = st.session_state.predictions
            
            # Merge for analysis
            analysis_df = features_df.merge(results_df[['CustomerID', 'Predicted_CLV', 'CLV_Segment']], on='CustomerID')
            
            # Top customers
            st.markdown("### 🏆 Top 10 Customers by Predicted CLV")
            top_customers = analysis_df.nlargest(10, 'Predicted_CLV')[
                ['CustomerID', 'Predicted_CLV', 'CLV_Segment', 'Recency', 'Frequency', 'Monetary']
            ]
            st.dataframe(top_customers, use_container_width=True)
            
            # At-risk customers
            st.markdown("### ⚠️ At-Risk High-Value Customers")
            st.caption("High-value customers with high recency (haven't purchased recently)")
            
            if 'Recency' in analysis_df.columns:
                at_risk = analysis_df[
                    (analysis_df['CLV_Segment'].isin(['High', 'Medium-High'])) &
                    (analysis_df['Recency'] > analysis_df['Recency'].quantile(0.75))
                ].nlargest(10, 'Predicted_CLV')[
                    ['CustomerID', 'Predicted_CLV', 'Recency', 'Frequency', 'Monetary']
                ]
                
                if len(at_risk) > 0:
                    st.dataframe(at_risk, use_container_width=True)
                    st.warning(f"⚠️ {len(at_risk)} high-value customers are at risk of churning!")
                else:
                    st.success("✅ No high-value customers are currently at high risk.")
            
            # Segment characteristics
            st.markdown("### 📊 Segment Characteristics")
            
            numeric_cols = ['Recency', 'Frequency', 'Monetary', 'CustomerTenure'] if all(
                col in analysis_df.columns for col in ['Recency', 'Frequency', 'Monetary', 'CustomerTenure']
            ) else analysis_df.select_dtypes(include=[np.number]).columns[:4].tolist()
            
            segment_chars = analysis_df.groupby('CLV_Segment')[numeric_cols].mean().round(2)
            
            # Heatmap
            fig = px.imshow(
                segment_chars.T,
                text_auto='.2f',
                title='Average Feature Values by Segment',
                labels={'x': 'Segment', 'y': 'Feature', 'color': 'Value'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Generate predictions first to view customer insights.")
    
    with tab3:
        st.subheader("Model Comparison")
        
        trainer = st.session_state.trainer
        
        if trainer is not None:
            # Training summary
            summary_df = trainer.get_training_summary()
            
            st.markdown("### 📈 Model Performance Summary")
            st.dataframe(summary_df, use_container_width=True)
            
            # Performance comparison chart
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('R² Scores', 'Training Time')
            )
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, (idx, row) in enumerate(summary_df.iterrows()):
                fig.add_trace(
                    go.Bar(
                        name=row['Model'],
                        x=['Train R²', 'Validation R²'],
                        y=[row['Train R²'], row['Validation R²']],
                        marker_color=colors[i % len(colors)],
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            fig.add_trace(
                go.Bar(
                    x=summary_df['Model'],
                    y=summary_df['Training Time (s)'],
                    marker_color=colors[:len(summary_df)],
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=400, title_text="Model Comparison")
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model details
            st.markdown(f"### 🏆 Best Model: {trainer.best_model_name}")
            
            best_results = trainer.training_results.get(trainer.best_model_name, {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Train R²", f"{best_results.get('train_score', 0):.4f}")
            with col2:
                st.metric("Validation R²", f"{best_results.get('val_score', 0):.4f}")
            with col3:
                st.metric("Training Time", f"{best_results.get('training_time', 0):.2f}s")
            
            # Feature importance comparison
            st.markdown("### 📊 Feature Importance (Best Model)")
            
            importance_df = trainer.get_feature_importance(top_n=15)
            if not importance_df.empty:
                fig = px.bar(
                    importance_df,
                    x='Importance_Normalized',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance (%)',
                    labels={'Importance_Normalized': 'Importance (%)', 'Feature': ''}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Download model
            st.markdown("---")
            st.markdown("### 💾 Export Model")
            
            if st.button("Save and Download Best Model"):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                    model_data = {
                        'model': trainer.best_model,
                        'scaler': trainer.scaler,
                        'feature_names': trainer.feature_names,
                        'model_name': trainer.best_model_name,
                        'training_results': best_results,
                        'saved_at': datetime.now().isoformat()
                    }
                    joblib.dump(model_data, f.name)
                    
                    with open(f.name, 'rb') as file:
                        st.download_button(
                            label="📥 Download Model (.pkl)",
                            data=file,
                            file_name=f"clv_model_{trainer.best_model_name}_{datetime.now().strftime('%Y%m%d')}.pkl",
                            mime="application/octet-stream"
                        )


# Footer
def show_footer():
    """Display footer."""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>CLV Predictor v1.0.0 | Built with Streamlit & Scikit-learn</p>
            <p>© 2024 CLV Predictor Team</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
    show_footer()