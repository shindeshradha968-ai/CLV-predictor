# 💰 CLV Predictor - Customer Lifetime Value Prediction

A comprehensive machine learning application for predicting Customer Lifetime Value (CLV) using Linear Regression, Random Forest, and XGBoost algorithms.

## 📋 Table of Contents

* Overview
* Features
* Installation
* Usage
* Project Structure
* Data Requirements
* Models
* API Reference
* Testing
* Contributing
* License

## 🎯 Overview

CLV Predictor is a full-stack machine learning application that helps businesses:

* Predict the lifetime value of their customers
* Segment customers based on their predicted value
* Generate actionable insights for marketing and retention strategies

## ✨ Features

* **Data Processing**: Automated data cleaning and preprocessing
* **Feature Engineering**: RFM analysis, behavioral features, and time-based features
* **Multiple ML Models**: Linear Regression, Random Forest, and XGBoost
* **Model Evaluation**: Comprehensive metrics and visualizations
* **Interactive UI**: Streamlit-based web application
* **Customer Segmentation**: Automatic segment assignment with recommendations
* **Export Capabilities**: Download predictions and models

## 🚀 Installation

### Prerequisites

* Python 3.8 or higher
* pip package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/clv_predictor.git
cd clv_predictor
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## 📊 Usage

* Upload your dataset through the Streamlit interface
* Perform data preprocessing and feature engineering
* Train machine learning models
* Evaluate model performance
* Generate customer lifetime value predictions

## 📁 Project Structure

```
clv_predictor/
│── data/
│── models/
│── notebooks/
│── src/
│── app.py
│── requirements.txt
│── README.md
```

## 📌 Data Requirements

* Customer transaction data
* Purchase history
* Frequency and recency information

## 🤖 Models

* Linear Regression
* Random Forest
* XGBoost

## 🧪 Testing

Run tests using:

```bash
pytest
```

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

## 📄 License

This project is licensed under the MIT License.
