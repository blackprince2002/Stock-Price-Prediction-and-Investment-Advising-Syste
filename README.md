# Stock-Price-Prediction-and-Investment-Advising-System

This project aims to predict stock prices using sentiment analysis of Reddit posts and historical stock data. The goal is to provide investment advice based on the predicted stock prices.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Collection](#data-collection)
- [Sentiment Analysis](#sentiment-analysis)
- [Model Training](#model-training)
- [Prediction and Advice](#prediction-and-advice)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Stock price prediction is a challenging task that involves analyzing various factors, including historical stock prices and market sentiment. In this project, we use sentiment analysis of Reddit posts and historical stock data to predict stock prices for a given company. The project aims to provide investment advice based on the predicted stock prices.

## Project Structure

```plaintext
stock-price-prediction/
│
├── data/                         # Directory to store datasets
│   ├── raw/                      # Raw data files
│   └── processed/                # Processed data files
│
├── scripts/                      # Python scripts for data processing, modeling, etc.
│   ├── data_collection.py        # Script to collect Reddit posts and stock data
│   ├── sentiment_analysis.py     # Script to perform sentiment analysis
│   ├── model_training.py         # Script to train the machine learning model
│   └── predict_and_advice.py     # Script to predict stock prices and provide investment advice
│
├── notebooks/                    # Jupyter notebooks for analysis and experimentation
│   └── exploratory_analysis.ipynb
│
├── README.md                     # Project documentation
├── .gitignore                    # Git ignore file
├── requirements.txt              # List of dependencies
└── cleaned_merged_data.csv       # Processed data file
```
## Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt
```
## Usage

1. **Data Collection**: Run the `data_collection.py` script to fetch Reddit posts and stock data.
 
   ```bash
   python scripts/data_collection.py
   
2. **Sentiment Analysis**: Process the fetched data to analyze sentiment.
   
   ```bash
   python scripts/sentiment_analysis.py

3. **Model Training**: Train the model using the processed data.

   ```bash
   python scripts/model_training.py

4. **Prediction and Advice**: Use the trained model to predict stock prices and get investment advice.

   ```bash
   python scripts/predict_and_advice.py


