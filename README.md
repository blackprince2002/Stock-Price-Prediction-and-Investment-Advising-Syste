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

## Data Collection

The `data_collection.py` script fetches Reddit posts related to specific companies and their historical stock data. The script performs the following tasks:

1. Downloads the NLTK VADER lexicon and stopwords.
2. Initializes a Reddit instance using PRAW (Python Reddit API Wrapper).
3. Fetches Reddit posts for multiple companies within a specified timeframe.
4. Cleans the text data and performs sentiment analysis.
5. Fetches historical stock data for each company from Yahoo Finance.
6. Merges the Reddit sentiment data with the stock data.
7. Saves the processed data to a CSV file.

## Sentiment Analysis

The `sentiment_analysis.py` script processes the fetched Reddit posts to analyze sentiment. The script performs the following tasks:

1. Combines the title and selftext of each Reddit post.
2. Cleans the text data by removing URLs, punctuation, numbers, and stopwords, and applies stemming.
3. Performs sentiment analysis using the VADER sentiment analysis tool.
4. Saves the processed sentiment scores to a JSON file.

## Model Training

The `model_training.py` script trains a machine learning model to predict stock prices. The script performs the following tasks:

1. Loads the cleaned and merged data.
2. Creates additional features, such as lag features for stock prices.
3. Splits the data into training and testing sets.
4. Trains a RandomForestRegressor model on the training data.
5. Saves the trained model for future use.
6. Evaluates the model using the RMSE metric.

## Prediction and Advice

The `predict_and_advice.py` script uses the trained model to predict stock prices for a specific company and date, and provides investment advice. The script performs the following tasks:

1. Loads the trained model and cleaned data.
2. Filters the data for the specified company and date.
3. Predicts the stock price using the trained model.
4. Provides investment advice based on the predicted stock price.

## Results

The results of the project are as follows:

1. **raining RMSE**: The root mean squared error (RMSE) on the training data.
2. **Test RMSE**: The root mean squared error (RMSE) on the test data.
3. **Investment Advice**: Buy or Sell recommendation based on the predicted stock price.



