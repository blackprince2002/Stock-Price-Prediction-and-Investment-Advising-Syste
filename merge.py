import os
import praw
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import yfinance as yf
from datetime import datetime, timedelta
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Check if cleaned merged data file exists
file_path = 'cleaned_merged_data.csv'

# If the file exists, load it directly
if os.path.exists(file_path):
    print("Cleaned merged data file exists. Loading the file...")
    data = pd.read_csv(file_path)
else:
    # If the file does not exist, perform data collection and processing

    # Download the NLTK VADER lexicon and stopwords
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

    sid = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Reddit API credentials
    client_id = 'CxKbMI01jpV10H7YNz-XOQ'
    client_secret = 'lewa8HuAdTvzKfAoUlAf1AbWYIctJg'
    user_agent = 'stock analysis script by /u/Plenty-Calendar-7738'

    # Initialize Reddit instance
    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)

    # Function to fetch Reddit posts for multiple companies within a specific timeframe
    def fetch_reddit_posts(subreddit, queries, start_date, end_date, limit=1000):
        all_posts = []
        for query in queries:
            subreddit_instance = reddit.subreddit(subreddit)
            for submission in subreddit_instance.search(query, time_filter='all', limit=limit):
                created_utc = datetime.utcfromtimestamp(submission.created_utc)
                if start_date <= created_utc <= end_date:
                    all_posts.append({
                        'title': submission.title,
                        'selftext': submission.selftext,
                        'created_utc': submission.created_utc,
                        'query': query
                    })
        return all_posts

    # List of companies (stock tickers) to analyze
    companies = ['AAPL', 'TSLA', 'AMZN', 'MSFT', 'GOOGL']

    # Define the start and end dates for fetching Reddit posts and stock data
    start_date_reddit = datetime(2010, 1, 1)
    end_date_reddit = datetime(2024, 7, 1)

    # Fetch Reddit posts
    reddit_posts = fetch_reddit_posts('stocks', companies, start_date_reddit, end_date_reddit, limit=1000)

    # Check if any posts were fetched
    if not reddit_posts:
        print("No Reddit posts were fetched. Exiting.")
        exit()

    # Convert to DataFrame
    reddit_df = pd.DataFrame(reddit_posts)

    # Remove duplicates
    reddit_df.drop_duplicates(subset=['title', 'selftext', 'created_utc', 'query'], inplace=True)

    # Extract relevant columns
    reddit_df = reddit_df[['title', 'selftext', 'created_utc', 'query']]

    # Combine title and selftext
    reddit_df['text'] = reddit_df['title'] + ' ' + reddit_df['selftext']

    # Clean text data
    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        words = text.split()  # Split into words
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        words = [stemmer.stem(word) for word in words]  # Apply stemming
        cleaned_text = ' '.join(words)  # Rejoin words
        return cleaned_text

    reddit_df['text'] = reddit_df['text'].apply(clean_text)

    # Perform sentiment analysis
    reddit_df['sentiment'] = reddit_df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])

    # Handle missing values in sentiment (if any)
    reddit_df['sentiment'].fillna(0, inplace=True)

    # Convert created_utc to datetime
    reddit_df['date'] = pd.to_datetime(reddit_df['created_utc'], unit='s').dt.date

    # Save Reddit posts with sentiment scores to a JSON file
    reddit_df.to_json('reddit_posts_with_sentiment.json', orient='records')

    # Initialize an empty DataFrame to store stock data
    all_stock_data = pd.DataFrame()

    # Fetch stock data for each company from 2010-01-01 to 2024-07-01
    start_date = '2010-01-01'
    end_date = '2024-07-01'
    for company in companies:
        stock_data = yf.download(company, start=start_date, end=end_date)
        stock_data['company'] = company
        stock_data['date'] = stock_data.index.date
        all_stock_data = pd.concat([all_stock_data, stock_data])

    # Remove duplicates in stock data
    all_stock_data.drop_duplicates(subset=['company', 'date'], inplace=True)

    # Handle missing values in stock data
    all_stock_data.fillna(method='ffill', inplace=True)  # Forward fill missing values

    # Merge Reddit sentiment data with stock data based on date and company
    merged_df = pd.merge(all_stock_data, reddit_df, left_on=['date', 'company'], right_on=['date', 'query'], how='left')

    # Create historical sentiment features
    def create_historical_features(df, window_size=7):
        # Step 1: Fill missing sentiment values with zero
        df['sentiment'].fillna(0, inplace=True)

        # Step 2: Forward fill
        df['sentiment'].fillna(method='ffill', inplace=True)

        # Step 3: Backward fill
        df['sentiment'].fillna(method='bfill', inplace=True)

        # Step 4: Interpolate
        df['sentiment'].interpolate(method='linear', inplace=True)

        # Create rolling mean and standard deviation features
        df['sentiment_rolling_mean'] = df.groupby('company')['sentiment'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        df['sentiment_rolling_std'] = df.groupby('company')['sentiment'].transform(lambda x: x.rolling(window=window_size, min_periods=1).std())

        # Fill NaN values that may result from rolling calculations
        df['sentiment_rolling_std'].fillna(0, inplace=True)  # Standard deviation can be zero for a single observation

        return df

    # Apply feature engineering
    merged_df = create_historical_features(merged_df)

    # Drop unnecessary columns if they exist
    columns_to_drop = ['title', 'selftext', 'created_utc', 'query']
    merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns], inplace=True)

    # Verify Data Types
    print("Data Types:")
    print(merged_df.dtypes)

    # Inspect the Data
    print("Data Inspection:")
    print(merged_df.head())
    print(merged_df.describe())

    # Save the Data
    merged_df.to_csv(file_path, index=False)
    print(f"Data saved to '{file_path}'")

    # Load the Cleaned Data
    data = pd.read_csv(file_path)

# Verify Data Types after loading cleaned data
print("Data Types after loading cleaned data:")
print(data.dtypes)

# Inspect the Data after loading cleaned data
print("Data Inspection after loading cleaned data:")
print(data.head())
print(data.describe())

# Feature Engineering (if any additional features are needed)
# For example, you can create lag features for stock prices
def create_lag_features(df, lag=5):
    for i in range(1, lag+1):
        df[f'Close_lag_{i}'] = df.groupby('company')['Close'].shift(i)
    return df

data = create_lag_features(data, lag=5)

# Handle missing values again after creating lag features
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# Select Features and Target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment', 'sentiment_rolling_mean', 'sentiment_rolling_std',
            'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_4', 'Close_lag_5']
target = 'Close'

# Split the Data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training or Loading
model_file = 'stock_prediction_model.pkl'
if os.path.exists(model_file):
    # Load the pre-trained model
    model = joblib.load(model_file)
    print(f"Model loaded from '{model_file}'")
else:
    # Train a new model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model for future use
    joblib.dump(model, model_file)
    print(f"Model saved as '{model_file}'")

# Model Evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
print(f'Training RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Function to predict stock price for a specific company and provide investment advice
def predict_and_advice(company, date, model, data, features):
    # Function to get the next available trading day
    def get_next_trading_day(date):
        next_day = date + timedelta(days=1)
        while next_day not in data['date'].values:
            next_day += timedelta(days=1)
        return next_day

    # Filter data for the specific company and date
    company_data = data[(data['company'] == company) & (data['date'] == date)]

    # If no data available for the given date, get the next available trading day
    if company_data.empty:
        date = get_next_trading_day(date)
        company_data = data[(data['company'] == company) & (data['date'] == date)]

    # Prepare input features
    X_input = company_data[features]

    # Predict the stock price
    predicted_price = model.predict(X_input)[0]

    # Get the actual closing price
    actual_price = company_data['Close'].values[0]

    # Provide investment advice based on the prediction
    if predicted_price > actual_price:
        advice = "Buy"
    else:
        advice = "Sell"

    # Display the results
    print(f"Company: {company}")
    print(f"Date: {date}")
    print(f"Actual Closing Price: {actual_price}")
    print(f"Predicted Closing Price: {predicted_price}")
    print(f"Investment Advice: {advice}")

# Example usage of the predict_and_advice function
predict_and_advice('AAPL', datetime(2024, 6, 23).date(), model, data, features)


# STILL WORKING ON PREDICTIONS>>>>>MODEL IS NOT COMPLETED!!!!!!!!!
