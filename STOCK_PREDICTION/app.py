import flask
from flask import Flask, render_template
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import plotly
import plotly.express as px
import json 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request
from io import BytesIO
import base64
import plotly
import plotly.express as px
import datetime

app = Flask(__name__)

finviz_url = 'https://finviz.com/quote.ashx?t='

def get_news(ticker):
    url = finviz_url + ticker
    req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}) 
    response = urlopen(req)    
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response)
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')
    return news_table

# parse news into dataframe
def parse_news(news_table):
    parsed_news = []
    for x in news_table.findAll('tr'):
        text = x.a.get_text()
        date_scrape = x.td.text.split()
        if len(date_scrape) == 1:
            time = date_scrape[0]
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        # Handle the "Today" case
        if date == 'Today':
            date = pd.to_datetime('today').strftime('%Y-%m-%d')
        parsed_news.append([date, time, text])
    columns = ['date', 'time', 'headline']
    parsed_news_df = pd.DataFrame(parsed_news, columns=columns)
    # Create a pandas datetime object from the strings in 'date' and 'time' column
    parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])
    return parsed_news_df

        
def score_news(parsed_news_df):
    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()
    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()
    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)
    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')
    parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')
    parsed_and_scored_news = parsed_and_scored_news.drop(['date', 'time'], 1)    
    parsed_and_scored_news = parsed_and_scored_news.rename(columns={"compound": "sentiment_score"})
    return parsed_and_scored_news

def plot_hourly_sentiment(parsed_and_scored_news, ticker):
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.resample('H').mean()
    # Plot a bar chart with plotly
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Hourly Sentiment Scores')
    return fig # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later

def plot_daily_sentiment(parsed_and_scored_news, ticker):
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.resample('D').mean()
    # Plot a bar chart with plotly
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Daily Sentiment Scores')
    return fig # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later

model = load_model('Stock Predictions Model.keras')

def load_data(stock, start, end):
    data = yf.download(stock, start, end)
    return data

def preprocess_data(data):
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    scaler = MinMaxScaler(feature_range=(0, 1))
    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    return data_test_scale, scaler

def plot_moving_averages(data, ma_50_days, ma_100_days, ma_200_days):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 18))
    ax1.plot(ma_50_days, 'r', label='MA50')
    ax1.plot(data.Close, 'g', label='Close Price')
    ax1.set_title('Price vs MA50')
    ax1.legend()
    ax2.plot(ma_50_days, 'r', label='MA50')
    ax2.plot(ma_100_days, 'b', label='MA100')
    ax2.plot(data.Close, 'g', label='Close Price')
    ax2.set_title('Price vs MA50 vs MA100')
    ax2.legend()
    ax3.plot(ma_100_days, 'r', label='MA100')
    ax3.plot(ma_200_days, 'b', label='MA200')
    ax3.plot(data.Close, 'g', label='Close Price')
    ax3.set_title('Price vs MA100 vs MA200')
    ax3.legend()
    return fig

def make_predictions(model, data_test_scale, scaler):
    x, y = [], []
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])
    x, y = np.array(x), np.array(y)
    predict = model.predict(x)
    predict = scaler.inverse_transform(predict)
    y = scaler.inverse_transform(y.reshape(-1, 1))
    return predict, y.flatten()

def plot_predictions(predict, y):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(predict, 'r', label='Predicted Price')
    ax.plot(y, 'g', label='Original Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title('Original Price vs Predicted Price')
    ax.set_ylim(min(y), max(y))
    ax.legend()
    return fig
def predict_future_prices_with_dates(model, scaler, latest_data, num_future_days=15):
    future_dates = [datetime.date.today() + datetime.timedelta(days=i) for i in range(1, num_future_days + 1)]

    # Scale the latest data
    latest_data_scaled = scaler.transform(latest_data.reshape(-1, 1))

    # Prepare input data for prediction
    x_input = np.array([latest_data_scaled[-100:].flatten()])

    # Reshape the input data for LSTM layer
    x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

    # Make predictions for the next 'num_future_days' days
    future_predictions = []
    for _ in range(num_future_days):
        next_day_prediction = model.predict(x_input)[0, 0]
        future_predictions.append(next_day_prediction)
        x_input = np.roll(x_input, shift=-1, axis=1)
        x_input[0, -1, 0] = next_day_prediction

    # Inverse transform the predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    return future_dates, future_predictions.flatten()



def plot_future_predictions(future_dates, future_predictions):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(future_dates, future_predictions, 'r', marker='o', label='Predicted Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Predicted Prices for the Next 15 Days')
    ax.legend()
    return fig


# def predict_max_returns_for_tickers(tickers):
#     max_returns = []
#     for ticker in tickers:
#         start_date = '2013-01-01'
#         end_date = '2023-11-30'

#         # Load data and preprocess
#         data = load_data(ticker, start_date, end_date)
#         data_test_scale, scaler = preprocess_data(data)

#         # Make predictions for the next 15 days
#         latest_data = data.Close.values[-100:]
#         _, future_predictions = predict_future_prices_with_dates(model, scaler, latest_data, num_future_days=15)

#         # Calculate the maximum return
#         max_return = (future_predictions[-1] - latest_data[-1]) / latest_data[-1] * 100
#         max_returns.append((ticker, max_return))

#     # Sort tickers based on max returns in descending order
#     sorted_tickers = [ticker for ticker, _ in sorted(max_returns, key=lambda x: x[1], reverse=True)]
#     return sorted_tickers


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment', methods = ['POST'])
def sentiment():
	ticker = flask.request.form['ticker'].upper()
	news_table = get_news(ticker)
	parsed_news_df = parse_news(news_table)
	parsed_and_scored_news = score_news(parsed_news_df)
	fig_hourly = plot_hourly_sentiment(parsed_and_scored_news, ticker)
	fig_daily = plot_daily_sentiment(parsed_and_scored_news, ticker)
	graphJSON_hourly = json.dumps(fig_hourly, cls=plotly.utils.PlotlyJSONEncoder)
	graphJSON_daily = json.dumps(fig_daily, cls=plotly.utils.PlotlyJSONEncoder)
	header= "Hourly and Daily Sentiment of {} Stock".format(ticker)
	description = """
	The above chart averages the sentiment scores of {} stock hourly and daily.
	The table below gives each of the most recent headlines of the stock and the negative, neutral, positive and an aggregated sentiment score.
	The news headlines are obtained from the FinViz website.
	Sentiments are given by the nltk.sentiment.vader Python library.
    """.format(ticker)
	return render_template('sentiment.html',graphJSON_hourly=graphJSON_hourly, graphJSON_daily=graphJSON_daily, header=header,table=parsed_and_scored_news.to_html(classes='data'),description=description)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        ticker = flask.request.form['ticker'].upper()
    else:
        ticker = 'GOOGL'
    start = '2013-01-01'
    end = '2023-11-30'
    data = load_data(ticker, start, end)
    data_test_scale, scaler = preprocess_data(data)
    ma_50_days = data.Close.rolling(50).mean()
    ma_100_days = data.Close.rolling(100).mean()
    ma_200_days = data.Close.rolling(200).mean()
    fig_ma = plot_moving_averages(data, ma_50_days, ma_100_days, ma_200_days)
    predict, y = make_predictions(model, data_test_scale, scaler)
    fig_pred = plot_predictions(predict, y)
    # Get only the last 30 rows of the data
    data_last_30_rows = data.tail(20).to_html()
    # Predict future prices for the next 15 days with date labels
    latest_data = data.Close.values[-100:]
    future_dates, future_predictions = predict_future_prices_with_dates(model, scaler, latest_data, num_future_days=15)
    # Plot the future predictions
    fig_future_pred = plot_future_predictions(future_dates, future_predictions)
    img_future_pred = BytesIO()
    fig_future_pred.savefig(img_future_pred, format='png')
    img_future_pred.seek(0)
    img_future_pred_base64 = base64.b64encode(img_future_pred.getvalue()).decode('utf-8')
    img_ma = BytesIO()
    fig_ma.savefig(img_ma, format='png')
    img_ma.seek(0)
    img_ma_base64 = base64.b64encode(img_ma.getvalue()).decode('utf-8')
    img_pred = BytesIO()
    fig_pred.savefig(img_pred, format='png')
    img_pred.seek(0)
    img_pred_base64 = base64.b64encode(img_pred.getvalue()).decode('utf-8')
    return render_template('prediction.html', ticker=ticker, data=data_last_30_rows, fig_ma=img_ma_base64, fig_pred=img_pred_base64, fig_future_pred=img_future_pred_base64)

# @app.route('/max_returns', methods=['GET'])
# def max_returns():
#     # Example tickers, replace with your own list
#     tickers_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

#     # Get tickers in order of max returns
#     sorted_tickers = predict_max_returns_for_tickers(tickers_list)

#     return render_template('max_returns.html', sorted_tickers=sorted_tickers)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
