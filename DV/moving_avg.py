import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def calculate_moving_average(stock_data, window):
    return stock_data['Close'].rolling(window=window).mean()

def save_to_csv(stock_data, filename):
    stock_data.to_csv(filename)
def plot_stock_data(stock_data, title, moving_avg_20, moving_avg_50):
    plt.figure(figsize=(12, 8))
    plt.plot(stock_data.index, stock_data['Close'], label='Closing Price', color='blue', linewidth=2)
    plt.plot(stock_data.index, moving_avg_20, label='20-day Moving Avg', color='orange', linestyle='--', linewidth=2)
    plt.plot(stock_data.index, moving_avg_50, label='50-day Moving Avg', color='green', linestyle='--', linewidth=2)

    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  
    plt.show()

if __name__ == "__main__":
    start_date = "2012-01-01"
    end_date = "2022-01-01"
    infosys_data = fetch_stock_data("INFY.BO", start_date, end_date)
    moving_avg_20 = calculate_moving_average(infosys_data, 20)
    moving_avg_50 = calculate_moving_average(infosys_data, 50)
    save_to_csv(infosys_data, "Infosys_stock_data.csv")
    plot_stock_data(infosys_data, "Infosys Stock Price with Moving Averages", moving_avg_20, moving_avg_50)
