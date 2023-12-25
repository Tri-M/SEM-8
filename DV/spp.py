import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def save_to_csv(stock_data, filename):
    stock_data.to_csv(filename)

def plot_combined_stock_data(ticker_data, titles):
    plt.figure(figsize=(10, 6))

    colors = ['blue', 'green', 'orange']
    for i, data in enumerate(ticker_data):
        plt.plot(data.index, data['Close'], label=titles[i], color=colors[i])

    plt.title("Stock Prices Over the Decade")
    plt.xlabel('Year')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    start_date = "2012-01-01"
    end_date = "2022-01-01"

    tcs_data = fetch_stock_data("TCS.BO", start_date, end_date)
    infosys_data = fetch_stock_data("INFY.BO", start_date, end_date)
    wipro_data = fetch_stock_data("WIPRO.BO", start_date, end_date)

    save_to_csv(tcs_data, "TCS_stock_data.csv")
    save_to_csv(infosys_data, "Infosys_stock_data.csv")
    save_to_csv(wipro_data, "Wipro_stock_data.csv")

    plot_combined_stock_data([tcs_data, infosys_data, wipro_data], ["TCS", "Infosys", "Wipro"])
