from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import preprocessing
from sklearn import utils
import time
import numpy as np
import yfinance as yf
import pandas as pd


def get_data(ticker_list: list):


        data = yf.download(  # or pdr.get_data_yahoo(...
                # tickers list or string as well
                # tickers = "SPY AAPL ^SPX",
                tickers=ticker_list,

                # use "period" instead of start/end
                # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                # (optional, default is '1mo')
                period = "1d",

                # fetch data by interval (including intraday if period < 60 days)
                # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                # (optional, default is '1d')
                interval = "1m",

                # group by ticker (to access via data['SPY'])
                # (optional, default is 'column')
                group_by = 'ticker',

                # adjust all OHLC automatically
                # (optional, default is False)
                auto_adjust = True,

                # download pre/post regular market hours data
                # (optional, default is False)
                prepost = True,

                # use threads for mass downloading? (True/False/Integer)
                # (optional, default is True)
                threads = True,

                # proxy URL scheme use use when downloading?
                # (optional, default is None)
                proxy = None
            )
        return data

if __name__ == "__main__":

        lm = LinearRegression()
        logreg = LogisticRegression()

        unix = lambda time_: int((time.mktime(time_.timetuple())))

        tickers = [
                # "SPY",
                "AAPL",
                "^SPX"
        ]

        data = get_data(tickers)
        cols = ['Open', 'High', 'Close']

        # minutes
        minutes = 15

        intervals_history = 13

        # range (start, stop, interval)
        ranges = range(minutes, minutes*intervals_history, minutes)

        results = []
        for tick in tickers:
                df = data[tick].dropna().reset_index()
                df['Datetime'] = df['Datetime'].map(unix)
                for col in cols:
                        preds = []
                        logreg_preds = []
                        for rng in ranges:
                                whole = df.iloc[-rng:, :]

                                x = whole[['Datetime']]
                                y = whole[[col]]

                                ############ LinearRegression ###########
                                lm.fit(x,y)

                                pred = lm.predict(np.array(int(time.time())).reshape(1, -1))
                                preds.append(pred)

                                results.append({
                                        "tick": tick,
                                        "col": col,
                                        'value': pred[0][0],
                                        'minutes': rng
                                        })

                                ############ LogisticRegression #########

                                # lab = preprocessing.LabelEncoder()
                                # y_transformed = lab.fit_transform(y)
                                # logreg.fit(x,y_transformed)
                                #
                                # log_pred = logreg.predict(np.array(int(time.time())).reshape(1, -1))
                                # logreg_preds.append(log_pred)

                        print(f"{tick} - {col} : AVERAGE / prediciton {np.mean(preds)}")
                        # print(f"{tick} - {col} : AVERAGE / prediciton {np.mean(logreg_preds)}")


        # pred_df = pd.DataFrame(results)