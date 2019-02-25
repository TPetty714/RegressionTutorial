#Based on sentdex tutorial on youtube "Regression Intro - Practical Machine Learning Tutorial with Python#

import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

def main():
    style.use('ggplot')

    df = quandl.get('WIKI/GOOGL')
    df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
    df["HL_PCT"] = ((df['Adj. High'] - df['Adj. Low']) / df['Adj. Close']) * 100.0
    df["PCT_Change"] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100.0

    df = df[['Adj. Close', 'HL_PCT','PCT_Change','Adj. Volume']]

    forecast_col = 'Adj. Close'
    df.fillna(-99999, inplace=True)

    forecast_out = int(math.ceil(0.1*len(df)))

    df['label'] = df[forecast_col].shift(-forecast_out)

    X = np.array(df.drop(['label', 'Adj. Close'], 1))
    X = preprocessing.scale(X)
    X_Lately = X[-forecast_out:]
    X = X[:-forecast_out]
    df.dropna(inplace=True)
    Y = np.array(df['label'])

    print(len(X), len(Y))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, Y_train)
    with open('LinearRegression.pickle','wb') as f:
        pickle.dump(clf, f)

    pickle_in = open('LinearRegression.pickle', 'rb')
    clf = pickle.load(pickle_in)
    accuracy = clf.score(X_test, Y_test)

    forecast_set = clf.predict(X_Lately)
    # print(forecast_set, accuracy, forecast_out)

    df['Forecast'] = np.nan
    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

    print(df.tail())

    df['Adj. Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()




if __name__ == "__main__":
    main()
