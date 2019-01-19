import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pickle
import seaborn as sns
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA


def clean_data():
    vix = pd.read_csv('vix_prices.csv')
    prime = pd.read_csv('historic_prime_rates.csv')
    vix['date'] = pd.to_datetime(vix['date'])
    vix.vix_close.replace(np.nan,vix.vix_close.mean(),inplace=True)
    vix.vix_close.isna().sum()
    vix_close = vix.loc[:,['date','vix_close']]
    vix_close.set_index('date',inplace=True)
    weekly_vix = vix_close.resample('W').mean()
    return vix_close, prime, weekly_vix

def dftest(timeseries):
    dftest = ts.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    #Determine rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

def split_data(df):
    #2 years for validation and test, don't want to get too close to 2009.
    df_len = df.shape[0]
    train = df.iloc[:df_len-52*4,:]
    validation = df.iloc[df_len-52*4:df_len-52*2,:]
    test = df.iloc[df_len-52*2:]
    total_len = train.shape[0]+test.shape[0]+validation.shape[0]
    if df_len == total_len:
        return train, validation, test
    else:
        return "Lengths don't match"

def optimize_ar(df, max_p):
    """Takes in timeseries dataframe, outputs optimal p value for ARIMA"""
    aic_res = []
    for i in range(1,max_p):
        model = ARIMA(df, order=(i,0,0))
        model_fit = model.fit()
        aic_res.append(model_fit.aic)

    plt.hist(aic_res)
    np_aic_res = np.array(aic_res)
    return (np_aic_res.min(),np_aic_res.argmin()+1)
