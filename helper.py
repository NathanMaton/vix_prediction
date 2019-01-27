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
from datetime import datetime


def appendDay(x): ## helper function for clean prime
    if len(x) == 1:
        x.append(1)
    return x

def createstr(cell): ## helper function for clean prime
    nc = []
    for i in cell:
        i=str(i)
        nc.append(i)
    ns = nc[0] + ', ' + nc[1] + ', ' + nc[2]
    return ns

def cleanprime(df):
    """This function cleans up the output of my scrapped prime data
    converting date strings into datetime objects"""
    df['MonthYear'] = df['Month'].str.split(',')
    df['MonthDay'] = df['MonthDay'].apply(appendDay)
    df['DateTime'] = df['MonthDay'] + df['MonthYear']
    df['DateTime'].apply(lambda x: x.insert(2,x.pop()))
    df['DateTime'].values[0] = ['January', '1', '1990', 'January 1']
    df['DateTime'].apply(lambda x: x.pop())
    df['DateTime2'] = df['DateTime']
    df = df.drop(336)
    df['DateTime2'] = df['DateTime2'].apply(createstr)
    df.reset_index(inplace=True)
    df['DateTime2'] = pd.to_datetime(df['DateTime2'])
    df = df.drop(['index','Month','MonthYear','MonthDay','DateTime'],axis=1)
    df['Prime Rate'] =df['Prime Rate'].astype(float)


def clean_data():
    """Performs basic data cleaning/loading tasks on vix & prime data"""
    #reads in files
    vix = pd.read_csv('data/vix_prices.csv')
    prime = pd.read_csv('data/clean_prime.csv')

    #turns prime into percent change & minor cleaning
    prime['prime_rate'] = prime.prime_rate.pct_change()
    prime['prime_rate'] = prime.prime_rate.fillna(value=0)
    prime.iloc[1,0] = '1990-02-01'
    prime['date'] = pd.to_datetime(prime['datetime'], format="%Y-%m-%d")
    prime.set_index('date',inplace=True)
    prime.drop('datetime',axis=1, inplace=True)

    #cleans vix data
    vix['date'] = pd.to_datetime(vix['date'])
    vix.vix_close.replace(np.nan,vix.vix_close.mean(),inplace=True)
    vix_close = vix.loc[:,['date','vix_close']]
    vix_close.set_index('date',inplace=True)
    vix_close = vix_close.sort_index()
    original_vix = vix_close.copy()
    vix_close.vix_close = vix_close.vix_close.pct_change()
    vix_close['key'] = vix_close.index

    #merges data together
    pct_df = pd.merge(vix_close,prime,how='left',
                    left_on=[vix_close.index.year, vix_close.index.month],
                    right_on=[prime.index.year, prime.index.month])


    pct_df = pct_df.set_index('key')
    pct_df.drop(['key_0','key_1'],axis=1,inplace=True)
    pct_df = pct_df.dropna() #drops ~60 of 7000 rows

    #resamples to weekly data to avoid issues with weekend timestamps
    weekly_pct = pct_df.resample('W').mean()
    return weekly_pct, original_vix

def split_data(weekly_pct):
    df_len = weekly_pct.shape[0]
    train_vix = weekly_pct.iloc[:df_len-52*4,0].values
    train_prime = weekly_pct.iloc[:df_len-52*4,1].values
    validation = weekly_pct.iloc[df_len-52*4:df_len-52*2,0].values
    validation_prime = weekly_pct.iloc[df_len-52*4-1:df_len-52*2-1,0].values
    test = weekly_pct.iloc[df_len-52*2:,0]
    return train_vix, train_prime, validation, validation_prime, test

def dftest(timeseries):
    """This code is from a Metis lecture on testing for stationary data with the
    Dickey Fuller test for time series analysis. It takes in a time series and outputs statistics and graphics
    If the p value is <.05, then the data passes the test and is ready for ARMA models."""
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

def RMSE(validation_points, prediction_points):
   """
   Calculate RMSE between two vectors
   """
   x = np.array(validation_points)
   y = np.array(prediction_points)

   return np.sqrt(np.mean((y-x)**2))
