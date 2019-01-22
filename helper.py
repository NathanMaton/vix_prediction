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

def cleanprime(df): ## Used to create clean_prime.csv
    df['MonthYear'] = df['Month'].str.split(',')
    #somehow created MonthDay, would need to re-write this line to re-use this function.
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

def vix_prime_combine(monthly_vix,prime):
    prime.datetime = pd.to_datetime(prime['datetime'])
    prime.set_index('datetime',inplace=True)
    monthly_vix['dt']= monthly_vix.index
    monthly_vix.dt = monthly_vix.dt.apply(lambda dt: dt.replace(day=1))
    monthly_vix.set_index('dt', inplace=True)
    vp_df = monthly_vix.join(prime,how='outer')
    vp_df = vp_df.drop(pd.Timestamp('1900-02-01'))
    vp_df.iloc[1,:] = float(10)
    return vp_df

def clean_data():
    vix = pd.read_csv('vix_prices.csv')
    prime = pd.read_csv('clean_prime.csv')
    vix['date'] = pd.to_datetime(vix['date'])
    vix.vix_close.replace(np.nan,vix.vix_close.mean(),inplace=True)
    vix.vix_close.isna().sum()
    vix_close = vix.loc[:,['date','vix_close']]
    vix_close.set_index('date',inplace=True)
    weekly_vix = vix_close.resample('W').mean()
    monthly_vix = vix_close.resample('M').mean()
    pct = vix_close.vix_close.pct_change()
    pct_df = pd.DataFrame(pct)
    pct_df = pct_df.iloc[1:,:]
    return pct_df, vix_close, prime, weekly_vix, monthly_vix

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

def RMSE(validation_points, prediction_points):
   """
   Calculate RMSE between two vectors
   """
   x = np.array(validation_points)
   y = np.array(prediction_points)

   return np.sqrt(np.mean((x - y)**2))

#got idea to try MAPE from here: https://towardsdatascience.com/implementing-facebook-prophet-efficiently-c241305405a3
def MAPE(y_true,y_pred):
    y_true,y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred) / y_true)) * 100


def split_data(df,time_loop):

    """Splits data into train, validation & test. Designed to get
    #2 years for validation and test, didn't want to get too close to 2009.

    INPUTS:
        df = dataframe to split
        time_loop = either 52 or 12 for weekly or monthly data to equal a year
    OUTPUTS:
        3 dfs with the data split into time ranges
    """

    df_len = df.shape[0]
    train = df.iloc[:df_len-time_loop*4,:]
    validation = df.iloc[df_len-time_loop*4:df_len-time_loop*2,:]
    test = df.iloc[df_len-time_loop*2:]
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

    #plt.hist(aic_res)
    np_aic_res = np.array(aic_res)
    return (np_aic_res.min(),np_aic_res.argmin()+1)

def optimize_ar_rmse(df,val,tot, max_p):
    """Takes in timeseries dataframe, outputs optimal p value for ARIMA"""
    rmse = []
    models = []
    for i in range(1,max_p):
        model = ARIMA(df, order=(i,0,0))
        model_fit = model.fit()
        val['preds'] = model_fit.predict(tot.shape[0]-52*4, tot.shape[0]-52*2, dynamic=False)
        models.append(val)
        score = RMSE(val.vix_close.values,val.preds.values)
        rmse.append(score)

    #plt.plot(rmse)
    np_rmse = np.array(rmse)
    return (models, np_rmse.min(),np_rmse.argmin()+1)

def plot_preds(df,val,tot):
    model = ARIMA(df, order=(13,0,0))
    model_fit = model.fit()
    val['preds'] = model_fit.predict(tot.shape[0]-52*4, tot.shape[0]-52*2, dynamic=False)
    val[['vix_close','preds']].plot()

def format_prophet_data(train,validation):
    p_train = train
    p_train.columns = ['y']
    p_train['ds']= p_train.index
    p_validation = validation
    p_validation['ds']=p_validation.index
    p_validation.columns = ['y','ds']
    return p_train,p_validation
