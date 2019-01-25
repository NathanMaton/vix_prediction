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


### The functions below I ended up taking out of my notebook code


def vix_prime_combine(monthly_vix,prime):
    """Takes in 2 data sources and outputs 1 df"""
    prime.datetime = pd.to_datetime(prime['datetime'])
    prime.set_index('datetime',inplace=True)
    monthly_vix['dt']= monthly_vix.index
    monthly_vix.dt = monthly_vix.dt.apply(lambda dt: dt.replace(day=1))
    monthly_vix.set_index('dt', inplace=True)
    vp_df = monthly_vix.join(prime,how='outer')
    vp_df = vp_df.drop(pd.Timestamp('1900-02-01'))
    vp_df.iloc[1,0] = monthly_vix.vix_close.mean()
    vp_df.iloc[1,1] = float(10)
    return vp_df


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



#got idea to try MAPE from here: https://towardsdatascience.com/implementing-facebook-prophet-efficiently-c241305405a3
def MAPE(y_true,y_pred):
    y_true,y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred) / y_true)) * 100




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
