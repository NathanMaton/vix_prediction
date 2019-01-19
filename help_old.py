import pandas as pd
import requests
from bs4 import BeautifulSoup
import pickle

def strip_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    return df

def create_datetimes(df):
    fmt = '%m/%d/%Y%H:%M:%S'
    df['datetimes'] = df['date'] + df['time']
    df['datetimes'] = pd.to_datetime(df['datetimes'], format=fmt)
    dates = pd.DatetimeIndex(df['datetimes'])
    df['day'] = dates.day
    return df

def scrape_movies(url,save_csv=False):
    """Takes the https://www.boxofficemojo.com/yearly/chart/?yr=2013&p=.htm
    url and turns the main chart into a DataFrameself.

    Can also set save_csv=True if you want it to also save the file to a csv.
    """
    page = requests.get("https://www.boxofficemojo.com/yearly/chart/?yr=2013&p=.htm")
    soup=BeautifulSoup(page.content,"html.parser")
    rows = soup.find_all("tr")
    output_columns = ['rank', 'title', 'studio', 'totalGross', 'grossTheaters', 'openingGross',
                      'openingTheaters', 'open', 'close']
    cleaned_rows = []
    #rows[9].find_all("td")[0].get_text()
    for j in rows[9:109]:
        row = []
        [row.append(i.get_text()) for i in j.find_all("td")]
        cleaned_rows.append(row)
    movies2013 = pd.DataFrame(cleaned_rows,columns=output_columns)
    if save_csv == True:
        movies2013.to_csv('2013_movies.csv')
    return movies2013

def get_movie_links(url, pickle=False):
    page = requests.get("https://www.boxofficemojo.com/yearly/chart/?yr=2013&p=.htm")
    soup=BeautifulSoup(page.content,"html.parser")
    rows = soup.find_all("a")
    hrefs = []
    for i in rows[51:349]:
        hrefs.append(i.get('href'))
    res = []
    for i in range(0,len(hrefs),3):
        res.append(hrefs[i])
    hrefs_list = []
    for i in res:
        hrefs_list.append('https://www.boxofficemojo.com'+i)
    if pickle == True:
        pickling_on = open("hrefs_list","wb")
        pickle.dump(hrefs_list, pickling_on)
        pickling_on.close()
    return hrefs_list
