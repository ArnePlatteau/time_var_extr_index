# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:37:49 2021

@author: arnep
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def loader(filename = 'data/TX_MILAN_ITALY.txt', time_column = 'date', 
           months = [6,7,8], year_upper_threshold = 2000,
           year_lower_threshold = None):
    """Specific method to create a df which has all the data in there needed
    for analyses. """
    #Get the data
    df = dataloader(filename = filename, time_column = time_column )
    
    #keep the data for the prespecified months
    months = months
    df = season_selector(df, months, time_column = time_column)
    
    #only keep data between the thresholds
    if year_upper_threshold is not None:
        df = df[df.year <= year_upper_threshold]
        
    if year_lower_threshold is not None:
        df = df[df.year >= year_lower_threshold]

    #return prepared df
    return df

def datetime_converter(df, time_column):
    """Convert time column to datetime """
    temp = df.copy()
    temp[time_column] =  pd.to_datetime(temp[time_column],  format='%Y-%m-%d')
  
    return temp


def date_converter(df, time_column):
    """Convert the time column to date instead of datetime"""
    temp = df.copy()
    temp[time_column] =  pd.to_datetime(temp[time_column],  format='%Y-%m-%d').dt.date

    return temp


def dataloader(filename = 'data/TX_MILAN_ITALY.txt', time_column = 'date' ):
    """function reading in the respective csv file. Converts to datetime and then to
    date, as to obtain functionality of datetime for a few operations, and then 
    have it in date (reason is that datetime assumes an hour, whereas the date 
    only contains days"""
    df = pd.read_csv(filename)
    df = datetime_converter(df, time_column)
    df = date_converter(df, time_column)
    return df


def season_selector(df, months, time_column = 'date'):
    """function to select a number of months """
    #change the type of the date column to datetime
    temp = datetime_converter(df, time_column)
    
    #obtain the year
    temp['year']= temp.date.dt.year
    
    #select based on the respective months given
    temp = temp[temp.date.dt.month.isin(months)]
    
    #convert again to have date not in datetime anymore
    temp = date_converter(temp, time_column)

    return temp


def latex_table(df, table_name):
    """function which prints a latex table with a description of all  """
    descriptives_date = df[['date']].describe(include='all', ).loc[['first', 'last']].rename(columns = {'date' : table_name})
    descriptives_TX = df[['TX']].describe().rename(columns = {'TX' : table_name})
    
    descriptives = descriptives_date.append(descriptives_TX)
    latex = descriptives.to_latex()
    latex = latex.replace(".00", "")
    latex = latex.replace("00:00:00", "")
    print(latex)
    
    
    
def rename_adf_cols(df):
    df = df.rename(columns = {0 : 'test statistic',
                              1 : 'p-value',
                              2 : 'number of lags',
                              3 : 'observations used in regression',
                              5 : 'AIC'
                              })
    return df


def plot_adf_hist(data, name, title, ticks = True , bins = 50):
    plt.figure(figsize=(10, 6), dpi=200)
    if ticks: 
        plt.xticks(np.arange(0,1.05,0.05))
    plt.hist(data, bins)
    plt.title(title)
    plt.savefig('pictures/hist_adf_' + name + '.png')
    plt.show()
    
    
def dick_ful_sum(df, print_ = True, period = None, AIC = False):
    df = pd.DataFrame(df).transpose()
    if AIC:
        df = df[[0,1,2,3,5]].apply(pd.to_numeric)

    else:
        df = df[[0,1,2,3]].apply(pd.to_numeric)

    if period is not None:
        df['period'] = period
        cols = df.columns.to_list()
        cols = cols[-1:] + cols[:-1]
        df = df[cols] 
    df = rename_adf_cols(df)
    df.index.name ='Deterministic components'

    if print_:
        print(df.to_latex(float_format="%.4f").replace(".0000", ""))
    return df


