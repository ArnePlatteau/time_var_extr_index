# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:50:04 2021

@author: arnep

To do's : 
    -years are independent
    -handle missing values
"""

import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scipy
import scipy.stats as st


def indicator_function_ei(X_i, M, X_nk_n):
    """Returns 1 if M is smaller or equal than X_nk_n and if X_nk_n is smaller
    than X_i, and 0 otherwise.""" 
    return 1*(M <= X_nk_n < X_i)



def non_parametric_extremal_index_d_known_yearly(data, d, k, year_column, value_column):
    """Calculates the extremal index estimator of Cai (2019) assuming d is known
    An adaptation is made, to account for the fact that consecutive observations 
    belonging to different years are independent. 
    """
    temp = data.copy()
    #accumulator for indicator function
    sum_indicator_functions = 0 
    I =[]
    #get k-largest value
    X_nk_n = np.sort(temp[value_column])[-k-1]
    
    #effective k
    k_eff = len(temp[temp[value_column] > X_nk_n])
    #go through each year
    for i in temp[year_column].unique():
        #keep data of the respective year, and get the length
        X_year = temp[temp[year_column] == i][value_column]

        X_year = X_year.append(pd.Series(np.ones(d)*(min(temp[value_column])-1)))
        L = len(X_year)
        #loop over the elements (range is again correct as starts at 0 and ends
        #at the previous value)
        for j in range(L - d):
            #get the max of the following elements
            M = max(X_year.iloc[range(j + 1, j + d)])
            #calculate the respective indicator and add it to the accumulator
            sum_indicator_functions = sum_indicator_functions + indicator_function_ei(X_year.iloc[j], M, X_nk_n) 

    #theta is the weighted average
    theta = (1/k_eff) *sum_indicator_functions 
    return theta, X_nk_n, k_eff, I, sum_indicator_functions


def non_parametric_extremal_index_d_known(data, d, k):
    """Calculates the extremal index estimator of Cai (2019) assuming d is known
    """
    X = data.copy()
    
    #accumulator for indicator function
    sum_indicator_functions = 0 
    
    #number of observations
    n = len(X)

    #get the kth-largest observation from the sample
    X_nk_n = np.sort(X)[-k-1]
    
    #effective k
    k_eff = len(X[X > X_nk_n])

    #for each element from 1 to n-d + 1 (here the range is correct as we start 
    #at 0 and not do the last one)
    for i in range(n - d + 1):
        #get the maximum in the following observations
        M = max(X.iloc[range(i + 1, i +d)])
        #calculate the indicator function and add the result to the sum
        sum_indicator_functions = sum_indicator_functions + indicator_function_ei(X.iloc[i], M, X_nk_n) 
        print(i)
    #theta is the weighted average
    theta = (1/k) *sum_indicator_functions 
    
    return theta, X_nk_n, k_eff



def non_parametric_day_based(data, d, k_proposed):
    """function which exploits the date structure to increase the speed.
    Instead of looping over all cases, it just selects the values above the 
    threshold, and checks if in a period of d days, a similar value is 
    also included in the dataset. Advantage is that it loops over k values 
    instead of over n."""
    data = data.reset_index().drop(columns = ['index'])
    col_name = data.columns[0]
    X_nk_n = np.sort(data[col_name])[-k_proposed - 1]

    #select values above threshold
    X = data[data[col_name] > X_nk_n].copy().reset_index()
    X = X[~X.isna()]
    #make another column, initialise at 1 (baseline that future closeby max
    #is not in the sample
    
    sum_ = 1
    #loop over the values (except the last one as no more future maxima can be in the dataset)
    for i in range(len(X)-1):
        #if the date of the next one in the sample is smaller than d (+ a small value)
        if X['index'].iloc[i+1]  - X['index'].iloc[i] >=  d - 1 + 0.01:
            sum_ += 1
            
    #determine the effective k (as the k'th value is not necessarily unique,
    #and ignoring this may )
    k_eff = len(X)
    
    #theta is the mean of the indicators
    theta = sum_/k_eff
    theta = sum_/k_proposed
    
    return theta, X_nk_n, k_eff, X['index'][0]


def d_estimator(data, d_u, k, func, **kwargs):
    """Function to estimate d* as the
    min{h max(i between h and d_u): theta(i) - theta(i+1) < 1/sqrt(k)}"""
        
    #initialise array of theta's
    v_theta = np.zeros(d_u-1)
    
    #loop over remaining elements (range is plus 1 as last element not taken)
    for d in range(2, d_u + 1):
        #calculate the value for the actual d
        params = (d, k)
        v_theta[d-2], X_nk_n, k_eff, idx = func(data, *params, **kwargs)
    
    #calculate the theta(i) - theta(i+1)
    delta = -np.diff(v_theta)  
    
    #look at which elements are smaller than 1/sqrt(k)
    smaller = delta < 1/(np.sqrt(k))
    
    #if all elements are smaller than 1/sqrt(k), then d* is the smallest, i.e 2
    if smaller[smaller == False].size == 0:
        d_star = 2
        
    #otherwise, look at the index of the largest false value, substract this from the
    #length + 2 (as indices start at 0 and d starts at 2)
    else:
        d_star = len(smaller) - smaller[::-1].argmin() + 2
    return d_star



def d_estimator_yearly(data, d_u, k, year_column, value_column):
    """Function to estimate d* as the
    min{h max(i between h and d_u): theta(i) - theta(i+1) < 1/sqrt(k)}"""
        
    #initialise array of theta's
    v_theta = np.zeros(d_u-1)
    
    #loop over remaining elements (range is plus 1 as last element not taken)
    for d in range(2, d_u + 1):
        #calculate the value for the actual d
        params = (d, k)
        v_theta[d-2], X_nk_n, k_eff, I, sum_indicator_functions = non_parametric_extremal_index_d_known_yearly(data,
            *params, year_column, value_column)
    
    #calculate the theta(i) - theta(i+1)
    delta = -np.diff(v_theta)  
    
    #look at which elements are smaller than 1/sqrt(k)
    smaller = delta < 1/(np.sqrt(k))
    
    #if all elements are smaller than 1/sqrt(k), then d* is the smallest, i.e 2
    if smaller[smaller == False].size == 0:
        d_star = 2
        
    #otherwise, look at the index of the largest false value, substract this from the
    #length + 2 (as indices start at 0 and d starts at 2)
    else:
        d_star = len(smaller) - smaller[::-1].argmin() + 2
    return d_star


def non_parametric_extremal_index_yearly(data,d_u, k, year_column, value_column):
    """"Function to link the whole non-parametric estimator of Cai (2019) without
    yearly adaptation. First, d* is calculated, then theta based on this d*¨"""
    #estimate d*
    d = d_estimator_yearly(data, d_u, k, year_column, value_column)
    
    #calculate theta based on d*
    theta, X_nk_n, k, I, sum_indicator_functions = non_parametric_extremal_index_d_known_yearly(data, d, k,
                                                                    year_column, value_column)
    
    output = {'theta' : theta,
              'X_nk_n' : X_nk_n,
              'd' : d,
              'k_eff' : k}
    return output


def non_parametric_extremal_index(data, d_u, k, func, **kwargs):
    """"Function to link the whole non-parametric estimator of Cai (2019) without
    yearly adaptation. First, d* is calculated, then theta based on this d*¨"""
    #estimate d*
    d = d_estimator(data, d_u, k, func ,**kwargs)
    
    #calculate theta based on d*
    params = (d, k)

    theta, X_nk_n, k, idx = func(data,*params, **kwargs)
    
    output = {'theta' : theta,
              'X_nk_n' : X_nk_n,
              'd' : d,
              'k_eff' : k,
              'idx' : idx}
    return output


    
def horizon_block_estimator(data, d_u, k, horizon_length, func, **kwargs):
    """Horizon estimator, but in block format: so instead of estimating a rolling
    horizon type of estimator, it is done on separate blocks.
    
    """
    #calculate how many blocks are in the data
    block_number = int(np.floor(len(data)/horizon_length))
    
    #initialise a vector with the length of the block number
    v_theta = np.zeros(block_number)
    X_nk_n = np.zeros(block_number)
    d = np.zeros(block_number)
    k_eff = np.zeros(block_number)

    j = 0
    
    #loop over the blocks
    for i in range(block_number):
        #select a subset of the data
        temp_data = data.iloc[range(i, i +  horizon_length + 1)]
        
        #calculate theta
        result = non_parametric_extremal_index(temp_data, d_u, k, func, **kwargs)
        v_theta[i] = result['theta']
        X_nk_n[i]  = result['X_nk_n']
        d[j]       = result['d']
        k_eff[j]   = result['k_eff']
      
        #adapt j, as it goes in blocks
        j = j + horizon_length

    output =  {'theta' : v_theta, 'X_nk_n' : X_nk_n, 'd': d, 'k_eff' : k_eff}
    return output



def horizon_rolling_estimator(data, d_u, k, horizon_length, func, **kwargs):
    """Function iteratively calculates the nonparametric estimator of Cai (2019)
    on a subset of the data, which is always sliding 1 at the time. As such,
    it is possible to see if the theta is remaining constant in time.
    """
    
    #initialise the vector of theta's (length is the length of the data minus 
    #the horizon)
    v_theta = np.zeros(len(data) - horizon_length)
    X_nk_n = np.zeros(len(data) - horizon_length)
    d = np.zeros(len(data) - horizon_length)
    k_eff = np.zeros(len(data) - horizon_length)
    i = 0
    #looping over all elements in the vector
    while i < len(v_theta):

        #get subset of the data (range is a bit weird but necessary as last 
        # element of range is not taken into consideration)
        temp_data = data.iloc[range(i, i +  horizon_length + 1)]
        
        #calculate theta
        result = non_parametric_extremal_index(temp_data, d_u, k, func, **kwargs)
        v_theta[i] = result['theta']
        X_nk_n[i]  = result['X_nk_n']
        d[i]       = result['d']
        k_eff[i]   = result['k_eff']
        idx= result['idx']

        i += 1
        try:
            i
            #print('element_normal_run'  + str(i + horizon_length + 1) + ': ' +  str(data.iloc[ i +  horizon_length + 1])) 
        except IndexError:
            break

        while idx > 0:
            try:

                if X_nk_n[i-1] > data.iloc[i + horizon_length ]:
                    v_theta[i] = v_theta[i-1]
                    X_nk_n[i]  = X_nk_n[i-1]
                    d[i]       = d[i-1]
                    k_eff[i]   = k_eff[i-1]
                    idx = idx -1
                    i += 1
                else:
                    break
            except IndexError:
                break
                

      #  print(str(i/(len(v_theta))*100) + '% complete')
    output =  {'theta' : v_theta, 'X_nk_n' : X_nk_n, 'd': d, 'k_eff' : k_eff}
    return output


def horizon_rolling_estimator_2(data, d_u, k, horizon_length, func, **kwargs):
    """Function iteratively calculates the nonparametric estimator of Cai (2019)
    on a subset of the data, which is always sliding 1 at the time. As such,
    it is possible to see if the theta is remaining constant in time.
    """
    
    #initialise the vector of theta's (length is the length of the data minus 
    #the horizon)
    v_theta = np.zeros(len(data) - horizon_length)
    X_nk_n = np.zeros(len(data) - horizon_length)
    d = np.zeros(len(data) - horizon_length)
    k_eff = np.zeros(len(data) - horizon_length)

    #looping over all elements in the vector
    for i in range(len(v_theta)):
        #get subset of the data (range is a bit weird but necessary as last 
        # element of range is not taken into consideration)
        temp_data = data.iloc[range(i, i +  horizon_length + 1)]
        
        #calculate theta
        result = non_parametric_extremal_index(temp_data, d_u, k, func, **kwargs)
        v_theta[i] = result['theta']
        X_nk_n[i]  = result['X_nk_n']
        d[i]       = result['d']
        k_eff[i]   = result['k_eff']

        print(str(i/(len(v_theta))*100) + '% complete')
    output =  {'theta' : v_theta, 'X_nk_n' : X_nk_n, 'd': d, 'k_eff' : k_eff}
    return output


def horizon_rolling_estimator_yearly(data, d_u, k, horizon_length, year_column, 
                                                      value_column):
    """Function iteratively calculates the nonparametric estimator of Cai (2019)
    on a subset of the data, which is always sliding 1 at the time. As such,
    it is possible to see if the theta is remaining constant in time.
    """
    
    #initialise the vector of theta's (length is the length of the data minus 
    #the horizon)
    v_theta = np.zeros(len(data) - horizon_length)
    X_nk_n = np.zeros(len(data) - horizon_length)
    d = np.zeros(len(data) - horizon_length)
    k_eff = np.zeros(len(data) - horizon_length)

    #looping over all elements in the vector
    for i in range(len(v_theta)):
        #get subset of the data (range is a bit weird but necessary as last 
        # element of range is not taken into consideration)
        
        #calculate theta
        result = non_parametric_extremal_index_yearly(data, d_u, k, year_column, 
                                                      value_column)
        v_theta[i] = result['theta']
        X_nk_n[i]  = result['X_nk_n']
        d[i]       = result['d']
        k_eff[i]   = result['k_eff']
        print(str(i/(len(v_theta)-1)*100) + '% complete')
    output =  {'theta' : v_theta, 'X_nk_n' : X_nk_n, 'd': d, 'k_eff' : k_eff}
    return output


def theta_sliding_blocks(X, bn, return_sum = False):
    """
    Implementation of the sliding block estimators of Berghaus and Bucher (2018)
    """
    #set the length of the sample n
    n = len(X)
    Z = np.zeros(n)
    Y = np.zeros(n)

    #calculate the empirical CDF
    F = ECDF(X)
    j = 0
    #go over the n -bn + 1 sliding blocks 
    for i in X.index[:-(bn)]:
        #get the maximum of Xi, ... Xi+bn -1 (range is a bit weird, as the last 
        #element of range is not taken into consideration)
        M = np.nanmax(X.reindex([*range(i, i + bn)]))

        #calculate the empirical quantile of M
        N = F(M)
        
        #calculate Z and Y, defined in BB 2018
        Z[j] = bn*(1 - N)
        Y[j] = -bn*np.log(N)
        j += 1
        
    #weight is given in BB 2018
    weight = 1/(n - bn + 1)
    
    #given in BB 2018, theta is the inverse of the weighted sum.
    theta_Z = (weight*np.sum(Z))**(-1)
    theta_Y = (weight*np.sum(Y))**(-1)
    
    if return_sum:
        return theta_Z, theta_Y, Z, Y
    else:
        return theta_Z, theta_Y
    
    
def construct_psi_k2(theta, y, X, kappa = 30):
    """
    Kappa-based filter for time-varying autoregressive component, based on 
    Platteau (2021)
    """
    #get parameter vector
    T = len(y)
    omega = theta[0]
    alpha = theta[1]
    beta =  theta[2]

    #Filter Volatility
    psi = np.zeros(T)
    
    #initialize volatility at unconditional variance
    psi[0] = omega/(1-alpha)  
    
    #initialise the regression filter values
    t = 0
    xylist = [X.iloc[t]*(y.iloc[t])]
    x2list = [X.iloc[t]**2]
    xysum = sum(xylist)
    x2sum = sum(x2list)
    
    #do the first filtering
    psi[t+1] = omega + (alpha )*(psi[t]) + (beta)*(np.tanh(psi[t])  - xysum/ x2sum )

    #Continue filtering, as long as kappa not reached, use all available elements
    for t in range(1,kappa):
        xylist.append(X.iloc[t]*(y.iloc[t]))
        x2list.append(X.iloc[t]**2)

        xysum = sum(xylist)
        x2sum = sum(x2list)

        psi[t+1] = omega + (alpha )*(psi[t]) + (beta)*(np.tanh(psi[t])  - xysum/ x2sum )
        
    #When kappa is reached, also drop the first instance in each iteration
    for t in range(kappa -1,T-1):
        xylist.append(X.iloc[t]*(y.iloc[t]))
        x2list.append(X.iloc[t]**2)

        xylist.pop(0)
        x2list.pop(0)
        xysum = sum(xylist)
        x2sum = sum(x2list)

        psi[t+1] = omega + (alpha )*(psi[t]) + (beta)*(np.tanh(psi[t])  - xysum/ x2sum )

    #return the autoregressive component
    return psi, 0, 1


def construct_psi_robust_k2(theta, y, X):
    """
    Robust filter for time-varying autoregressive component, based on 
    Platteau (2021)
    """

    T = len(y)
    omega = theta[0]
    alpha = theta[1]
    beta =  theta[2]

    # Filter Volatility
    psi = np.zeros(T)
    
    #initialize volatility at unconditional variance
    psi[0] = omega/(1-alpha)  
    
    ## Calculate Log Likelihood Values
    xylist = []
    x2list = []
    
    #go through the whole sequence. Iteratively filter the autoregressive component.  
    #always based on the last observations only.
    for t in range(0,T-1):            
        xylist.append(X.iloc[t]*(y.iloc[t] ))
        x2list.append(X.iloc[t]**2)

        xysum = sum(xylist)
        x2sum = sum(x2list)
        xylist.pop(0)
        x2list.pop(0)
        psi[t+1] = omega + (alpha )*(psi[t]) + (beta)*np.tanh(np.tanh(psi[t])- xysum/ x2sum) 
        
    return psi, 0, 1
    
    
def llik_ARtfilter(theta, y, X, fun):
    """ 
    Calculate the log likelihood function for an AR(1) filter with student-t 
    distributed errors, given coefficients
    
    PARAMETERS:
    theta : parameter vector of the ARt(1) filter
    y : dependent data
    X : independent data
    fun : filter function
    """
    #get the student-t parameter
    nu = theta[3]
    
    #get time-varying elemtns
    psi, mu, sigma = fun(theta, y, X)
    
    #construct sequence of log lik contributions
    l = np.log(st.t.pdf( y - np.tanh(psi)*(X ) , nu)+ 1e-20)

    # mean log likelihood, negative because MLE minimises
    return -np.mean(l)


def MLE(X, y, fun, bounds, theta_init, llik_fun = llik_ARtfilter,
        options = {'eps':1e-06, 'disp': True, 'maxiter':150}, method = 'L-BFGS-B' ):
    """
    Maximum likelihood optimizer for filter-based criteria 
    
    PARAMETERS:
    X : independent data
    y : dependent data
    fun : filter criterion
    bounds : parameter bounds of the estimation. Should be compact set
    theta_init : initialisation of parameter vector of the ARt(1) filter
    llik_fun : log likelihood function
    options : options for the optimize function
    method : optimisation method
    """
    
    #optimize the log likelihood
    results = scipy.optimize.minimize(llik_fun, theta_init, args=(y, X, fun),
                                      options = options,  method=method, 
                                      bounds= bounds   ) #restrictions in parameter space
    
    ## Print Output
    
    print('parameter estimates:')
    print(results.x)
    
    print('log likelihood value:')
    print(results.fun)
    
    print('Successful minimization?')
    print(results.success)
        
    return results

    

"""Helping functions """
def yearly_average(df):
    """
    Get the average temperature per year
    """
    
    df['yearly_avg'] = np.nan
    for year in df.year.unique():
        df.at[df[df.year == year].index ,'yearly_avg'] = df[df.year == year].TX.mean()
    return df


def polyreg(df, y, degree):
    """
    Polynomial regression to remove annual seasonality
    
    PARAMETERS: 
        
    df : dataframe containing the data
    y : the column with the data in the df
    degree : degree of polynomial
    
    """
    #Get the annual feature
    X = np.array([i%365.25 for i in range(0, len(df))]).reshape(-1,1)
    
    #make polynomial features up to the specified degre
    poly = PolynomialFeatures(degree = degree)
    X_poly = poly.fit_transform(X)
    
    #make longer term cycle features
    X = np.array([i%(365.25*5) for i in range(0, len(df))]).reshape(-1,1)
    
    #make the polynomial features up to the specified degree
    poly = PolynomialFeatures(degree = degree)
    X_poly_2 = poly.fit_transform(X)

    #adding the polynomial features together
    X = np.zeros((X_poly.shape[0],X_poly.shape[1]*2+2))
    X[:,0] = np.array([i for i in range(0, len(df))])
    X[:,1] = np.array([i**2 for i in range(0, len(df))])/1000
    X[:,2:X_poly.shape[1]+2] = X_poly
    X[:,X_poly.shape[1]+2:] = X_poly_2

    #fit the polynomials in LS regression
    lin = LinearRegression()
    lin.fit(X, y)
    
    #remove the estimated polynomial trends
    curve = lin.predict(X)
    resid = y - curve
    coef = lin.coef_
    
    #return the coefficients of the LS regression and the detrended values
    return coef, curve, resid
