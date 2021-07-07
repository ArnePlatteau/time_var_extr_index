# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:22:11 2021

@author: arnep
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error 
from methods_extremal_index import non_parametric_extremal_index, non_parametric_day_based
from methods_extremal_index import horizon_rolling_estimator, theta_sliding_blocks
from pathlib import Path
import pickle 
from scipy.stats import invweibull


"""Simulator class"""
class theta_simulator():
    """
    Object containing all necessary data for simulations of the performance of
    extremal index estimators. The initialisation
    creates the data on which the estimation will be done and decides the 
    criterion to compare performance. Then, by specifying an estimator with all
    relevant parameters, the extremal index will be estimated for each sample.
    The performance is then calculated based on the criterion function. 
    Each object can only hold one simulation data model settings, but on this 
    data, several estimators can be used.
    
    
    Initialisation parameters:
        name: name of the object
        n: the length of each simulation run
        runs: number of simulation runs
        func_gt: ground truth function for the extremal index (is the same for each run)
        func_rand: function to generate random samples
        func_crit: function which determines the criterion used for the comparison
        of different options (standard MSE)
        kwargs: other elements needed for one of the functions
    """

    def __init__(self, name, n, runs,  func_gt, func_rand, func_crit, **kwargs):
        """
        initialise the object with data given, and make empty dicts for the 
        output from the simulations.
        """
        self.__name__ = name
        self.sim_df = func_rand(n, runs, **kwargs)
        self.ground_truth = pd.DataFrame(func_gt(n, **kwargs))
        self.res = pd.DataFrame()
        self.nan = pd.DataFrame()
        self.theta= dict()
        self.avg_theta= dict()
        self.gt_name = func_gt.__name__
        self.rand_name = func_rand.__name__
        self.crit_name = func_crit.__name__
        self.func_crit = func_crit
        self.est_kwargs = {}
        self.est_func = {}


    def append_sim_array(self, runs, func_rand, **kwargs):
        """
        Add runs to the simulation array. Can be useful for doing a first simulation
        on a few runs and then expanding it after EDA.
        """

        #check to see if same data generating function is used
        if func_rand.__name__ != self.rand_name:
            print('warning: other function used than before')
            
        #make array to append
        append = func_rand(len(self.sim_df), runs, **kwargs)
        
        #check the last index and rename the append array column names
        last_idx = self.sim_df.columns.max()
        append.columns = pd.RangeIndex(last_idx + 1,last_idx +1+ len(append.columns), step = 1 )
        
        #merge the original simulation data with the new
        self.sim_df = self.sim_df.merge(append, left_index=True, right_index=True)
   
    
    def simulate(self, func_est, name =None , redo_from_scratch= False,  **kwargs):
        """
        Estimate the extremal index on each run. Can be interrupted and later
        be continued.
        func_est : estimator function used
        name : name of the simulation
        redo_from_scratch : if False, the estimation will continue from where 
        it was left. If True, it will be redone completely
        kwargs : arguments for the estimator
        """
        
        #set name of the simulation
        if name is not None:
            func_name = name
        else: 
            func_name = func_est.__name__
            
        #get the columns for which to estimate the extremal index, default all of them
        sim_column = self.sim_df.columns
        
        #if there is already a part for which it was estimated, these can be left out
        if (func_name in self.est_func.keys())&(not redo_from_scratch) :   
            sim_column = self.sim_df.columns[~self.sim_df.columns.isin(self.theta[func_name])]
            
            #in case the length of the simulation df doesn't match the results df
            #it will be appended so to match it.
            if len(self.sim_df.columns) != len(self.res):
                new_array = np.empty((len(sim_column), self.res.shape[1]))
                new_array[:]= np.nan
                new_df = pd.DataFrame(new_array,  columns = self.res.columns)
                self.res = self.res.append(new_df, ignore_index  = True)
                new_df = pd.DataFrame(new_array,  columns = self.nan.columns)
                self.nan = self.nan.append(new_df, ignore_index  = True)

        else: 
            #in case the length of the simulation df doesn't match the results df
            #it will be appended so to match it.
            if len(self.sim_df.columns) != len(self.res):
                new_array = np.empty((len(sim_column) - len(self.res), self.res.shape[1]))
                new_array[:]= np.nan

                new_df = pd.DataFrame(new_array,  columns = self.res.columns)
                self.res = self.res.append(new_df, ignore_index  = True)
                new_df = pd.DataFrame(new_array,  columns = self.nan.columns)
                self.nan = self.nan.append(new_df, ignore_index  = True)
                
            #add the name of the simulation to all result dataframes
            self.res[func_name] = np.empty(len(sim_column))
            self.nan[func_name] = np.empty(len(sim_column))
            self.theta[func_name]  = pd.DataFrame()
            self.est_kwargs[func_name] = kwargs
            self.est_func[func_name] = func_est

        #simulation: go through all prespecified columns and estimate the extremal index
        for col in sim_column:
            self.theta[func_name][col] = func_est(self.sim_df[col],**kwargs)
            
        #calculate the performance measures after simulation has been performed
        self.calculate_criterion(func_name)
        self.calc_stats()
        self.avg_theta[func_name] = np.mean(self.theta[func_name], axis = 1)
        
        
    def plot_theta_gt(self,  name , color = 'grey', alpha = 0.3, fn = None, 
                      save = False):
        """Plot of the sample average, the density and ground_truth."""
        if save:
            #set path.
            Path('pictures/simulation/' + self.__name__).mkdir(parents=True, exist_ok=True)

        #initialise plot
        plt.figure(figsize=(10, 6), dpi=200)
        
        #loop over all columns and plot them
        for col in self.theta[name].columns:
            plt.plot(self.theta[name][col], color = color, alpha = alpha)
            
        #layout parameters
        plt.grid(color = 'black')
        plt.plot(self.ground_truth, color = 'darkred', label = 'Ground truth')
        plt.plot(self.avg_theta[name], color = 'navy', label = 'Sample average')
        plt.yticks(np.arange(0, 1.2, .2))

        plt.legend()
        plt.title(name)
        
        #function to prevent problems with the filename (special characters)
        if fn is None:
            filename = 'pictures/simulation/' + self.__name__ + "/" + self.__name__ + ' ' + name + '.png'
            filename = filename.replace(':', '')
            filename = filename.replace('\\', '')
            filename = filename.replace('$', '')
        else:
            filename = 'pictures/simulation/' + self.__name__ + "/" + self.__name__ + ' ' + fn + '.png'
            filename = filename.replace(':', '')
            filename = filename.replace('\\', '')
            filename = filename.replace('$', '')
            
        #save and plot
        if save:
            plt.savefig(filename)
            print(filename)
        plt.show()
        
        
    def plot_gt(self, name, fn = None, save = False):
        """Plot of the ground_truth."""
        if save:
            #set path
            Path('pictures/simulation/' + self.__name__).mkdir(parents=True, exist_ok=True)
    
        #initialise plt and set the layout
        plt.figure(figsize=(10, 6), dpi=200)
        plt.grid(color = 'black')
        plt.plot(self.ground_truth, color = 'darkred')
        plt.yticks(np.arange(0, 1.2, .2))

        plt.title("$theta$ for " + name)
        
        #replace characters to prevent problems in the filename
        name = name.replace(':', '')
        name = name.replace('\\', '')
        name = name.replace('$', '')

        if fn is None:
            filename = 'pictures/simulation/' + self.__name__ + "/" + self.__name__ + name +' ground_truth.png'
            filename = filename.replace(':', '')
            filename = filename.replace('\\', '')
            filename = filename.replace('$', '')
        else:
            filename = 'pictures/simulation/' + self.__name__ + "/" + self.__name__ + fn +' ground_truth.png'
            filename = filename.replace(':', '')
            filename = filename.replace('\\', '')
            filename = filename.replace('$', '')
        #save and plot
        if save:
            plt.savefig(filename)
            print(filename)
        plt.show()

    def plot_sim_example(self, name, fn = None, save = False):
        
        if save:
            #check path
            Path('pictures/simulation/' + self.__name__).mkdir(parents=True, exist_ok=True)

        #initialise picture and set layout things
        plt.figure(figsize=(10, 6), dpi=200)
        plt.grid(color = 'black')
        plt.plot(self.sim_df[0])
        plt.title(name + ': sample')
        
        #replace characters to prevent problems in the filename
        if fn is None:
            filename ='pictures/simulation/' + self.__name__ + "/" + self.__name__ + name +' sim_example.png'
            filename = filename.replace(':', '')
            filename = filename.replace('\\', '')
            filename = filename.replace('$', '')
        else:
            filename ='pictures/simulation/' + self.__name__ + "/" + self.__name__ + fn +' sim_example.png'
            filename = filename.replace(':', '')
            filename = filename.replace('\\', '')
            filename = filename.replace('$', '')
            
        #save and plot
        if save:
            plt.savefig(filename)
            print(filename)
        plt.show()

    def calc_stats(self):
        """Description of basic statistics """
        self.crit_stats = self.res.describe()  
        self.crit_stats.loc['Sample size', ]= len(self.sim_df)
        
    def set_first_theta_nan(self, theta_name, number):
        """
        Set the first number of estimators to NAN. Used when estimator undefined
        for a certain time period.
        """
        self.theta[theta_name][:number] = np.nan
    
    
    def calculate_criterion(self, theta_name):
        """ 
        Calculate the criterion function.
        """
        #get the columns and initialise result arrays
        sim_column = self.sim_df.columns
        res = np.empty(len(sim_column))
        nnan = np.empty(len(sim_column))

        #loop over each run, and calculate the criterion. Also keeps tracks of how
        #many NANs in the sample;
        for col in sim_column:
            res[col], nnan[col] = self.func_crit(self.ground_truth[0], self.theta[theta_name][col])

        #add results and nans to the result dict
        self.res[theta_name] = res
        self.nan[theta_name] = nnan


    def fill_calc(self):
        """
        Function to do all estimations previously undone. Used in case 
        the size of the simulation array increases, or in case the process 
        was interrupted.
        """
        
        #go through all estimators currently available
        for est in self.est_func.keys():
            print(self.est_func[est].__name__)
            
            #simulate whatever hasn't been simulated yet.
            self.simulate(self.est_func[est], est, **self.est_kwargs[est])
            
            #calculate criterion and statistics
            self.calculate_criterion(est)
        self.calc_stats()
        

"""Functions to determine the ground truth and random values"""

def random_ARt(n, runs, dof, phi ):
    """returns an array of random samples, generated by an AR(1) model with
    student-t distributed errors
    
    PARAMETERS:
    n : number of rows
    runs : number of columns
    dof : degrees of freedom of the student)t distribution
    phi : autoregressive coefficient 
    """
    
    #generate errors
    eps = np.random.standard_t(dof, size = (n, runs))
    
    ### create the ARt series
    ARt =  np.zeros((n,runs))
    ARt[0] = eps[0]
    for i in range(1,len(ARt)):
        ARt[i] = phi[i]*ARt[i-1]+eps[i]
        
    #return data
    return pd.DataFrame(ARt)


def ground_truth_ARt(n, dof, phi):
    """Determine the ground truth for the extremal index for data generated by
    an AR(1) model with student-t distributed errors
    
    PARAMETERS:
    n : number of rows
    dof : degrees of freedom of the student)t distribution
    phi : autoregressive coefficient 
    """
    
    gt =  1- np.abs(phi)**dof
    gt[0] = 1
    
    
    return gt


def random_MaxAR(n, runs, phi):
    """returns an array of random samples, generated by an MAxAR model with
    inverse-weibull distributed errors
    
    PARAMETERS:
    n : number of rows
    runs : number of columns
    phi : autoregressive coefficient 
    """
    
    #generate errors
    eps = invweibull.rvs(1, size = (n, runs) )
    
    ### create the ARt series
    y =  np.zeros((n,runs))
    y[0] = eps[0]/(1-phi[0])
    for i in range(1,len(y)):
        y[i] = np.maximum(phi[i]*y[i-1], eps[i])
    
    #return data
    return pd.DataFrame(y)


def ground_truth_MaxAR(n, phi):
    """Determine the ground truth for the extremal index for data generated by
    an MAxAR model with inverse-weibull distributed errors
    
    PARAMETERS:
    n : number of rows
    phi : autoregressive coefficient 
    """

    gt =  1- phi
    return gt

def random_wn(n, runs):
    """returns an array of white noise samples
    
    PARAMETERS:
    n : number of rows
    runs : number of columns
    """

    return pd.DataFrame(np.random.normal(size=(n,runs)))
    
def ground_truth_wn(n):
    """Determine the ground truth for white noise
    
    PARAMETERS:
    n : number of rows
    """

    return pd.DataFrame(np.ones(n))


def ground_truth_fixed(n, dof, phi):
    """Dummy function which takes phi as extremal index. only used to have 
    a similar function compared to the other ground truth functions.
    
    PARAMETERS:
    n : dummy value
    dof : dummy value
    phi : ground truth
    """

    return pd.DataFrame(phi)

def random_ARCH(n, runs, om, alpha, beta ):
    """returns an array of random samples, generated by an ARCH model
    
    PARAMETERS:
    n : number of rows
    runs : number of columns
    om : constant
    alpha : autoregressive component
    beta : error component
    """
    
    #generate error terms
    eps = np.random.normal(size = (n, runs))
    
    ### create the ARt series
    y =  np.zeros((n,runs))
    sig = np.zeros((n,runs))

    #initialise the series
    sig[0] = om/(1 - alpha - beta)
    y[0] = sig[0]* eps[0]

    #generate the samples
    for i in range(1,len(y)):
        sig[i] = om + alpha* y[i-1]**2 + beta*sig[i-1]
        y[i] = np.sqrt(sig[i])*eps[i]
        
    #return the data
    return pd.DataFrame(y)


def gt_ARCH_a25_b70(n, om, alpha, beta):
    """Determine the ground truth for an ARCH model with a = .25 and b = .7
    
    PARAMETERS:
    n : number of rows
    om : omega
    alpha : a
    beta : b
    """

    gt = np.ones(n)*0.447
    gt[0] = 1
    return gt

    


"""Estimator functions"""
"""WARNING!!! parametric may suffer from bugs if the order of coefs is changed """

def est_ARt_parametric(data, func_filter, theta_init, bounds, func_mle ):
    """
    Function which estimates a parametric AR(1) with student-t error filter on 
    each data column.
    
    PARAMETERS:
    data : simulated data
    func_filter : filter function to use for estimation
    theta_init : initialisation of the filter
    bounds : parameter bounds for the filter
    func_mle : parameter estimation function of the filter
    
    """
    
    #get the data y, and lagged data for the filter in X
    y = data[1:]
    X = data.shift(1)[1:]
    
    #Estimate the filter model
    results = func_mle(X, y, func_filter, bounds, theta_init)
    
    #get the parameter estimates
    theta = results.x
    
    #based on the filter, get the filtered values of the process
    psi_hat, mu_hat , sigma_hat = func_filter(theta, y, X)
    
    #get the filtered values in an array
    psi = np.empty(len(data))
    psi[:] = np.nan
    psi[1:] = psi_hat
    
    #calculate extremal index estimates
    return pd.Series(1 - abs(np.tanh(psi))**theta[3])


def est_non_parametr_stable(data, **kwargs):
    """
    Function which estimates the stable non-parametric estimator of Cai (2019) 
    by the non_parametric_extremal_index function
    
    PARAMETERS:
    data : simulated data
    kwargs : additional arguments for the estimator function    
    """
    
    #get the estimator (stable) in res, and in array form in theta
    res = non_parametric_extremal_index(data,**kwargs)
    theta = res['theta']*np.ones(len(data))
    
    #return estimator for each point in time
    return pd.Series(theta)


def est_non_parametr_stable_d_set(data, **kwargs):
    """
    Function which estimates the stable non-parametric estimator of Cai (2019) 
    by the non_parametric_day_based function. Difference with previous is another
    implementation, which is a bit more efficient when certain conditions are met.
    
    PARAMETERS:
    data : simulated data
    kwargs : additional arguments for the estimator function    
    """

    print(kwargs['k_proposed'])
    res = non_parametric_day_based(data, **kwargs)
    print(res)
    theta = res[0]*np.ones(len(data))
    return pd.Series(theta)


def est_non_parametr_rol_hor(data, **kwargs):
    """
    Function which estimates the rolling horizon non-parametric estimator of Cai (2019) 
    by the non_parametric_day_based function. Di
    
    PARAMETERS:
    data : simulated data
    kwargs : additional arguments for the estimator function    
    """

    res = horizon_rolling_estimator(data,**kwargs)
    theta = np.empty(len(data))
    theta[:] = np.nan
    print(kwargs)
    
    #shift the estimates to be centered
    theta[int(np.round(kwargs['horizon_length']/2 + .01)):-int(np.round(kwargs['horizon_length']/2))] = res['theta']
    return pd.Series(theta)



def est_sliding_blocks_Z(data, **kwargs):
    """Estimation of the Z-estimator of Berghaus & Bucher (2018), and putting it
    in a homogeneous format"""
    res = theta_sliding_blocks(data, **kwargs)
    return pd.Series(np.ones(len(data))*res[0])
    

def est_sliding_blocks_Y(data, **kwargs):
    """Estimation of the Y-estimator of Berghaus & Bucher (2018), and putting it
    in a homogeneous format"""
    res = theta_sliding_blocks(data, **kwargs)
    return pd.Series(np.ones(len(data))*res[1])

    

"""Criterion functions"""
def MSE_miss_handler(y, y_hat):
    """Mean squared error function, which can handle missing values """
    
    #get the indices of the missing values and remove them from y and yhat
    na_idx = y[y.isna()].index.union( y_hat[y_hat.isna()].index)
    y = y[~y.index.isin(na_idx)]
    y_hat = y_hat[~y_hat.index.isin(na_idx)]
    
    #calculate the mse for the non-missing values
    mse = mean_squared_error(y, y_hat)
    
    #get the number of nan values
    number_nan = len(na_idx)
    return mse, number_nan


"""Handling functions """
def save_object(obj, filename):
    """save the object in a pickle file """
    Path(filename).mkdir(parents=True, exist_ok=True)
    pickle.dump(obj, open( filename, "wb" ), pickle.HIGHEST_PROTOCOL)


def output_table(obj):
    """Make a latex table reporting the results of a simulation."""
    #get the name and calculate the stats
    label = obj.__name__
    obj.calc_stats()
    
    #prepare the output
    output = obj.crit_stats[0:1].copy()
    output.loc['sample size',:] =  obj.crit_stats.iloc[8].copy()
    output.loc['$\hat{\theta}$ length',:] = obj.crit_stats.iloc[8].copy() -obj.nan.iloc[0]
    output.loc['MSE',:] =  obj.crit_stats.loc['mean'].copy()
    output.loc['std MSE',:] =  obj.crit_stats.loc['std'].copy()

    output = output.rename(index = {'count' : 'samples'})
    output = np.transpose(output)
    
    #print the output
    print(output.to_latex(float_format="%.4f", escape = False, label = label).replace(".0000", ""))
