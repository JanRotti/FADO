import sys, sys
import logging 

import numpy as np
import pyDOE2

from scipy.optimize import minimize, fmin_slsqp
from scipy.stats import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt

# PARAMETERS
_numDataPoints = 1000
_numSamples    = 10
_trainingSamples = []
_dataPoints = []
_training_samples_func = None
_training_samples_grad = None
_penalty = None

# Define an uncertain parameter
par = {"name":"MACH", "mu":0.8, "var":0.1, "type":"normal", "marker":"__MACH_NUMBER__"} # Example for normal distribution
par = {"name":"MACH", "lb":0.73, "ub": 0.87, "type":"uniform", "marker":"__MACH_NUMBER__"} # Example for uniform distribution
par = {"name":"MACH", "data":[0.73, 0.76, 0.8, 0.83, 0.85], "type":"data", "marker":"__MACH_NUMBER__"}

pars = [par]

def empty_function(design, x):
    return 0.0

def empty_gradient(design, x):
    return np.zeros(len(design))

mu = 0.8
sigma = 0.03

# unwanted helper function
def _generate_data_points():
    lb = mu - 3 * sigma
    ub = mu + 3 * sigma
    X = norm.rvs(lb, ub, loc=mu, scale=sigma, size=_numDataPoints)
    X.sort()
    return X.reshape(-1, 1)

def sample_normal_distribution(
        mu, sigma, numSamples    
    ):
    # Normal Sampling
    samples = np.random.normal(mu, sigma, numSamples)
    samples = samples.reshape(-1, 1)
    return samples

def sample_uniform_distribution(
        lb, ub, numSamples    
    ):
    samples = np.random.uniform(lb, ub, numSamples)
    samples = samples.reshape(-1, 1)
    return samples

def optimal_lhc_sampling(
        lb, ub, numSamples    
    ):
    # Optimal Latin Hypercube Sampling
    samples = pyDOE2.latin_hypercube_sampling(lb, ub, numSamples)
    samples = samples.reshape(-1, 1)
    return samples

def train_GP(X_train, y_train):
    gaussian_process = GaussianProcessRegressor()
    gaussian_process.fit(X_train, y_train)
    return gaussian_process  

def dro_pen_kriging_func(design, training_samples=None, penalty=0.0, scaling_fac=1e-6):
    
    logging.debug("Entering 'dro_pen_kriging_func'.")
    original_stdout = sys.stdout

    if training_samples is None:
        raise ValueError("Training samples are not provided. #TODO: Imlpement a value to sample from fixed parameters and strategy") 
    else:
        logging.debug("Entering 'fixed training samples' evaluation.")
    
    fs = [_training_samples_func(design, x) for x in training_samples] # TODO: Implement
    logging.debug(f"sample locations: {training_samples}")
    logging.debug(f"function values: {fs}") 
    
    # Gaussian Process Regression 
    gaussian_process = train_GP(training_samples, fs)
    
    # Generate data points to predict # TODO: Make this more general
    X = _generate_data_points()           
    
    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
    fs_X = mean_prediction
    file_name = "FUNCTION"
    plot_kriging_res(training_samples, fs, X, mean_prediction, std_prediction, file_name)
    
    # Compute empirical mean value
    f_em = 0
    num_data = X.shape[0]
    scaling = 1. / num_data
    for i in range(num_data):
        f_em = f_em + scaling * fs_X[i]
    
    f_em = np.mean(fs_X)
    f_ev = np.var(fs_X)
    logging.debug(f"f_em = {f_em}")
    logging.debug(f"f_ev = {f_ev}")

    # compute "equivalent" MV value
    pvar = 1. / ( 2. * penalty)    
    f_dro = f_em + pvar * f_ev    
    
    
    with open('history_log.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print("##################################################")
        print("[mean, var, stdv] = [{:1.5e}, {:1.5e}, {:1.5e}]".format(f_em, f_ev, np.sqrt(f_ev)))
        print("f_dro = {:1.5e}".format(f_dro))
        print("##################################################")
        sys.stdout = original_stdout # Reset the standard output to its original value    
    
    # rescale 
    f_dro = f_dro * scaling_fac
    return f_dro  


def dro_pen_kriging_fprime(design, training_samples=None, penalty=0.0, scaling_fac=1e-6):
    design_dim = len(design)
    logging.debug("Entering 'dro_pen_kriging_fprime'.")
    original_stdout = sys.stdout 

    if training_samples is None:
        raise ValueError("Training samples are not provided. #TODO: Imlpement a value to sample from fixed parameters and strategy") 
    else:
        logging.debug("Entering 'fixed training samples' evaluation.")
    
    fs = [_training_samples_func(design, x) for x in training_samples] # TODO: Implement            
    logging.debug(f"sample locations: {training_samples}")
    logging.debug(f"function values: {fs}") 
    
    # Gaussian Process Regression 
    gaussian_process = train_GP(training_samples, fs)
    
    # Generate data points to predict # TODO: Make this more general
    X = _generate_data_points()
    
    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
    fs_X = mean_prediction
    
    ##### GRADIENT #####
    dfs = [_training_samples_grad(design, x) for x in training_samples] # TODO: Implement
    dfdxs = np.zeros((design_dim, _numSamples)) 
    for j in range(_numSamples):
        df = dfs[j]
        print("df = ", df)
        for i in range(design_dim):
            dfdxs[i, j] = df[i] / scaling_fac            
    logging.debug(f"dfdxs: {dfdxs}")
    logging.debug(f"dfdxs shape: {dfdxs.shape}")
        
    # Training dfdxi(sample) for each gradient entries
    dfdx_X_trans = np.zeros((design_dim, _numDataPoints))
    for i in range(design_dim):
        y_train = dfdxs[i, :]
        gaussian_process = train_GP(training_samples, y_train)
        
        # predict gradients with GPR model
        mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
        dfdxi_X = mean_prediction
        dfdx_X_trans[i,:] = dfdxi_X
        
        if i == 0 or i == 15 or i == 30:
            print('y_train = ', y_train)
            file_name = f"GRADIENT_{i}"
            plot_kriging_res(training_samples, y_train, X, mean_prediction, std_prediction, file_name)
        
    dfdx_X = np.transpose(dfdx_X_trans)   
    
    ########################################
    # Compute robust gradient: sum_sample dFdx(sample)
    q_sup = _dro_pen_sup(fs_X, penalty)
    ddrof = np.zeros(design_dim)
    for i in range(_numDataPoints):
        scaling = q_sup[i]
        # print first5 data points
        if i < 5:
            with open('history_log.txt', 'a') as f:
                sys.stdout = f
                print('Sample({}): xi = {:0.4f}, fs_X = {:0.4f}, scaling = {:0.5e}'.format(i, X[i][0], fs_X[i], scaling))
                sys.stdout = original_stdout
        # print last 5 data points
        if i > _numDataPoints - 5:
            with open('history_log.txt', 'a') as f:
                sys.stdout = f 
                print('Sample({}): xi = {:0.4f}, fs_X = {:0.4f}, scaling = {:0.5e}'.format(i, X[i][0], fs_X[i], scaling))
                sys.stdout = original_stdout
        ddrof = ddrof + scaling * dfdx_X[i,:]
        
    for i in range(design_dim):
        ddrof[i] = ddrof[i] * scaling_fac
                    
    logging.debug(f"ddrof = {ddrof}")         
    return ddrof 

def _dro_pen_sup(fis, p):
    def loss(w):
        'function to minimize. w is a vector of weight fractions.'
        return -(np.dot(w, fis) -  np.sum(p/n*0.5*(n*w - 1)**2) )

    def ec(w):
        'weight fractions sum to one constraint'
        return 1 - np.sum(w)    
    
    n = len(fis)
    w0 = 1 / n*np.ones(n)
    
    y = fmin_slsqp(loss,   
        w0,
        eqcons= [ec], 
        bounds=[(0,1e6)] * len(w0),
        acc = 1e-12,
        #epsilon = 1.0e-04,
        iprint = -1)
    
    return y  

def plot_kriging_res(X_train, y_train, X, mean_prediction, std_prediction, file_name):
    plt.scatter(X_train, y_train, label="Observations")
    plt.plot(X, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$Mach$")
    plt.ylabel(file_name)
    _ = plt.title("Gaussian process regression on noise-free dataset")
    #plt.show()
    plt.savefig('__WORKDIR__/'+file_name+'.png')
    plt.close()