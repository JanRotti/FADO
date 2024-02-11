import sys, sys
import logging 
import copy

import numpy as np
import pyDOE2

from scipy.optimize import minimize, fmin_slsqp
from scipy.stats import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt

from .dro_base import *
from .ordered_bunch import *
from FADO import *

def parse_RUN_to_robust_RUN(fadoRUN, files_to_check=[], numSamples=10, parameters_to_add=None):
    dataFiles = fadoRUN._dataFiles
    file_index_to_change = []
    if files_to_check=="all":
        file_index_to_change = np.arange(0, len(dataFiles))
    else:
       for i, file in enumerate(dataFiles):
            if file in files_to_check or any([file.endswith(f) for f in files_to_check]):
                file_index_to_change.append(i)
        #end       

    runList = []
    if parameters_to_add is not None:
        numSamples = len(parameters_to_add)

    for i in range(numSamples):
        run = copy.deepcopy(fadoRUN)
        run._workDir = f"{fadoRUN._workDir}_{i}"
        
        for j, file in enumerate(run._expectedFiles):
            new_file = file.lstrip(f"{fadoRUN._workDir}")
            run._expectedFiles[j] = f"{run._workDir}/{new_file}"

        for j, file in enumerate(run._dataFiles):
            if j in file_index_to_change:
                substrings = file.split("/", 1)
                run._dataFiles[j] = f"{substrings[0]}_%i/{substrings[1]}" % i

        run.addParameter(parameters_to_add[i])
        runList.append(run)

    return runList


def parse_Function_to_robust_function(fadoFunction, numSamples=10):
    functionList = []
    name = fadoFunction.getName()
    outFile = fadoFunction._outFile
    tmp = outFile.split("/")
    newOutFile = os.path.join(f"{tmp[0]}_%i", *tmp[1:]) # Add a counter
    
    for i in range(numSamples):
        fun = copy.deepcopy(fadoFunction)
        # Add the additional parameter change
        fun._name = f"{name}_{i}"
        fun._outFile = newOutFile % i
        functionList.append(fun)

    return functionList

def generate_samples(par, numSamples=10):
    par = OrderedBunch(par)
    marker = par.marker

    if par.type=="normal":
        values = sample_normal_distribution(par.mu, par.var, numSamples)
    elif par.type=="uniform":
        values = sample_uniform_distribution(par.lb, par.ub, numSamples)
    elif par.type=="data":
        if numSamples <= len(par.data):
            values = np.random.choice(par.data, numSamples, replace=False)
        else:
            logging.warning("Number of samples is smaller than the number of data points. Using all data points.")
            values = par.data
    else:
        raise Exception("Unknown distribution type.")
    
    return values