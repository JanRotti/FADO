import os
import time
import copy
import numpy as np
import logging
import itertools

from scipy.optimize import minimize, fmin_slsqp
from scipy.stats import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt

from .dro_parameter import DROParameter

from FADO.drivers import ConstrainedOptimizationDriver
from FADO import Parameter

class DROScipyDriver(ConstrainedOptimizationDriver):
    """
    Driver that sets up a Distributionally Robust Optimization Problem for
    SciPy optimizers, especially the constrained ones.
    """
    def __init__(self):
        ConstrainedOptimizationDriver.__init__(self)

        self._penalty = 1e6
        self._droParameters = []
        self._droExpandedParameters = {}
        self._trainingSamples = None
        self._numTrainingSamples = 5
        self._numDataPoints = 1000
        
        # Mapping for functions
        self._objMap    = {}
        self._consEqMap = {}
        self._consGtMap = {}

        # Numbers of functions
        self._nObj = 0
        self._nConsEQ = 0
        self._nConsGT = 0


    def setPenalty(self, penalty):
        """Set the penalty for the DRO problem."""
        assert penalty > 0.0, "Penalty must be positive."
        self._penalty = penalty


    def getBounds(self):
        """Return the variable bounds in a format compatible with SciPy."""
        return self._bounds
        

    def getConstraints(self):
        """Returns the constraint list that can be passed to SciPy."""
        return self._constraints


    def setNumDataPoints(self, numDataPoints):
        """
        Set the number of data points to use for the sampling of the 
        surrogate model that is used to approximate the internal maximization 
        problem DRO problem.
        """
        self._numDataPoints = numDataPoints


    def setNumTrainingSamples(self, numTrainingSamples):
        """
        Set the number of training samples to use for the internal surrogate
        that is used to approximate the internal maximization problem DRO 
        problem.
        """
        self._numTrainingSamples = numTrainingSamples


    def _sampleDROParameters(self, n):
        self._trainingSamples = []
        for par in self._droParameters:
            x = par.sample(n)
            self._trainingSamples.append(x.reshape(-1, 1))
        return self._trainingSamples


    def _preprocessVariables(self):
        super()._preprocessVariables()
        self._nObj    = len(self._objectives)
        self._nConsEQ = len(self._constraintsEQ)
        self._nConsGT = len(self._constraintsGT)
        self._nCons   = self._nConsEQ + self._nConsGT
        self._initializeDRO()
        super()._preprocessVariables()


    def _getDROParameters(self):
        if not self._droParameters:
            mask = [isinstance(par, DROParameter) for par in self._parameters]
            for param in [self._parameters[i] for i in range(len(mask)) if mask[i]]:
                self._droParameters.append(param)
        return self._droParameters


    def _expandDROParameters(self):
        """
        Expands DRO Parameters to a list of Parameters.
        """
        self._droExpandedParameters = {}
        for i, param in enumerate(self._droParameters):
            parser = param._parser # Copy original parser
            base_name = f"{param._name}"
            curr_expansion = []
            for j, value in enumerate(self._trainingSamples[i]):
                # Pass by reference, so changes to training samples will be reflected
                curr_expansion.append(Parameter([self._trainingSamples[i][j]], parser))
            self._droExpandedParameters[f"{param._name}"] = curr_expansion
        return self._droExpandedParameters


    def _parseRunToDRORun(self, fadoRUN):
        numSamples = self._numTrainingSamples
        dataFiles = fadoRUN._dataFiles
        
        runList = []
        for i in range(numSamples):
            run = copy.deepcopy(fadoRUN)
            # We want to have the parameters as reference, but not the whole attribute _parameters
            for j, param in enumerate(run._parameters):
                if isinstance(param, DROParameter):
                    run._parameters[j] = self._droExpandedParameters[param._name][i]
                else:
                    run._parameters[j] = fadoRUN._parameters[j] 
            run._workDir = f"{fadoRUN._workDir}_{i}"

            # Cleanup the already set workDir based files
            for j, file in enumerate(run._expectedFiles):
                new_file = file.lstrip(f"{fadoRUN._workDir}/")
                run._expectedFiles[j] = f"{run._workDir}/{new_file}"

            runList.append(run)

        return runList


    def _expandFunction(self, function, droRuns):
        
        base_name = function.function.getName()
        outFile = function.function._outFile
        substrings = outFile.split("/")
        newOutFile = os.path.join(f"{substrings[0]}_%i", *substrings[1:])

        # Find all files that need to be checked for data dependencies
        check_previous_flag = False
        prev_files = []
        new_functions = [copy.deepcopy(function) for i in range(self._numTrainingSamples)]

        # Set files that are outputs of DRO dependent evaluations
        index = 100000
        for i, evl in enumerate(function.function._funEval): # Find first eval that depends on DRO
            if evl._workDir in droRuns.keys():
                index = i
                break
        
        # Collect files that need an iterate
        for run in function.function._funEval[index:]:
            prev_files.extend(run._expectedFiles)

        # Replace the files in runs if necessary
        for i, run in enumerate(function.function._funEval):
            if i <= index:
                continue
            else:
                for j, file in enumerate(run._dataFiles):
                    if file in prev_files:
                        substrings = os.path.split(file)
                        file = f"{substrings[0]}_%i/{substrings[1]}"
                        for k, _ in enumerate(new_functions):
                            droRuns[run._workDir][k]._dataFiles[j] = file % k

        # Replace runs accordingly
        for i, run in enumerate(function.function._funEval):
            if i < index: # runs should be shared between functions!
                for j, _ in enumerate(new_functions):
                    new_functions[j].function._funEval[i] = function.function._funEval[i]
            else:
                for j, _ in enumerate(new_functions):
                    new_functions[j].function._funEval[i] = droRuns[run._workDir][j]

        # Repeat for grads
        # Set files that are outputs of DRO dependent evaluations
        if index < len(function.function._funEval):
            index = 0  # all evaluation are treated!
        else:
            for i, evl in enumerate(function.function._gradEval): # Find first eval that depends on DRO
                if evl._workDir in droRuns.keys():
                    index = i
                    break
        
        # Collect files that need an iterate
        for run in function.function._gradEval[index:]:
            prev_files.extend(run._expectedFiles)

        # Replace the files in runs if necessary
        for i, run in enumerate(function.function._gradEval):
            if i < index:
                continue
            else:
                for j, file in enumerate(run._dataFiles):
                    if file in prev_files:
                        substrings = os.path.split(file)
                        file = f"{substrings[0]}_%i/{substrings[1]}"
                        for k, _ in enumerate(new_functions):
                            droRuns[run._workDir][k]._dataFiles[j] = file % k

        # Replace runs accordingly
        for i, run in enumerate(function.function._gradEval):
            if i < index: # runs should be shared between functions!
                for j, _ in enumerate(new_functions):
                    new_functions[j].function._gradEval[i] = function.function._funEval[i]
            else:
                for j, _ in enumerate(new_functions):
                    new_functions[j].function._gradEval[i] = droRuns[run._workDir][j]

        # Replace the _gradFiles
        for i, file in enumerate(function.function._gradFiles):
            if file in prev_files:
                substrings = os.path.split(file)
                file = f"{substrings[0]}_%i/{substrings[1]}"
                for j, _ in enumerate(new_functions):
                    new_functions[j].function._gradFiles[i] = file % j

        for i, func in enumerate(new_functions):
            new_functions[i].function._name = f"{base_name}_{i}"
            new_functions[i].function._outFile = newOutFile % i
            new_functions[i].function._variables = function.function._variables
        
        return new_functions


    def _initializeDRO(self):
        # Collects self._droParameters from functions
        self._getDROParameters()
        # Fills self._trainingSamples as list of length(self._droParameters)
        self._sampleDROParameters(self._numTrainingSamples) 
        # Creates param.name -> List[Parameter] Dictionary at self._droExpandedParameters
        self._expandDROParameters()
        # Creates func.name -> Function Dictionary
        functions_to_expand = self._findFunctionsToExpand()
        # Creates list[Runs]
        runs_to_expand = self._findRunsToExpand(functions_to_expand) 
        # Creates run.workDir -> List[Run]; Does not fix data dependency yet!
        expanded_runs = {} 
        for i, run in enumerate(runs_to_expand):
            expanded_runs[run._workDir] = self._parseRunToDRORun(runs_to_expand[i]) 
        
        # Replace function with list of functions | Required to fit in the FADO framework
        expanded_functions = {}
        counter = 0
        for i, func in enumerate(self._objectives):
            func = self._objectives[i] # get reference
            if func.function.getName() in functions_to_expand.keys():
                expanded_functions[func] = self._expandFunction(func, expanded_runs)
                loc = self._objectives.index(func) 
                self._objectives.remove(func)
                self._objectives = self._objectives[:loc] + expanded_functions[func] + self._objectives[loc:]
                self._objMap[i] = np.arange(counter, counter+len(expanded_functions[func]))
                counter += len(expanded_functions[func])
            else:
                self._objMap[i] = [counter]
                counter += 1

        counter = 0
        for i, func in enumerate(self._constraintsEQ):
            if func.function.getName() in functions_to_expand.keys():
                expanded_functions[func] = self._expandFunction(func, expanded_runs)
                loc = self._constraintsEQ.index(func) 
                self._constraintsEQ.remove(func)
                self._constraintsEQ = self._constraintsEQ[:loc] + expanded_functions[func] + self._constraintsEQ[loc:]
                self._consEqMap[i] = np.arange(counter, counter+len(expanded_functions[func]))
                counter += len(expanded_functions[func])
            else:
                self._consEqMap[i] = [counter]
                counter += 1

        for i, func in enumerate(self._constraintsGT):
            if func.function.getName() in functions_to_expand.keys():
                expanded_functions[func] = self._expandFunction(func, expanded_runs)
                loc = self._constraintsGT.index(func) 
                self._constraintsGT.remove(func)
                self._constraintsGT = self._constraintsGT[:loc] + expanded_functions[func] + self._constraintsGT[loc:]
                self._consGtMap[i] = np.arange(counter, counter+len(expanded_functions[func]))
                counter += len(expanded_functions[func])
            else:
                self._consGtMap[i] = [counter]
                counter += 1
        
        # Clean up the computation graph if in parallel
        self.setEvaluationMode(self._parallelEval, self._waitTime)


    def _findFunctionsToExpand(self):
        functions_to_expand = {} # name -> func (reference)

        for i, func in enumerate(self._objectives): 
            if _checkFunctionForDROParameters(func):
                functions_to_expand[func.function.getName()] = self._objectives[i]
        for i, func in enumerate(self._constraintsEQ): 
            if _checkFunctionForDROParameters(func):
                functions_to_expand[func.function.getName()] = self._constraintsEQ[i]
        for i, func in enumerate(self._constraintsGT):
            if _checkFunctionForDROParameters(func):
                functions_to_expand[func.function.getName()] = self._constraintsGT[i]

        return functions_to_expand


    def _findRunsToExpand(self, functions_to_expand):
        runs_to_expand = []
        for i, name in enumerate(functions_to_expand):
            func = functions_to_expand[name]
            index = len(func.function._funEval)
            for j, evl in enumerate(func.function._funEval): # Find first eval that depends on DRO
                if any([isinstance(param, DROParameter) for param in evl.getParameters()]):
                    index = j
                    break
            # Access function as reference
            for j in range(index, len(func.function._funEval)):
                evl = functions_to_expand[name].function._funEval[j]
                if not evl in runs_to_expand:
                    runs_to_expand.append(evl)

            index = 0 if index < len(func.function._funEval) else len(func.function._funEval)
            for j, evl in enumerate(func.function._gradEval): # Find first eval that depends on DRO
                if any([isinstance(param, DROParameter) for param in evl.getParameters()]):
                    index = j
                    break
            for j in range(index, len(func.function._gradEval)):
                evl = functions_to_expand[name].function._gradEval[j]
                if not evl in runs_to_expand:
                    runs_to_expand.append(evl)

        return runs_to_expand


    def preprocess(self):
        """
        Prepares the optimization problem, including preprocessing variables,
        and setting up the lists of constraints and variable bounds that SciPy
        needs. Must be called after all functions are added to the driver.
        """
        ConstrainedOptimizationDriver.preprocess(self)

        class _fun:
            def __init__(self,fun,idx):
                self._f = fun
                self._i = idx
            def __call__(self,x):
                return self._f(x,self._i)
        #end
        self._constraints = []

        for i in range(self._nConsEQ):
            self._constraints.append({'type' : 'eq',
                                      'fun' : _fun(self._eval_g, i),
                                      'jac' : _fun(self._eval_jac_g, i)})
        
        for i in range(self._nConsEQ, self._nConsGT + self._nConsEQ):
            self._constraints.append({'type' : 'ineq',
                                      'fun' : _fun(self._eval_g, i),
                                      'jac' : _fun(self._eval_jac_g, i)})

        # variable bounds
        self._bounds = np.array((self.getLowerBound(),self.getUpperBound()),float).transpose()

        # size the gradient and constraint jacobian
        self._grad_f = np.zeros((self._nVar,))
        self._old_grad_f = np.zeros((self._nVar,))
        self._jac_g = np.zeros((self._nVar,self._nCons))
        self._old_jac_g = np.zeros((self._nVar,self._nCons))


    def _eval_g(self, x, idx):
        self._evaluateFunctions(x)
        if idx >= self._nConsEQ:
            idx = idx - self._nConsEQ
            indizes = self._consGtMap[idx]
            values = [self._gtval[index] for index in indizes]
        else:
            indizes = self._consEqMap[idx]
            values = [self._eqval[index] for index in indizes]
        if len(values) == 1:
            return values[0]
        elif len(values) == self._numTrainingSamples:
            return self._dro_kriging_func(values, self._penalty, f"CONSTRAINT_{idx}")
        else:
            raise ValueError("Number of values does not match the number of training samples")


    def _eval_jac_g(self, x, idx):
        self._jacTime -= time.time()
        try:
            self._evaluateGradients(x)
            os.chdir(self._workDir)
            mask = self._variableStartMask
            if idx >= self._nConsEQ:
                indizes = self._consGtMap[idx-self._nConsEQ]
                values = [self._gtval[index] for index in indizes]
                if len(values) == 1:
                    index = indizes[0]
                    self._jac_g[:,idx] = self._constraintsGT[index].function.getGradient(mask) * self._constraintsGT[index].scale / self._varScales
                else:
                    df = [self._constraintsGT[index].function.getGradient(mask) for index in indizes]
                    self._jac_g[:,idx] = self._dro_kriging_fprime(values, df, self._penalty, f"GRADIENT_{idx}")
            else:
                indizes = self._consEqMap[idx]
                values = [self._eqval[index] for index in indizes]
                if len(values) == 1:
                    index = indizes[0]
                    self._jac_g[:,idx] = self._constraintsEQ[index].function.getGradient(mask) * self._constraintsEQ[index].scale / self._varScales
                else:
                    df = [self._constraintsEQ[index].function.getGradient(mask) for index in indizes]
                    self._jac_g[:,idx] = self._dro_kriging_fprime(values, df, self._penalty, f"GRADIENT_{idx}")
            self._old_jac_g[:,idx] = self._jac_g[:,idx]
        except:
            if self._failureMode == "HARD": raise
            self._jac_g[:,idx] = self._old_jac_g[:,idx]

        if not self._parallelEval:
            self._runAction(self._userPostProcessGrad)

        self._jacTime += time.time()
        os.chdir(self._userDir)

        return self._jac_g[:,idx]


    def fun(self, x):
        """Method passed to SciPy to get the objective function value."""
        self._evaluateFunctions(x)
        out = 0.0
        for idx in range(self._nObj):
            indizes = self._objMap[idx]
            values = self._ofval[indizes]
            if len(values) == 1:
                out += values[0]
            else:
                out += self._dro_kriging_func(values, self._penalty, f"OBJECTIVE_{idx}")
        return out


    def grad(self, x):
        """Method passed to SciPy to get the objective function gradient."""
        self._jacTime -= time.time()
        if True:
        #try:
            self._evaluateGradients(x)
            os.chdir(self._workDir)
            self._grad_f[()] = 0.0
            for idx in range(self._nObj):
                indizes = self._objMap[idx]
                if len(indizes) == 1:
                    self._grad_f += self._objectives[indizes[0]].function.getGradient(self._variableStartMask) * self._objectives[indizes[0]].scale
                else:
                    values = self._ofval[indizes]
                    df = [self._objectives[index].function.getGradient(self._variableStartMask) * self._objectives[index].scale for index in indizes]
                    self._grad_f += self._dro_kriging_fprime(values, df, self._penalty, f"GRADIENT_{idx}")
        #except:
        #    if self._failureMode == "HARD": raise
        #    else:
        #        logging.error("'grad' evaluation failed. Using old value.")
        #    self._grad_f[()] = self._old_grad_f

        if not self._parallelEval:
            self._runAction(self._userPostProcessGrad)

        self._jacTime += time.time()
        os.chdir(self._userDir)

        return self._grad_f


    def _dro_kriging_func(self, vals, penalty = 0.0, name="FUNCTION"): # TODO: Extend to multiple DRO Parameters
        logging.debug("Entering 'dro_pen_kriging_func'.")
        gaussian_process = self._train_GP(self._trainingSamples[0], vals)

        # Generate data points to predict # TODO: Make this more general
        X = self._generate_samples()[0]       

        mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
        fs_X = mean_prediction

        self.plot_kriging_res(self._trainingSamples[0], vals, X, mean_prediction, std_prediction, name)
        
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
    
        # rescaling is done in _eval_g, func, ...
        return f_dro 


    def _dro_kriging_fprime(self, vals, grads, penalty = 0.0, name="GRADIENT"): # TODO: Extend to multiple DRO Parameters
        design_dim = self._nVar
        logging.debug("Entering 'dro_pen_kriging_fprime'.")

        # Gaussian Process Regression 
        gaussian_process = self._train_GP(self._trainingSamples[0], vals)
        
        # Generate data points to predict # TODO: Make this more general
        X = self._generate_samples()[0]
        
        mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
        fs_X = mean_prediction
        
        ##### GRADIENT #####
        dfs = grads
        dfdxs = np.zeros((design_dim, len(dfs))) 
        for j in range(len(dfs)):
            df = dfs[j]
            logging.debug("df = ", df)
            for i in range(design_dim):
                dfdxs[i, j] = df[i]           
        logging.debug(f"dfdxs: {dfdxs}")
        logging.debug(f"dfdxs shape: {dfdxs.shape}")
            
        # Training dfdxi(sample) for each gradient entries
        dfdx_X_trans = np.zeros((design_dim, self._numDataPoints))
        for i in range(design_dim):
            y_train = dfdxs[i, :]
            gaussian_process = self._train_GP(self._trainingSamples[0], y_train)
            
            # predict gradients with GPR model
            mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
            dfdxi_X = mean_prediction
            dfdx_X_trans[i,:] = dfdxi_X
            
            if i == 0 or i == 15 or i == 30:
                file_name = f"{name}_{i}"
                self.plot_kriging_res(self._trainingSamples[0], y_train, X, mean_prediction, std_prediction, file_name)
            
        dfdx_X = np.transpose(dfdx_X_trans)   
        
        ########################################
        # Compute robust gradient: sum_sample dFdx(sample)
        q_sup = self._dro_pen_sup(fs_X, penalty)
        ddrof = np.zeros(design_dim)
        for i in range(self._numDataPoints):
            scaling = q_sup[i]
            ddrof = ddrof + scaling * dfdx_X[i,:]
            
        # Scaling is applied in _eval_jac_g
        logging.debug(f"ddrof = {ddrof}")         
        return ddrof 


    def _dro_pen_sup(self,fis, p):
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

    def _train_GP(self, X_train, y_train):
        gaussian_process = GaussianProcessRegressor()
        gaussian_process.fit(X_train, y_train)
        return gaussian_process 

    def _generate_samples(self):
        data = []
        for par in self._droParameters:
            if par._type == 'normal':
                mu = par._mu
                sigma = np.sqrt(par._var)
                lb = mu - 3 * sigma
                ub = mu + 3 * sigma
                X = norm.rvs(lb, ub, loc=mu, scale=sigma, size=self._numDataPoints)
                X.sort()  
            elif par._type == "uniform":
                X = np.linspace(par._lb, par._ub, self._numDataPoints)
            elif par._type == "data":
                X = np.linspace(np.min(par._data), np.max(par._data), self._numDataPoints)
            data.append(X.reshape(-1, 1))
        return data

    # utilitiy function
    def plot_kriging_res(self, X_train, y_train, X, mean_prediction, std_prediction, file_name):
        plt.clf()

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
        
        plt.savefig(os.path.join(self._userDir, self._workDir, f"{file_name}.png"))
        plt.close()  


###########################################
#
#   HELPER FUNCTIONS | DRO FUNCTIONS
#
###########################################  
def _checkFunctionForDROParameters(func):
        for param in func.function.getParameters():
            if isinstance(param, DROParameter):
                return True
        return False