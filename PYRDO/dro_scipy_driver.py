#  Copyright 2019-2023, FADO Contributors (cf. AUTHORS.md)
#
#  This file is part of FADO.
#
#  FADO is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  FADO is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with FADO.  If not, see <https://www.gnu.org/licenses/>.

import os
import logging
import time
import numpy as np

from scipy.optimize import minimize, fmin_slsqp
from scipy.stats import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt

from FADO import ConstrainedOptimizationDriver

class StaticDRODriver(ConstrainedOptimizationDriver):
    """
    Driver to use with the SciPy optimizers, especially the constrained ones.
    """

    def __init__(self):
        ConstrainedOptimizationDriver.__init__(self)
        
        self._trainingSamples = None
        self._lengthTrainingSamples = None
        self._numDataPoints = 1000

        self._droObjectiveIndizes = []
        self._droConstraintIndizesEQ = []
        self._droConstraintIndizesGT = []

        self._nDroFun = None
        self._nDroCon = None
        self._nEQ = None
        self._nINEQ = None

        self._DRO_Parameter = None
        self._penalty = 1e6
    #end

    def addPenalty(self, penalty):
        assert penalty > 0.0, "Penalty must be positive."
        self._penalty = penalty

    def setTrainingSamples(self, X):
        self._trainingSamples = X.reshape(-1, 1)
        self._lengthTrainingSamples = len(X)

    def addDROParameter(self, parameter):
        self._DRO_Parameter = parameter

    def setNumDataPoints(self, numDataPoints):
        self._numDataPoints = numDataPoints

    def addDROObjective(self, type, functions, scale=1.0, weight=1.0):
        assert len(functions) == self._lengthTrainingSamples, "Number of functions must match number of training samples." 
        if scale <= 0.0: raise ValueError("Scale must be positive.")
        current_idx = len(self._objectives)
        self._droObjectiveIndizes.append((current_idx, current_idx + len(functions)))
        for func in functions:
            self._objectives.append(self._Objective(type,func,scale,weight))
    
    def addDROEquality(self, functions, target=0.0, scale=1.0):
        assert len(functions) == self._lengthTrainingSamples, "Number of functions must match number of training samples." 
        if scale <= 0.0: raise ValueError("Scale must be positive.")
        current_idx = len(self._constraintsEQ)
        self._droConstraintIndizesEQ.append((current_idx, current_idx + len(functions)))
        for func in functions:    
            self._constraintsEQ.append(self._Constraint(func,scale,target))
    
    def addDROLowerBound(self, functions, bound=0.0, scale=1.0):
        assert len(functions) == self._lengthTrainingSamples, "Number of functions must match number of training samples." 
        if scale <= 0.0: raise ValueError("Scale must be positive.")
        current_idx = len(self._constraintsGT)
        self._droConstraintIndizesGT.append((current_idx, current_idx + len(functions)))
        for func in functions:
            self._constraintsGT.append(self._Constraint(func,scale,bound))
    
    def addDROUpperBound(self, functions, bound=0.0, scale=1.0):
        assert len(functions) == self._lengthTrainingSamples, "Number of functions must match number of training samples." 
        if scale <= 0.0: raise ValueError("Scale must be positive.")
        current_idx = len(self._constraintsGT)
        self._droConstraintIndizesGT.append((current_idx, current_idx + len(functions)))
        for func in functions:
            self._constraintsGT.append(self._Constraint(func,-1*scale,bound))
        
    def addDROUpLowBound(self, functions, lower=-1.0, upper=1.0):
        assert len(functions) == self._lengthTrainingSamples, "Number of functions must match number of training samples." 
        if lower >= upper: raise ValueError("Upper bound must be greater than lower bound.")
        scale = 1.0/(upper-lower)
        current_idx = len(self._constraintsGT) 
        self._droConstraintIndizesGT.append((current_idx, current_idx + len(functions)))
        for func in functions:
            self._constraintsGT.append(self._Constraint(functions, scale,lower))
        current_idx = len(self._constraintsGT)
        self._droConstraintIndizesGT.append((current_idx, current_idx + len(functions)))
        for func in functions:
            self._constraintsGT.append(self._Constraint(functions,-1*scale,upper))    

    def preprocess(self):
        """
        Prepares the optimization problem, including preprocessing variables,
        and setting up the lists of constraints and variable bounds that SciPy
        needs. Must be called after all functions are added to the driver.
        """
        ConstrainedOptimizationDriver.preprocess(self)

        if not (self._droObjectiveIndizes or self._droConstraintIndizesEQ\
                or self._droConstraintIndizesGT):
            assert self._DRO_Parameter is not None, "DRO Parameter must be set."

        class _fun:
            def __init__(self,fun,idx):
                self._f = fun
                self._i = idx
            def __call__(self,x):
                return self._f(x,self._i)
        #end
            
        _constraints = []
        # Create an index map for the association of constraint functions
        index = 0
        idx_list = np.arange(0, len(self._constraintsEQ))
        for i, j in self._droConstraintIndizesEQ:
            while not index==i:
                _constraints.append({
                    'idx' : [index],
                    'type' : 'eq',
                    'fun' : _fun(self._eval_g, index),
                    'jac' : _fun(self._eval_jac_g, index),
                    }
                )
                index += 1
            _constraints.append({
                'idx' : idx_list[i:j],
                'type' : 'eq',
                'fun' : _fun(self._eval_g, idx_list[i:j]),
                'jac' : _fun(self._eval_jac_g, idx_list[i:j]),
                }
            )
        self._nEQ = len(self._constraints)
        index = 0
        for i, j in self._droConstraintIndizesGT:
            while not index==i:
                _constraints.append({
                    'idx' : [index],
                    'type' : 'ineq',
                    'fun' : _fun(self._eval_g, index),
                    'jac' : _fun(self._eval_jac_g, index),
                    }
                )
                index += 1
            _constraints.append({
                'idx' : idx_list[i:j],
                'type' : 'ineq',
                'fun' : _fun(self._eval_g, idx_list[i:j]),
                'jac' : _fun(self._eval_jac_g, idx_list[i:j]),
                }
            )
        self._nINEQ = len(self._constraints) - self._nEQ
        
        # Reset nCon by considering DRO functions
        self._nCon = self._nEQ + self._nINEQ

        # variable bounds
        self._bounds = np.array((self.getLowerBound(),self.getUpperBound()),float).transpose()

        # size the gradient and constraint jacobian
        self._grad_f = np.zeros((self._nVar,))
        self._old_grad_f = np.zeros((self._nVar,))
        self._jac_g = np.zeros((self._nVar,self._nCon))
        self._old_jac_g = np.zeros((self._nVar,self._nCon))
    #end

    def fun(self, x):
        """Method passed to SciPy to get the objective function value."""
        # Evaluates all functions if necessary.
        self._evaluateFunctions(x)
        out = 0.0
        index = 0
        idx_list = np.arange(0, len(self._objectives))
        for i, j in self._droObjectiveIndizes:
            while not index==i:
                out += self._ofval[index]
                index += 1
            out += self._dro_kriging_func(self._ofval[idx_list[i:j]], self._penalty)
        return out
    #end

    def grad(self, x):
        """Method passed to SciPy to get the objective function gradient."""
        # Evaluates gradients and functions if necessary, otherwise it
        # simply combines and scales the results.
        self._jacTime -= time.time()
        try:
            self._evaluateGradients(x)

            os.chdir(self._workDir)

            self._grad_f[()] = 0.0
            for obj in self._objectives:
                self._grad_f += obj.function.getGradient(self._variableStartMask) * obj.scale
            
            index = 0
            idx_list = np.arange(0, len(self._objectives))
            for (i, j) in self._droObjectiveIndizes:
                while not index==i:
                    obj = self._objectives[index]
                    self._grad_f += obj.function.getGradient(self._variableStartMask) * obj.scale
                    index += 1
                required_indexes = idx_list[range(i, j)]
                objectives_to_handle = [self._objectives[index] for index in required_indexes]  
                grad_f_list = [obj.function.getGradient(self._variableStartMask) * obj.scale for obj in objectives_to_handle]
                
                self._grad_f += self._dro_kriging_fprime(self._ofval[idx_list[i:j]], grad_f_list, self._penalty)

            self._grad_f /= self._varScales

            # keep copy of result to use as fallback on next iteration if needed
            self._old_grad_f[()] = self._grad_f
            
        except:
            if self._failureMode == "HARD": raise
            else:
                logging.error("'grad' evaluation failed. Using old value.")
            self._grad_f[()] = self._old_grad_f
        #end

        if not self._parallelEval:
            self._runAction(self._userPostProcessGrad)

        self._jacTime += time.time()
        os.chdir(self._userDir)

        return self._grad_f
    #end

    # Method passed to SciPy to expose the constraint vector.
    def _eval_g(self, x, idx):
        self._evaluateFunctions(x)
        if hasattr(idx, '__iter__'):
            if idx[0] < len(self._constraintsEQ):
                out = self._dro_kriging_func(self._eqval[idx], self._penalty)
            else:
                ref_length = len(self._constraintsEQ)
                out = self._dro_kriging_func(self._gtval[idx-ref_length], self._penalty)
        else:
            if idx < len(self._constraintsEQ):
                out = self._eqval[idx]
            else:
                ref_length = len(self._constraintsEQ)
                out = self._gtval[idx-ref_length]
            #end

        return out
    #end

    # Method passed to SciPy to expose the constraint Jacobian.
    def _eval_jac_g(self, x, idx): #TODO: Adapt to DRO
        # This repeats self._eval_g syntax -> Required function vals in dro helper
        self._evaluateFunctions(x)
        if hasattr(idx, '__iter__'):
            if idx[0] < len(self._constraintsEQ):
                vals = self._eqval[idx]
            else:
                ref_length = len(self._constraintsEQ)
                vals = self._gtval[idx-ref_length]
        else:
            if idx < len(self._constraintsEQ):
                vals = self._eqval[idx]
            else:
                ref_length = len(self._constraintsEQ)
                vals = self._gtval[idx-ref_length]
        #end
        
        self._jacTime -= time.time()
        try:
            self._evaluateGradients(x)

            os.chdir(self._workDir)

            mask = self._variableStartMask

            if hasattr(idx, '__iter__'):
                if idx[0] < len(self._constraintsEQ):
                    con = self._constraintsEQ[idx]
                    f = -1.0 # for purposes of lazy evaluation equality is always active
                else:
                    ref_length = len(self._constraintsEQ)
                    f = self._gtval[idx-ref_length]
            
                if f < 0.0 or not self._asNeeded:
                    df = [c.function.getGradient(mask) for c in con]
                    self._jac_g[:,idx] = self._dro_kriging_fprime(vals, df) * con.scale / self._varScales
                else:
                    self._jac_g[:,idx] = 0.0
                #end
            else:
                if idx < len(self._constraintsEQ):
                    con = self._constraintsEQ[idx]
                    f = -1.0 # for purposes of lazy evaluation equality is always active
                else:
                    con = self._constraintsGT[idx-len(self._constraintsEQ)]
                    f = self._gtval[idx-len(self._constraintsEQ)]
                #end

                if f < 0.0 or not self._asNeeded:
                    self._jac_g[:,idx] = con.function.getGradient(mask) * con.scale / self._varScales
                else:
                    self._jac_g[:,idx] = 0.0
                #end
            #end
            # keep reference to result to use as fallback on next iteration if needed
            self._old_jac_g[:,idx] = self._jac_g[:,idx]
        except:
            if self._failureMode == "HARD": raise
            self._jac_g[:,idx] = self._old_jac_g[:,idx]
        #end

        if not self._parallelEval:
            self._runAction(self._userPostProcessGrad)

        self._jacTime += time.time()
        os.chdir(self._userDir)

        return self._jac_g[:,idx]
    #end
#end

    def _dro_kriging_func(self, vals, penalty = 0.0, name="FUNCTION"):
        logging.debug("Entering 'dro_pen_kriging_func'.")
        gaussian_process = self.train_GP(self._trainingSamples, vals)

        # Generate data points to predict # TODO: Make this more general
        X = self._generate_samples()       

        mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
        fs_X = mean_prediction

        self.plot_kriging_res(self._trainingSamples, vals, X, mean_prediction, std_prediction, name)
        
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

    def _dro_kriging_fprime(self, vals, grads, penalty = 0.0, name="GRADIENT"):
        design_dim = self._nVar
        logging.debug("Entering 'dro_pen_kriging_fprime'.")

        # Gaussian Process Regression 
        gaussian_process = self.train_GP(self._trainingSamples, vals)
        
        # Generate data points to predict # TODO: Make this more general
        X = self._generate_samples()
        
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
            gaussian_process = self.train_GP(self._trainingSamples, y_train)
            
            # predict gradients with GPR model
            mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
            dfdxi_X = mean_prediction
            dfdx_X_trans[i,:] = dfdxi_X
            
            if i == 0 or i == 15 or i == 30:
                file_name = f"{name}_{i}"
                self.plot_kriging_res(self._trainingSamples, y_train, X, mean_prediction, std_prediction, file_name)
            
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

    def train_GP(self, X_train, y_train):
        gaussian_process = GaussianProcessRegressor()
        gaussian_process.fit(X_train, y_train)
        return gaussian_process  
    
    # Not so nice to have here, but for now it works
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
        
        plt.savefig(f'{self._userDir}/{self._workDir}/{file_name}.png')
        plt.close()    
        
    def _generate_samples(self):
        par = self._DRO_Parameter
        if par['type'] == 'normal':
            mu = par['mu']
            sigma = np.sqrt(par['var'])
            lb = mu - 3 * sigma
            ub = mu + 3 * sigma
            X = norm.rvs(lb, ub, loc=mu, scale=sigma, size=self._numDataPoints)
            X.sort()  
        elif par['type'] == "uniform":
            X = np.linspace(par['lb'], par['ub'], self._numDataPoints)
        elif par['type'] == "data":
            X = np.linspace(np.min(par['data']), np.max(par['data']), self._numDataPoints)
        return X.reshape(-1, 1)
    
    def getConstraints(self):
        """Returns the constraint list that can be passed to SciPy."""
        return self._constraints

    def getBounds(self):
        """Return the variable bounds in a format compatible with SciPy."""
        return self._bounds