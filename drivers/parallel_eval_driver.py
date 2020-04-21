#  Copyright 2019-2020, Pedro Gomes.
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
import time
from drivers.base_driver import DriverBase


# Intermediate class that adds parallel evaluation capabilities.
# asNeeded: If True, the gradients of constraints are only evaluated if they are
#           active, this is possible for the exterior penalty driver.
class ParallelEvalDriver(DriverBase):
    def __init__(self, asNeeded = False):
        DriverBase.__init__(self)

        # timers
        self._funTime = 0
        self._jacTime = 0

        # variables for parallelization of evaluations
        self._asNeeded = asNeeded
        self._parallelEval = False
        self._funEvalGraph = None
        self._jacEvalGraph = None
        self._waitTime = 10.0
    #end

    # build evaluation graphs for parallel execution
    def setEvaluationMode(self,parallel=True,waitTime=10.0):
        self._parallelEval = parallel
        if not parallel: return # no need to build graphs
        self._waitTime = waitTime

        # get all unique evaluation steps
        valEvals = set()
        jacEvals = set()

        def _addEvals(flist,vlist,jlist):
            for obj in flist:
                vlist.update(obj.function.getValueEvalChain())
                jlist.update(obj.function.getGradientEvalChain())
            #end
        #end
        _addEvals(self._objectives   ,valEvals,jacEvals)
        _addEvals(self._constraintsEQ,valEvals,jacEvals)
        _addEvals(self._constraintsLT,valEvals,jacEvals)
        _addEvals(self._constraintsGT,valEvals,jacEvals)
        _addEvals(self._constraintsIN,valEvals,jacEvals)

        # for each unique evaluation list its direct dependencies
        self._funEvalGraph = dict(zip(valEvals,[set() for i in range(len(valEvals))]))
        self._jacEvalGraph = dict(zip(jacEvals,[set() for i in range(len(jacEvals))]))

        def _addDependencies(flist,funGraph,jacGraph):
            for obj in flist:
                evals = obj.function.getValueEvalChain()
                for i in range(1,len(evals)):
                    funGraph[evals[i]].add(evals[i-1])
                    
                evals = obj.function.getGradientEvalChain()
                for i in range(1,len(evals)):
                    jacGraph[evals[i]].add(evals[i-1])
            #end
        #end
        _addDependencies(self._objectives   ,self._funEvalGraph,self._jacEvalGraph)
        _addDependencies(self._constraintsEQ,self._funEvalGraph,self._jacEvalGraph)
        _addDependencies(self._constraintsLT,self._funEvalGraph,self._jacEvalGraph)
        _addDependencies(self._constraintsGT,self._funEvalGraph,self._jacEvalGraph)
        _addDependencies(self._constraintsIN,self._funEvalGraph,self._jacEvalGraph)
    #end

    # run evaluations extracting maximum parallelism
    def _evalFunInParallel(self):
        self._funTime -= time.time()
        while True:
            allRun = True
            for evl,depList in self._funEvalGraph.items():
                # either running or finished, move on
                if evl.isIni() or evl.isRun():
                    evl.poll() # (starts or updates internal state)
                    allRun &= evl.isRun()
                    continue
                #end
                allRun &= evl.isRun()

                # if dependencies are met, start evaluation
                for dep in depList:
                    if not dep.isRun(): break
                else:
                    evl.initialize()
                    evl.poll()
                #end
            #end
            if allRun: break
            time.sleep(self._waitTime)
        #end
        self._funTime += time.time()
    #end

    # same for gradients but having in mind which functions are active
    def _evalJacInParallel(self):
        self._jacTime -= time.time()

        # determine what evaluations are active based on functions
        active = dict(zip(self._jacEvalGraph.keys(),\
                          [False for i in range(len(self._jacEvalGraph))]))

        for obj in self._objectives:
            for evl in obj.function.getGradientEvalChain():
                active[evl] = True

        for obj in self._constraintsEQ:
            for evl in obj.function.getGradientEvalChain():
                active[evl] = True

        for (obj,f) in zip(self._constraintsLT,self._ltval):
            if f > 0.0 or not self._asNeeded:
                for evl in obj.function.getGradientEvalChain():
                    active[evl] = True

        for (obj,f) in zip(self._constraintsGT,self._gtval):
            if f < 0.0 or not self._asNeeded:
                for evl in obj.function.getGradientEvalChain():
                    active[evl] = True

        for (obj,f) in zip(self._constraintsIN,self._inval):
            if f > 1.0 or f < 0.0 or not self._asNeeded:
                for evl in obj.function.getGradientEvalChain():
                    active[evl] = True

        while True:
            allRun = True
            for evl,depList in self._jacEvalGraph.items():
                if not active[evl]: continue

                # ensure all dependencies are active
                for dep in depList:
                    active[dep] = True
                
                # either running or finished, move on
                if evl.isIni() or evl.isRun():
                    evl.poll() # (starts or updates internal state)
                    allRun &= evl.isRun()
                    continue
                #end
                allRun &= evl.isRun()

                # if dependencies are met, start evaluation
                for dep in depList:
                    if not dep.isRun(): break
                else:
                    evl.initialize()
                    evl.poll()
                #end
            #end
            if allRun: break
            time.sleep(self._waitTime)
        #end
        self._jacTime += time.time()
    #end

    # evaluate everything, either in parallel or sequentially, and fetch the function values
    def _evalAndRetrieveFunctionValues(self):
        os.chdir(self._workDir)

        if self._parallelEval:
            try:
                self._evalFunInParallel()
            except:
                if self._failureMode is "HARD": raise
        #end

        self._funEval += 1
        self._funTime -= time.time()

        def fetchValues(dst, src):
            for i in range(dst.size):
                try:
                    dst[i] = src[i].function.getValue()
                except:
                    if src[i].function.hasDefaultValue() and self._failureMode is "SOFT":
                        dst[i] = src[i].function.getDefaultValue()
                    else:
                        raise
                #end
            #end
        #end

        fetchValues(self._ofval, self._objectives)
        fetchValues(self._eqval, self._constraintsEQ)
        fetchValues(self._ltval, self._constraintsLT)
        fetchValues(self._gtval, self._constraintsGT)
        fetchValues(self._inval, self._constraintsIN)

        self._funTime += time.time()

        # monitor convergence (raw function values)
        self._writeHisLine()

        # shift constraints and scale as required
        for i in range(self._ofval.size):
            self._ofval[i] *= self._objectives[i].scale

        for i in range(self._eqval.size):
            self._eqval[i] -= self._constraintsEQ[i].bound1
            self._eqval[i] *= self._constraintsEQ[i].scale

        for i in range(self._ltval.size):
            self._ltval[i] -= self._constraintsLT[i].bound1
            self._ltval[i] *= self._constraintsLT[i].scale

        for i in range(self._gtval.size):
            self._gtval[i] -= self._constraintsGT[i].bound1
            self._gtval[i] *= self._constraintsGT[i].scale

        for i in range(self._inval.size):
            self._inval[i] -= self._constraintsIN[i].bound1
            self._inval[i] *= self._constraintsIN[i].scale

        os.chdir(self._userDir)
    #end
#end

