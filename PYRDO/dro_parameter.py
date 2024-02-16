import numpy as np

class DROParameter:

    def __init__(self, par: dict, parser):
        super().__init__()
        par = par
        self._name = par["name"]
        self._parser = parser
        self._type = par["type"]
        self._data = None
        self._mu = None
        self._var = None
        self._lb = None
        self._ub = None
        self.initialize(par)

    def initialize(self, par):
        if self._type == "normal":
            self._mu = par["mu"]
            self._var = par["var"]
            self._lb = par["mu"] - 3 * par["var"]
            self._ub = par["mu"] + 3 * par["var"]
        elif self._type == "uniform":
            self._lb = par["lb"]
            self._ub = par["ub"]
        elif self._type == "data":
            self._data = par["data"]
            self._lb = min(self._data)
            self._ub = max(self._data)
        else:
            raise ValueError("Invalid type for parameter: {}".format(self._name))

    def sample(self, n: int):
        if self._type == "normal":
            return np.random.normal(self._mu, self._var, n)
        elif self._type == "uniform":
            return np.random.uniform(self._lb, self._ub, n)
        elif self._type == "data":
            return np.random.choice(self._data, n, replace=False)