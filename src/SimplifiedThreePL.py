import numpy as np
from scipy.optimize import minimize

class SimplifiedThreePL:
    def __init__(self, experiment):
        self.experiment = experiment
        self._base_rate = None
        self._logit_base_rate = None
        self._discrimination = None
        self._is_fitted = False
    
    def summary(self):
        return {
            "n_total": len(self.experiment.trials),
            "n_correct": sum(self.experiment.trials),
            "n_incorrect": len(self.experiment.trials) - sum(self.experiment.trials),
            "n_conditions": len(set(self.experiment.conditions))
        }
    
    def predict(self, parameters):
        discrimination, base_rate = parameters
        logit_base_rate = np.log(base_rate / (1 - base_rate))
        return 1 / (1 + np.exp(-discrimination * (self.experiment.conditions - logit_base_rate)))
    
    def negative_log_likelihood(self, parameters):
        probs = self.predict(parameters)
        log_likelihood = np.sum(self.experiment.trials * np.log(probs) + (1 - self.experiment.trials) * np.log(1 - probs))
        return -log_likelihood
    
    def fit(self):
        result = minimize(self.negative_log_likelihood, x0=[1, 0.5], bounds=[(0, None), (0, 1)])
        self._discrimination, self._base_rate = result.x
        self._logit_base_rate = np.log(self._base_rate / (1 - self._base_rate))
        self._is_fitted = True
    
    def get_discrimination(self):
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return self._discrimination
    
    def get_base_rate(self):
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return self._base_rate
