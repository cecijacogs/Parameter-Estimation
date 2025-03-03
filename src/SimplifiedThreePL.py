import numpy as np
from scipy.optimize import minimize

class SimplifiedThreePL:
    def __init__(self, experiment):
        # input validation
        if experiment is None:
            raise ValueError("Experiment cannot be None")
        if not hasattr(experiment, 'conditions') or not hasattr(experiment, 'trials'):
            raise ValueError("Experiment must have conditions and trials attributes")
        if len(experiment.conditions) == 0 or len(experiment.trials) == 0:
            raise ValueError("Experiment conditions and trials cannot be empty")
        if len(experiment.conditions) != len(experiment.trials):
            raise ValueError("Number of conditions must match number of trials")
        
        # initialize model parameters
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

        # handle division by zero
        # clip the base rate in the range of 0 - 1
        base_rate = np.clip(base_rate, 1e-10, 1-1e-10)
        logit_base_rate = np.log(base_rate / (1 - base_rate))

        # cast conditions from a list into an array before performing operations
        conditions = np.array(self.experiment.conditions)
        return 1 / (1 + np.exp(-discrimination * (self.experiment.conditions - logit_base_rate)))
    
    def negative_log_likelihood(self, parameters):
        probs = self.predict(parameters)
        probs = np.clip(probs, 1e-10, 1-1e-10) # clip the probabilities to avoid log(0)

        log_likelihood = np.sum(self.experiment.trials * np.log(probs) + (1 - self.experiment.trials) * np.log(1 - probs))
        return -log_likelihood
    
    def fit(self):
        result = minimize(self.negative_log_likelihood, x0=[1, 0.5], bounds=[(0.001, None), (0.001, 0.999)]) # exclude 0 and 1 from the search space
        self._discrimination, self._base_rate = result.x
        self._logit_base_rate = np.log(self._base_rate / (1 - self._base_rate))
        self._is_fitted = True
    
    def get_discrimination(self):
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        if self._discrimination is None:
            raise ValueError("Discrimination parameter is missing.")
        return self._discrimination
    
    def get_base_rate(self):
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        if self._base_rate is None:
            raise ValueError("Base rate parameter is missing.")
        return self._base_rate
