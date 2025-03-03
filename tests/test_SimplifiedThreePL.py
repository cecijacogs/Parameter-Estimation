# unit tests
import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL

class MockExperiment:
    def __init__(self, conditions, trials):
        self.conditions = conditions
        self.trials = trials

class TestSimplifiedThreePL(unittest.TestCase):

    def setUp(self):
        # set up test parameters
        conditions = [2.0, 1.0, 0.0, -1.0, -2.0]
        trials = np.array([
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],  # 7/10 correct
            [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],  # 7/10 correct
            [1, 0, 1, 0, 0, 1, 0, 1, 0, 1],  # 5/10 correct
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # 3/10 correct
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]   # 1/10 correct
        ]).flatten()

        # create experiment with 10 trials per condition
        self.experiment = MockExperiment(conditions * 10, trials)
        self.model = SimplifiedThreePL(self.experiment)
    
