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
    
    def test_initialization(self):
        # test constructor and error handling
        self.assertEqual(self.model.experiment, self.experiment)
        self.assertFalse(self.model._is_fitted)
        self.assertIsNone(self.model._base_rate)
        self.assertIsNone(self.model._logit_base_rate)
        self.assertIsNone(self.model._discrimination)
        # error handling
        with self.assertRaises(ValueError):
            SimplifiedThreePL(None)
        with self.assertRaises(ValueError):
            SimplifiedThreePL(MockExperiment([], []))
        # mismatched lengths
        with self.assertRaises(ValueError):
            SimplifiedThreePL(MockExperiment([1.0, 2.0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    
    def test_prediction(self):
        # test output between 0 and 1 (inclusive)
        parameters = [1.0, 0.5] # discrimination, base rate
        predictions = self.model.predict(parameters)
        self.assertTrue(np.all(predictions >= 0.0) and np.all(predictions <= 1.0))

        # test higher base rate results in higher probabilities
        pred_low_base = self.model.predict([1.0, 0.2])
        pred_high_base = self.model.predict([1.0, 0.8])
        self.assertTrue(np.all(pred_high_base > pred_low_base))

        # test higher difficulty results in lower probabilities
        pred_low_diff = self.model.predict([1.0, 0.2])
        unique_conditions = np.unique(self.experiment.conditions) # selects unique condition labels (ex: easy, medium, hard)
        condition_indices = {c: np.where(np.array(self.experiment.conditions) == c)[0] for c in unique_conditions} # maps condition labels to indices
        # check that prediction accuracy decrease as difficulty increases
        for i in range(len(self.experiment.conditions)):
            easier_cond =unique_conditions[i]
            harder_cond = unique_conditions[i+1]
            self.assertGreater(
                np.mean(pred_low_diff[condition_indices[easier_cond]]),
                np.mean(pred_low_diff[condition_indices[harder_cond]])
            )
        
        # difficulty affects with negative discrimination (higher accuracy for harder conditions)
        pred_neg = self.model.predict([-1.0, 0.5])
        for i in range(4):
            easier_cond = unique_conditions[i]
            harder_cond = unique_conditions[i+1]
            self.assertLess(
                np.mean(pred_neg[condition_indices[easier_cond]]),
                np.mean(pred_neg[condition_indices[harder_cond]])
            )
        
        # test known parameter matches expected output
        test_conditions = np.array([1.0, 0.0, -1.0])
        test_model = SimplifiedThreePL(MockExperiment(test_conditions, np.array([1, 1, 0])))
        # apply the prediction formula for correct trials
        a, c = 1.0, 0.5
        logit_c = np.log(c / (1-c))  # should be 0 for c=0.5
        expected_probs = 1 / (1 + np.exp(-a * (test_conditions - logit_c)))
        actual_probs = test_model.predict([a, c])
        np.testing.assert_almost_equal(actual_probs, expected_probs)

    def test_parameter_estimation(self):
        # test that NLL improves after fitting
        # NLL = measures how bad the model is at predicting data (opposite of log-likelihood prediction)
        params = [1.0, 0.5]
        initial_nll = self.model.negative_log_likelihood(params)
        # test with fitted model
        self.model.fit()
        fitted_params = [self.model._discrimination, self.model._base_rate]
        fitted_nll = self.model.negative_log_likelihood(fitted_params)
        self.assertLess(fitted_nll, initial_nll) # nll should decrease (improve) after fitting

        # test that larger discrimination estimate is returned for steeper learning curves
        # conceptually: steeper learning curve = faster learning = higher discrimination ( easier to distinguish between conditions)
        conditions = [2.0, 1.0, 0.0, -1.0, -2.0]
        # low discrimination (flat curve, slower)
        low_disc_trials = np.array([
            [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],  # 6/10 correct
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # 5/10 correct
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # 5/10 correct
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # 5/10 correct
            [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]   # 4/10 correct
        ]).flatten().round().astype(int)
        # high discrimination (steep curve, faster)
        high_disc_trials = np.array([
            [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],  # 9/10 correct
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],  # 8/10 correct
            [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],  # 7/10 correct
            [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],  # 6/10 correct
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]   # 5/10 correct
        ]).flatten().round().astype(int)
        # run experiments
        low_disc_exp = MockExperiment(conditions * 10, low_disc_trials)
        high_disc_exp = MockExperiment(conditions * 10, high_disc_trials)
        
        low_disc_model = SimplifiedThreePL(low_disc_exp)
        high_disc_model = SimplifiedThreePL(high_disc_exp)
        
        low_disc_model.fit()
        high_disc_model.fit()
        
        # Check that the discrimination parameter is higher for steeper curve
        self.assertGreater(high_disc_model.get_discrimination(), low_disc_model.get_discrimination())

    def test_integration(self):
        # test that the model can integrate difficulty and accuracy rates across multiple conditions to form a prediction
        # test param stability after multiple fitting rounds
        self.model.fit()
        original_params = [self.model.get_discrimination(), self.model.get_base_rate()]

        # reset and refit 3 times
        for _ in range(3):
            # reset params
            self.model._is_fitted = False
            self.model._discrimination = None
            self.model._base_rate = None
            self.model._logit_base_rate = None
            #refit
            self.model.fit()
            # compare with original
            current_params = [self.model.get_discrimination(), self.model.get_base_rate()]
            for i in range(len(original_params)):
                self.assertAlmostEqual(current_params[i], original_params[i], places=2)
        
        # test with different accuracy rates
        conditions = [2.0, 1.0, 0.0, -1.0, -2.0]
        accuracy_rates = [0.95, 0.90, 0.75, 0.60, 0.55]
        # construct experiment
        # simulate 100 trials for each accuracy rate
        trials = np.concatenate([np.concatenate([np.ones(int(rate*100)), np.zeros(100-int(rate*100))]) 
                            for rate in accuracy_rates])
        conditions_repeated = np.repeat(conditions, 100)
        integration_model = SimplifiedThreePL(MockExperiment(conditions, trials))
        # fit model on simulated trial data
        integration_model.fit()
        params = [integration_model.get_discrimination(), integration_model.get_base_rate()]
        predictions = integration_model.predict(params)
        
        # predict accuracy rates with the fitted model
        for i, condition in enumerate(conditions):
            indices = np.where(conditions_repeated == condition)[0]
            pred_mean = np.mean(predictions[indices])
            self.assertAlmostEqual(pred_mean, accuracy_rates[i], places=2)
        
        # Verify monotonically decreasing predictions as difficulty increases
        for i in range(len(conditions)-1):
            indices_easier = np.where(conditions_repeated == conditions[i])[0]
            indices_harder = np.where(conditions_repeated == conditions[i+1])[0]
            self.assertGreater(np.mean(predictions[indices_easier]), np.mean(predictions[indices_harder]))
    

    def test_corruption(self):
        #