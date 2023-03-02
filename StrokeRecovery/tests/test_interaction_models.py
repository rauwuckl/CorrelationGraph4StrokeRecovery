from difflib import Differ
from re import M
import unittest
from StrokeRecovery.vectorprocess import VectorAutoregressiveProcess
from StrokeRecovery import interactionmodel as IM

import numpy as np


class TestInteractionModel(unittest.TestCase):

    def setUp(self):
        parametrization_matrix = np.zeros((4,3,3))
        parametrization_matrix[0, 0, 1] = 3 # entry 0 is influencecd by entry 1, one timesteps later

        self.process = VectorAutoregressiveProcess(parametrization_matrix=parametrization_matrix)



    def test_autocovariance_estimation(self):
        n=100

        timeseries = self.process.draw_one_timeseries(n)

        autocovariance_onesided = IM.estimate_onesided_autocovariance(timeseries) 
        autocov = IM.symetrize_autocovariance(autocovariance_onesided)

        autocovariance_slow = IM.slow_estimate_autocovariance(timeseries)

        for t in [99, 0, 1, 2, 67, -23, -1, -99, 45]:
            acv = autocov[(n-1)+t]
            acvs = autocovariance_slow[t]
            self.assertTrue(np.all(acv==acvs))
            pass


    def test_spectrum(self):
        def test_autocov_properties(autocov):
            n, d, _ = autocov.shape

            for k in range(n//2-1):
                diff = autocov[k].T - autocov[-k-1] # k[tau] = k[-tau]^T
                self.assertTrue(np.all(np.isclose(diff,0)))


        def test_spec_properties(spec):
            n, d, _ = spec.shape
            self.assertTrue(np.all(np.isclose(np.imag(spec[0]), 0))) # f(0) is real
            for k in range(2,n-2):
                this_freq = spec[k]
                self.assertFalse(np.all(np.isclose(np.imag(this_freq), 0))) # not only real elements

                diag = this_freq.diagonal()
                self.assertTrue(np.all(np.isclose(np.imag(diag), 0))) # diagonal elements are real elements


                # check hermitian 
                diff = this_freq - this_freq.conj().T
                self.assertTrue(np.all(np.isclose(diff, 0))) 

        timeseries = self.process.draw_one_timeseries(800)

        autocovariance0 = IM.estimate_autocovariance(timeseries)
        autocovariance = IM.window_autocovariance(autocovariance0, 'blackman')
        spectrum = IM.compute_spectrum(autocovariance)

        test_spec_properties(spectrum)
        test_autocov_properties(autocovariance0)


        dim = 3
        maxdelay = 4
        n_draws = 1000
        min_l_per_draw, max_l_per_draw = (3, 7)
        parametrization = np.zeros((maxdelay, dim, dim))
        parametrization[0, 0, 1] = 1 # 0 influenced by 1
        parametrization[2, 2, 0] = 2 # 2 is influenced by 0
        iInd = [0, 0]
        jInd = [1, 2]
        process = VectorAutoregressiveProcess(parametrization_matrix=parametrization)
        sequences = process.draw_collection_of_timeseries(min_l_per_draw=min_l_per_draw, max_l_per_draw=max_l_per_draw, n_draws=n_draws)

        covariance = IM.estimate_autocovariance_from_collection(sequences)
        spectrum = IM.compute_spectrum(covariance)
        
        test_autocov_properties(autocov=covariance)
        test_spec_properties(spectrum)


    def test_autocovariance_estimation(self):
        mLength = 7

        collection = self.process.draw_collection_of_timeseries(min_l_per_draw=3, max_l_per_draw=mLength, n_draws=1000)

        autocovariance_estimate = IM.estimate_onesided_autocovariance_from_collection(collection)

        print(autocovariance_estimate)

        self.assertEqual(autocovariance_estimate.shape, (mLength, self.process.dim, self.process.dim))

        self.assertEqual(np.isnan(autocovariance_estimate).sum(), 0)

    def test_independence_model(self):
        padding_factor = 3

        dim = 3
        maxdelay = 4
        n_draws = 1000
        min_l_per_draw, max_l_per_draw = (3, 7)
        parametrization = np.zeros((maxdelay, dim, dim))
        parametrization[0, 0, 1] = 1 # 0 influenced by 1
        parametrization[2, 2, 0] = 2 # 2 is influenced by 0
        # there will also be a correlation between 2 and 1, but this should not show as conditioned on 0 they are unrelated
        iInd = [0, 0]
        jInd = [1, 2]
        process = VectorAutoregressiveProcess(parametrization_matrix=parametrization)

        collection_of_draws = process.draw_collection_of_timeseries(min_l_per_draw=min_l_per_draw, max_l_per_draw=max_l_per_draw, n_draws=n_draws)

        sum_spectrum = IM.estimate(collection_of_draws, padding_factor=padding_factor, window_name='blackman') 

        # check that we recover the correct indices
        predicted_correlation = sum_spectrum.copy()
        predicted_correlation[np.tril_indices(dim)] = 0

        # get indices that should be predicted
        # iInd, jInd = np.nonzero(np.abs(parametrization).sum(axis=0) != 0)
        values_that_should_be_high = predicted_correlation[iInd, jInd]

        predicted_correlation[iInd, jInd] = 0
        other_values = predicted_correlation.flatten()

        self.assertTrue( np.min(values_that_should_be_high) > np.max(other_values) )


    def test_windowing(self):
        l = 5
        d=3

        total_l = 2*l +1

        autocov = np.random.rand(total_l, d, d)

        windowed = IM.window_autocovariance(autocov, 'hann')
        windowed = IM.window_autocovariance(autocov, 'blackman')
        windowed = IM.window_autocovariance(autocov, 'hamming')


    def test_padding(self):

        dim = 3
        length = 6

        pad_factor = 2

        example_onesided = np.random.rand(length, dim, dim)
        sym = IM.symetrize_autocovariance(example_onesided)

        padded = IM.pad_autocovariance(sym, 2)


        pad_size = (length-1)*pad_factor


        self.assertTrue(np.all(padded[:pad_size]==0))
        self.assertFalse(np.all(padded[pad_size+1]==0))

        self.assertTrue(np.all(padded[-pad_size:]==0))
        self.assertFalse(np.all(padded[-pad_size-1]==0))