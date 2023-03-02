from enum import auto
from time import time
import numpy as np
from scipy import fft, signal


def slow_estimate_autocovariance(timeseries, type='unbiased'):
    """

    Args:
        timeseries (_type_): timeseries
        type (str, optional): 'biased' or 'unbiased' corresponding to equations (2.2.3) and (2.2.4) in "Stoica2005 spectral analysis of signals" respectively. Defaults to 'unbiased'.

    Returns:
        _type_: _description_
    """
    allowed_types = ['biased', 'unbiased']
    if type not in allowed_types:
        raise ValueError('unkown "type" parameter value should be one of {}'.format(allowed_types))

    length, dim = timeseries.shape

    sum_collection = dict()
    count_collection = dict()

    for k in range(len(timeseries)):
        for l in range(len(timeseries)):
            diff = k-l

            if diff not in sum_collection.keys():
                assert diff not in count_collection.keys()

                current_sum = np.zeros((dim,dim))
                current_count = 0
            else:
                current_sum = sum_collection[diff]
                current_count = count_collection[diff]

            xk = timeseries[k, :, np.newaxis]
            xlT =  timeseries[l, np.newaxis, :]
            product = xk @ xlT

            current_sum += product
            current_count += 1

            sum_collection[diff] = current_sum
            count_collection[diff] = current_count

    
    # finalize 
    final_collection = dict()

    for k in sorted(sum_collection.keys()):
        summ = sum_collection[k]
        ccount = count_collection[k]
        if type=='unbiased':
            avg = summ/ccount 
        elif type=='biased':
            avg = summ/length 
        else:
            RuntimeError()

        final_collection[k] = avg

    return final_collection


def symetrize_autocovariance(onesided_autocovariance):
    """ Symmetrically completes the covariance estimate

    Args:
        onesided_autocovariance (np.array of shape (length, dim, dim)): the one-sided autocovariance function
    Returns:
        np.array: shape (2*length - 1, dim, dim), the middle position corresponds to delay zero (i.e. tau=0)
    """

    length, dim, dim1 = onesided_autocovariance.shape

    assert dim1 == dim
    out = np.zeros((2*length-1, dim, dim))

    negative_half = np.flip(onesided_autocovariance[1:, :, :], axis=0).transpose([0, 2, 1])
    out[:(length-1), :, :] = negative_half
    out[(length-1):, :, :] = onesided_autocovariance

    return out

def estimate_autocovariance(timeseries):
    """Estimates symmetric autocovariance
    """
    return symetrize_autocovariance(estimate_onesided_autocovariance(timeseries))

def estimate_onesided_autocovariance(timeseries):
    """ Computes the onesided autocovariance of a timeseries, i.e. no negative delays
    Args:
        timeseries (np.array): numpy array of shape (length, dim), first index starts with delay 0, delay 1, etc
    Returns:
        np.array: numpy array of shape (length, dim, dim)
    """
    length, dim = timeseries.shape
    
    out = np.zeros((length, dim, dim))

    for delay in range(length): # max delay is one less then the timeseries
        if delay == 0:
            left = timeseries
            right = timeseries
        else:
            left = timeseries[delay:] # left is later than right
            right = timeseries[:-delay] # duration, dim, right is earlier

        left_reshaped = left[:, :, np.newaxis] # shape duration, dim, 1
        right_reshaped = right[:, np.newaxis, :] # shape duration, 1, dim
        # the last two dimensions contain the matrices to multiply here the column vector and row vector

        outer = np.matmul(left_reshaped, right_reshaped) # shape duration, dim, dim

        out[delay, :, :] = np.mean(outer, axis=0)

    return out


def freqs_for_spectrum(n):
    """
        the frequency labels for the spectrum returned by compute_spectrum
    """
    return np.arange(n)/n

def compute_spectrum(autocovariance_function):
    """ Compute the spectrum, i.e. the fourier transform of the autocovariance function, f_XX(lambda) in Dahlhaus
    
    args:
        autocovariance_function (np.array): as returned from the estimate_auocovariance() method, should have shape (n, dim, dim), n should be odd, and the 0 point is considered to be in the middle, i.e. autocovariance_function can be seen as indexed -maxTau, ..., 0, ..., maxTau

    return:
        np.array of the same shape with the fourier transform, first index indexes over frequencies. i.e. result[k] = xHat( k / n) = xHat( k/(2*maxTau+1) ), see freqs_for_spectrum
    """ 
    more_bits = autocovariance_function.astype(np.longdouble)
    n, dim1, dim2 = more_bits.shape
    assert dim1 == dim2

    if n%2 != 1:
        raise ValueError('Autocovariance sequence should have an odd number of values. It can be interpreted as indexed from -tauMax, ..., 0, ... tauMax.')

    maxTau = int((n-1)/2)

    assert n == (2*maxTau + 1)
    
    transformed_raw = fft.fft(more_bits, axis=0)

    # autocovariance_function[0] corresponds to autocovariance(-tauMax), i.e. it is a shifted version of the original signal, we compensate this
    frequency_shift_coef = np.arange(n) * (2j*np.pi* (maxTau/(2*maxTau + 1)))

    frequency_shift = np.exp(frequency_shift_coef)

    transformed = frequency_shift[:, np.newaxis, np.newaxis] * transformed_raw

    return transformed

def compute_inverse_scaled_spectrum(autocovariance_function):
    """ Compute the inverse scaled spectrum from the estimate_covariance function, this is d(lambda) in Dahlhaus p.161
    
    args:
        autocovariance_function (np.array): as returned from the auocovariance() method, should have shape (max_delay, dim, dim)

    return:
        np.array of the same shape with the fourier transform
    """  
    maxdelay, dim, dim2 = autocovariance_function.shape
    assert dim==dim2

    spectrum = compute_spectrum(autocovariance_function)
    inverse_spectrum = np.linalg.inv(spectrum)


    outD = np.zeros((maxdelay, dim, dim), dtype=np.clongdouble)
    for f in range(maxdelay):
        spectrumThisFrequency = inverse_spectrum[f, :, :]
        diag = np.diagonal(spectrumThisFrequency)

        scaling = diag ** (-0.5)

        scalingmat = np.diag(scaling)

        a = np.matmul(scalingmat, spectrumThisFrequency)
        b = np.matmul(a, scalingmat)

        outD[f,:,:]=b

    return outD

def estimate_autocovariance_from_collection(timeseries_collection):
    return symetrize_autocovariance(estimate_onesided_autocovariance_from_collection(timeseries_collection))

def estimate_onesided_autocovariance_from_collection(timeseries_collection):
    """Compute the autocovariance function from a collection of different length timeseries

    Args:
        timeseries_collection (list[np.array]): list of np.arrays, each with dimension [length, dim], length can be different

    Returns:
        np.array: numpy array of shape (length, dim, dim)
    """
    max_length = -1
    dim = timeseries_collection[0].shape[1]
    for timeseries in timeseries_collection:
        leng, this_dim = timeseries.shape
        if leng > max_length:
            max_length = leng

        if dim != this_dim:
            raise ValueError("All timeseries should have the same dimension. We found {} and {}".format(dim, this_dim))
        
    n_of_series = len(timeseries_collection)

    collector = np.ones((n_of_series, max_length, dim, dim))*np.nan

    for i, timeseries in enumerate(timeseries_collection):
        this_covar = estimate_onesided_autocovariance(timeseries)

        collector[i, :len(timeseries), :, :] = this_covar

    out = np.nanmean(collector, axis=0)
    return out

def pad_autocovariance(autocovariance, padding_factor):
    """pad a symmetric autocovariance function

    Args:
        autocovariance (array, (2*length - 1, dim, dim)): 
        padding_factor (int): 
    Returns:
        zeropadded 
    """

    factor = int(padding_factor)

    total_length, dim, dim1 = autocovariance.shape

    assert total_length%2 == 1

    max_delay = int((total_length-1)/2)

    padding = max_delay*padding_factor
    
    return np.pad(autocovariance, [(padding,padding), (0,0), (0,0)], 'constant', constant_values=0)

def interaction_model_from_timeseries_collection(timeseries_collection, zero_padding_factor = None, norm='l1'):
    """Compute the interaction model from a collection of different length timeseries

    Args:
        timeseries_collection (list[np.array]): list of np.arrays, each with dimension [length, dim]
    Returns:
        np.array: numpy array of shape (dim, dim), high values indicate strong interactions
    """

    autocovariance_estimate = estimate_autocovariance_from_collection(timeseries_collection)

    if zero_padding_factor is not None:
        autocovariance_estimate = pad_autocovariance(autocovariance_estimate, zero_padding_factor)

    scaled_spectrum = compute_inverse_scaled_spectrum(autocovariance_function=autocovariance_estimate)

    if norm=='l1':
        magnitude = np.sum(np.abs(scaled_spectrum), axis=0)
    elif norm == 'l2':
        magnitude = np.sqrt(np.sum(np.abs(scaled_spectrum)**2, axis=0))
    else:
        raise ValueError('unkown norm')

    interaction_matrix = magnitude/scaled_spectrum.shape[0]
    return interaction_matrix



def _standard_biased_ACS_window(maxlength):
    # as in in equation 2.2.4 of Stoica2005 spectral analysis of signals
    assert maxlength % 2 == 1

    length_of_signal = int((maxlength +1)/2)

    scaling = (length_of_signal - np.arange(length_of_signal))/length_of_signal

    # now we have to symmegrize this
    negative_half = np.flip(scaling[1:])
    assert negative_half[-1] == scaling[1]

    out = np.concatenate([negative_half, scaling])
    return out


def window_autocovariance(covariance, window_name):
    if window_name is None:
        window = np.ones(covariance.shape[0])
    elif window_name == 'biased_estimator':
        window = _standard_biased_ACS_window(covariance.shape[0])
    else:
        window = signal.windows.get_window(window_name, covariance.shape[0]+1)[1:] #we want the window to be exactly symmetric around the origin of the autocovariance function


    issym = np.all(np.isclose(window, window[::-1]))
    assert issym

    tapered_covariance = covariance * window[:, np.newaxis, np.newaxis]

    return tapered_covariance



def estimate(normalized_sequences, window_name='blackman', padding_factor=1000, subsample_data=None, bootstrap=True, norm='l1'):
    """
        normalized_sequence: List[np.array], collection of the time-series, one per patient
        window_name: name of the windowing function which is applied to the estimated autocovariance, or "biased_estimator" to use equation 2.2.4 from Stoica2005 spectral analysis of signals
        padding_factor: integer, how many zeros to add to the autocovariance function before computing the fft
        subsample_data: use only a fraction of the data for the estimation
    """
    if subsample_data is not None:
        assert (subsample_data <= 1) and (subsample_data>0)
        
        n_samples = len(normalized_sequences)
        choice_indices = np.random.choice(n_samples, int(np.floor(n_samples*subsample_data)), replace=bootstrap)
        sequences = [normalized_sequences[cidx] for cidx in choice_indices]
    else:
        sequences = normalized_sequences
    
    covariance = estimate_autocovariance_from_collection(sequences)
    tapered_covariance = window_autocovariance(covariance=covariance, window_name=window_name)
    padded_covariance = pad_autocovariance(tapered_covariance, padding_factor)
    scaled_spectrum = compute_inverse_scaled_spectrum(padded_covariance)

    if norm=='l1':
        magnitude = np.sum(np.abs(scaled_spectrum), axis=0)
    elif norm == 'l2':
        magnitude = np.sqrt(np.sum(np.abs(scaled_spectrum)**2, axis=0))
    else:
        raise ValueError('unkown norm')

    interaction_matrix = magnitude/scaled_spectrum.shape[0]
    return interaction_matrix