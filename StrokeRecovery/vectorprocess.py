# tools to generate an autoregressive process which is used to generate sample data for testing the interactionmodel

import numpy as np


class VectorAutoregressiveProcess:

    def __init__(self, parametrization_matrix, noise_covariance=None):
        """ A vector autoregressive process as described in Section 4 of Dahlhaus

        Args:
            parametrization_matrix (np.matrix): numpy array of shape (max_delay, dim, dim) # the first coordinate indexes the delay starting from delay 1, i.e. [0] delay 1, [1] delay 2, ...
            noise_covariance (np.matrix): numpy array of shape (dim, dim)
        """
        sshape = parametrization_matrix.shape
        if len(sshape) != 3:
            raise ValueError("parametrization_matrix must be a 3D array")

        maxdelay, dim, dim2 = sshape
        if dim != dim2:
            raise ValueError("parametrization_matrix must be square")

        if noise_covariance is None:
            noise_covariance = np.eye(dim)
            
        if dim != noise_covariance.shape[-1]:
            raise ValueError("noise_covariance must be a square matrix with first dimesion equal to the dimension of the process")

        self.reversed_parametrization_matrix = np.flip(parametrization_matrix, axis=0) # [-1] delay by one, [-2] delay by two, etc.


        self.noise_covariance = noise_covariance


    @property
    def dim(self):
        return self.reversed_parametrization_matrix.shape[-1]

    @property
    def max_delay(self):
        return self.reversed_parametrization_matrix.shape[0]

    def draw_one_timeseries(self, length):
        """ Draws from the random process

        Args:
            length (int): length of the timeseries to draw

        Returns:
            np.array: numpy array of shape (length, dim)
        """

        noise_samples = np.random.multivariate_normal(np.zeros(self.dim), self.noise_covariance, length) # shape (length, dim)


        out = np.zeros((length, self.dim))

        for t in range(length):
            if t > 0:
                start_indx = max(t-self.max_delay, 0)
                prevs = out[ start_indx : t , :, None]
                relevant_dealy_coefficients = self.reversed_parametrization_matrix[-t:, :, :] # stack of matrices residing in last to columns
                autoregressive_part_weights = np.matmul(relevant_dealy_coefficients, prevs) # stack of vectors, 
                autoregressive_part0 = np.sum(autoregressive_part_weights, axis=0)
                autoregressive_part = np.squeeze(autoregressive_part0)
            else:
                autoregressive_part = np.zeros((self.dim,))



            out[t, :] = autoregressive_part + noise_samples[t, :]

        return out

    def draw_collection_of_timeseries(self, min_l_per_draw, max_l_per_draw, n_draws):
        """ Draw a collection of n_draws timeseries from the process. The timeseries will have different lengths,
        between min_l_per_draw and max_l_per_draw.

        Args:
            min_l_per_draw (int): 
            max_l_per_draw (int): 
            n_draws (int): 

        Returns:
            _type_: list of n_draws numpy arrays, each of shape (length, dim), where length is between min_l_per_draw and max_l_per_draw
        """
        collector = list()
        for i in range(n_draws):
            length_of_this_draw = np.random.randint(min_l_per_draw, max_l_per_draw+1)
            draw = self.draw_one_timeseries(length_of_this_draw)

            collector.append(draw)

        return collector