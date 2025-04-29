import numpy as np

def lms_filter(u, y, L, mu):
    """
    Least Mean Square (LMS) Filter

    Parameters
    ----------
    u : array_like of shape (K,)
        Input Signal u (will be filtered by h)
    y : array_like of shape(K,)
        Reference Signal y (will be compared with y_hat)
    L : int
        LMS filter length, has to be a positive integer
    mu : float
        LMS step size, has to be a positive number

    Return
    ------
    y_hat : np.ndarray of shape (K,)
        filter output estimation
    h_hat : np.ndarray of shape (K, filter_length)
        estimated filter coefficients
    e : np.ndarray of shape (K,)
        Error between estimate and reference signal
    """

    K = len(u) # length of signal
    y_hat = np.zeros(K) # allocate output estimate memory
    e = np.zeros(K) # allocate error memory
    h_hat = np.zeros((K+1, L)) # filter coefficients initialization

    dummy_array = np.random.randn(L)

    # iterate through the signal
    for k in range(L, K):

        # Convolution between the input signal and the filter coefficients (note that the input snippet requires inverse indexing in the convolution)

        # 1.  x = input signal snippet reversed
        x = u[k-L:k][::-1] # CHANGE THIS LINE (right side of equal sign) !!!

        # 2. output_estimat[k] = inner product of filter_coefficients and x
        y_hat[k] = np.dot(h_hat[k], x)

        # Calulate the error 
        e[k] = y[k] - y_hat[k]

        # Update the filter coefficients
        h_hat[k+1] = h_hat[k]+mu*e[k]*x # CHANGE THIS LINE (right side of equal sign) !!!


    return y_hat, h_hat[:K], e
