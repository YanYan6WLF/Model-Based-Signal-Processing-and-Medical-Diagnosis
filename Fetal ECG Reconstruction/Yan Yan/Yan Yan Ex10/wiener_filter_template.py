import numpy as np

def wiener_filter(u, y, L):
    """
    Wiener Filter

    Parameters
    ----------
    u : array_like of shape (K,)
        Input Signal u (will be filtered by h)
    y : array_like of shape(K,)
        Reference Signal y (will be compared with y_hat)
    L : int
        Filter length, has to be a positive integer

    Return
    ------
    y_hat : np.ndarray of shape (K,)
        filter output estimation
    h_hat : np.ndarray of shape (filter_length,)
        estimated filter coefficients
    """

    K = len(u) # length of signal
    dummy_array = np.zeros(K)
    dummy_array2 = np.random.randn(L)

    S = np.zeros((K, L)) # memory allocation

    # create S matrix
    for k in range(L, K):

        # 1. crate basis vector p_k and save it into the S matrix (note the starting index if k)
        S[k:,] = u[k-L:k][::-1]  # CHANGE THIS LINE (right side of equal sign) !!!

    # 2. estimate filter coefficients h_hat
    h_hat = np.linalg.inv(S.T@S)@S.T.dot(y)

    # 3. filter output estimate y_hat
    y_hat = S @ h_hat # CHANGE THIS LINE 

    return y_hat, h_hat
