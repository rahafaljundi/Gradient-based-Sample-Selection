
from typing import Iterable

import numpy as np
import scipy.stats


def compute_correlation(X: Iterable[float],
                        Y: Iterable[float],
                        ):
    """
    Compute the Pearson correlation coefficient from a sequence of samples of a joint variables (X_i, Y_i)

    :param X: (Iterable[float]) List of samples drawn from the first variable
    :param Y: (Iterable[float]) List of samples drawn from the second variable

    :return: Pearson coefficient of the joint variable (X, Y)

    """

    # Normalize the two variables
    X_normalized = (X - np.mean(X))/np.std(X)
    Y_normalized = (Y - np.mean(Y))/np.std(Y)

    # Compute the covariance of the normalized variables
    r = np.mean(X_normalized * Y_normalized)

    return r


def compute_pearson_correlation_p_value(X: Iterable[float],
                                        Y: Iterable[float],
                                        n: int = 10000.,
                                        ):
    """
    Compute the Pearson correlation p-value by permutation

    :param X: (Iterable[float]) List of samples drawn from the first variable
    :param Y: (Iterable[float]) List of samples drawn from the second variable
    :param n: (int) Number of times we simulate a correlation coefficient by permutation

    :return: p-values of the Pearson correlation test by permutation

    """

    # Compute the correlation coefficient on (X, Y)
    r = compute_correlation(X, Y)

    # Initialize the list of values comparisons
    r_list = []

    # MC
    for i in range(n):

        # Randomly permute X and Y to make the draws independent
        X = np.random.permutation(X)
        Y = np.random.permutation(Y)

        # Compute correlation on permuted variables and compare with r
        r_list.append(float(compute_correlation(X, Y) > r))

    # Compute the p-value of the Pearson correlation test
    p_value = np.mean(r_list)

    return p_value


if __name__ == '__main__':
    X = np.random.randn(1000)
    Y = np.random.randn(1000)
    print(scipy.stats.pearsonr(X, Y))
    print(compute_correlation(X, Y))
    print(compute_pearson_correlation_p_value(X, Y))

    X = np.arange(1000)
    Y = 7*np.arange(1000) + 4
    print(scipy.stats.pearsonr(X, Y))
    print(compute_correlation(X, Y))
    print(compute_pearson_correlation_p_value(X, Y))
