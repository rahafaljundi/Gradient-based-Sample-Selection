
from typing import Iterable

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from surrogate_empirical_study.pearson_correlation_test import compute_correlation
from surrogate_empirical_study.pearson_correlation_test import compute_pearson_correlation_p_value


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_and_plot(list_of_files_names: Iterable[str],
                  plot_fontsize: int,
                  ):
    """
    Load a list of yaml files containing correlation data and plot them

    :param list_of_files_names: (list[str]) list of yaml files names
    :param plot_fontsize:

    """

    # Initialize results lists
    correlation_coefs = []
    pearson_p_values = []

    # Come across all files in the input list
    for rank, file_name in enumerate(list_of_files_names):

        # Load the current file and parse data
        file = open(file_name, 'r')
        correlation_data = yaml.load(file)
        angle_measure = np.array(correlation_data['angle_measure'])
        criterion = np.array(correlation_data['criterion'])
        file.close()

        # Compute a linear regression on the data
        biased_criterion = np.concatenate((criterion.reshape(-1, 1), np.ones_like(criterion).reshape(-1, 1)), axis=1)
        w = np.linalg.solve(np.matmul(biased_criterion.T, biased_criterion), np.matmul(biased_criterion.T, angle_measure))
        lin_pred = np.matmul(biased_criterion, w)

        # Plot the correlation data of the current file as well as the regression line
        plt.figure(rank+1, figsize=(15, 9))
        plt.plot(criterion, angle_measure, '.')
        plt.plot(criterion, lin_pred, 'r')
        plt.xticks(fontsize=18)
        plt.xlabel('Criterion', fontsize=plot_fontsize)
        plt.yticks(fontsize=18)
        plt.ylabel('Angle value', fontsize=plot_fontsize)
        plt.legend(['Generated samples', 'Linear approximation'], prop={'size': 28})
        plt.savefig(file_name.split('.')[0] + '.pdf')

        # Compute and store the pearson p-value
        correlation_coefs.append(compute_correlation(criterion, angle_measure))
        pearson_p_values.append(compute_pearson_correlation_p_value(criterion, angle_measure, n=10000))

    # Plot correlation evolution through files as well as p values
    print(correlation_coefs)
    plt.figure(figsize=(15, 9))
    plt.plot(range(3, 3+len(list_of_files_names)), correlation_coefs, '-X')
    plt.xticks(fontsize=18)
    plt.xlabel('Dimension', fontsize=plot_fontsize)
    plt.yticks(fontsize=18)
    plt.ylabel('Pearson correlation coefficient', fontsize=plot_fontsize)
    plt.savefig(os.path.join(root_dir, 'results', 'surrogate', 'Pearson_correlation_coefs.pdf'))

    print(pearson_p_values)
    plt.figure(figsize=(15, 9))
    plt.plot(range(3, 3 + len(list_of_files_names)), pearson_p_values, '-X')
    plt.xticks(fontsize=18)
    plt.xlabel('Dimension', fontsize=plot_fontsize)
    plt.yticks(fontsize=18)
    plt.ylabel('Pearson p-values', fontsize=plot_fontsize)
    plt.savefig(os.path.join(root_dir, 'results', 'surrogate', 'Pearson_p_values.pdf'))


if __name__ == '__main__':
    list_of_files_names = [os.path.join(root_dir, 'results', 'surrogate', 'correlation_data_{}_dimensions.yaml'.format(d))
                           for d in range(3, 9)]
    read_and_plot(list_of_files_names=list_of_files_names, plot_fontsize=40)
