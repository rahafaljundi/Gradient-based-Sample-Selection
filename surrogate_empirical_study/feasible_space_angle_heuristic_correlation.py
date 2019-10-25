
import os
import yaml
from tqdm import tqdm

import numpy as np

from surrogate_empirical_study.angle import Angle


def collect_correlation_data(d: int=3):
    """
    Collect correlation data, namely angle MC estimation and heuristic value

    :param d: (int) number of dimension

    """
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n d={}".format(d))

    # Initialize correlation_data dict
    correlation_data = dict()
    correlation_data['angle_measure'] = []
    correlation_data['criterion'] = []

    # Repeat the list update a number of time equivalent to the number of needed data
    for j in tqdm(range(1000)):

        # Generate the d grads in d dimensional space that define the feasible set
        projected_norm = np.random.rand()
        grads = np.random.randn(d, d)
        for column in range(d):
            grads[:, column] = grads[:, column]/np.sqrt(np.sum(grads[:, column] ** 2))
        for column in range(1, d):
            grads[:, column] = projected_norm * grads[:, 0] + np.sqrt(1-projected_norm**2) * (grads[:, column] - np.sum(grads[:, column] * grads[:, 0]) * grads[:, 0])

        # Instantiate Angle
        angle = Angle(grads)

        # Store the MC angle estimation with variance of sqrt(p/(1000*2^d)) where p is the estimated angle
        # Then if p = 1/2^d, variance = p/sqrt(1000)
        correlation_data['angle_measure'].append(angle.angle_estimation(1000*2**d))

        # Store the value given by the heuristic
        correlation_data['criterion'].append(angle.angle_heuristic())

    # Dump the data on the hard drive
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_folder = os.path.join(root_dir, 'results', 'surrogate')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    yaml_file_name = os.path.join(result_folder, 'correlation_data_{}_dimensions.yaml'.format(d))
    yaml_file = open(yaml_file_name, 'w')
    yaml.dump(correlation_data, yaml_file)
    yaml_file.close()


if __name__ == '__main__':
    collect_correlation_data(d=3)
    collect_correlation_data(d=4)
    collect_correlation_data(d=5)
    collect_correlation_data(d=6)
    collect_correlation_data(d=7)
    collect_correlation_data(d=8)
