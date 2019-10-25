
import numpy as np


class Angle(object):
    """
    Class Angle
    """

    def __init__(self, G: np.array):
        """
        Store parameters of the angle such as the dimension of the space and the family of vector that defines the angle

        :param G: (ndarray) A square ndarray of dtype float that encodes a family of n vectors in an n dimensional
        space, for a certain n.

        """

        # Store dimension of the space
        self.dimension = G.shape[0]

        # Check that the number of dimensions correspond to the number of vectors
        assert G.shape == (self.dimension, self.dimension)

        # Normalize each and every vector
        for i in range(G.shape[1]):
            G[:, i] = G[:, i] / np.sqrt(np.sum(G[:, i] * G[:, i]))

        # Store G
        self.G = G

    def check_feasibility(self, v: np.array):
        """
        Compute the feasibility of a given vector v defined by:
        v is feasible iif

        .. math::
            \\forall i \in \left[|1; n|\right], ~ \left< v \mid G_i \right> \leq 0

        :param v: (ndarray) A n dimensional vector

        :return: 1 if v is feasible, 0 otherwise

        """

        # Come across all vector in G
        for i in range(self.dimension):

            # Compute the inner between v and G_i
            inner = np.sum(v * self.G[:, i])

            # Break the loop and return 0 as soon as the feasible condition is violated
            if inner >= 0:
                return 0.

        # Return 1 if v is feasible
        return 1.

    def angle_estimation(self, n: int):
        """
        Compute an estimation of the solid angle defined by G, by MC method.
        The angle is normalized so that the entire sphere described an angle of 1.

        :param n: (int) number of draws in the MC procedure

        :return: (float) angle estimation

        """

        # Initialize the probability that a random vector falls into the feasible set
        p = 0.

        # Draw n vector to update p, the estimation of the angle
        for i in range(n):

            # Draw v as a standard normal variable so that after normalization, v is a uniform r.v. on the sphere
            v = np.random.randn(self.dimension)

            # Update p
            p += self.check_feasibility(v)

        # Compute the final angle estimation
        p /= n

        return p

    def angle_heuristic(self):
        """
        Compute the heuristic proposed in the ICML2019 paper "Online continual learning with no task boundaries" by
        Rahaf Aljundi, Min Lin, Baptiste Goujaud and Yoshua Bengio.

        .. math::
            \\Vert \\sum_i G_i \\Vert_2^2

        :return: (float) the value obtained by the proposed heuristic

        """

        # Compute the sum of all the gradients in G
        sum_grads = np.sum(self.G, 1)

        # Compute the square of the L2 norm of the above
        heuristic_value = float(np.sum(sum_grads ** 2))

        return heuristic_value
