# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" Curve-fitting code """

from typing import List

import osqp
import numpy as np
from scipy.sparse import csc_matrix


class MonotonicIncreasingCurveFit:
    """
    This class provides mechanism to fit a curve using a monotonically increasing function with a high polynomial degree
    """

    # This is a known pylint bug. Not able to understand type aliases
    # pylint:disable=undefined-variable
    Coordinates = List[float]

    @staticmethod
    def _solve_qp(p, q, g):
        """
            Quadratic solver. Uses OSQP.
            Solves:
            Minimize     1/2 x^T Px - q^Tx
            Subject to   Gx >= 0
        :param p: The P term
        :param q: The q term
        :param g: The G term
        :return: The qp solution
        """
        solver = osqp.OSQP()
        upper_constraint = np.inf * np.ones(g.shape[0])
        lower_constraint = np.zeros(g.shape[0])
        solver.setup(csc_matrix(p), -q, csc_matrix(g), lower_constraint, upper_constraint, verbose=False)
        results = solver.solve()

        return results.x

    @classmethod
    def fit(cls, x_coordinates: Coordinates, y_coordinates: Coordinates) -> (Coordinates, List):
        """
        Takes a set of points in a 2-d line-graph (described using their x and y coordinates) and
        returns the y-coordinates of a resulting line graph that is constrained to be necessarily monotonically
        increasing
        :param x_coordinates: X-axis coordinates of the input points
        :param y_coordinates: Y-axis coordinates of the input points
        :return: Y-axis coordinates of corresponding points in the resulting monotonically increasing graph and the
                 polynomial coefficients
        """

        # Expect same number of x and y coordinates
        assert len(x_coordinates) == len(y_coordinates)

        # Convert x and y coordinates into float ndarrays
        x_coordinates = np.array(x_coordinates, dtype=float)
        y_coordinates = np.array(y_coordinates, dtype=float)

        # To understand the following code, you will need to follow the math in Section 2.1 of the
        # Greedy Compression-Ratio Selection HLD

        # For describing the constraints, we intentionally specify a large number of points
        # This helps ensure that the increasing monotonic constraint is enforced across the entire length of the
        # line-graph
        constraints_x_coordinates = np.arange(0, 0.98, 0.02)

        # We will use a high polynomial function to better match the original line-graph
        polynomial_degree = 8

        # Here we construct th phi and psi matrices as described in the HLD
        phi = (x_coordinates ** 0).reshape(-1, 1)
        psi = np.zeros((len(constraints_x_coordinates), 1))

        for i in range(1, polynomial_degree):
            #TODO The following lines needed to be excluded from PyLint since it crashes with the
            # following error: "RecursionError: maximum recursion depth exceeded"
            phi = np.hstack((phi, (x_coordinates ** i).reshape(-1, 1))) # pylint: disable=all
            psi = np.hstack((psi, i * (constraints_x_coordinates ** (i - 1)).reshape(-1, 1))) # pylint: disable=all
        
        # Next we calculate G = (phi.T) * phi and a = (phi.T) * Y
        # where Y is the list of y-coordinate points, and * represents a dot product

        G = np.dot(phi.T, phi)
        a = np.dot(y_coordinates, phi).reshape((-1))

        polynomial_coefficients = cls._solve_qp(G, a, psi)
        return_list = list(np.dot(phi, polynomial_coefficients))

        return return_list, polynomial_coefficients
