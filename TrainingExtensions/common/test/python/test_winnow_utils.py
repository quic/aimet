# /usr/bin/env python3.5
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
""" This file contains unit tests for testing the utilities in winnow_utils.py. """

import unittest
from aimet_common.winnow.winnow_utils import get_indices_among_ones_of_overlapping_ones, update_winnowed_channels


class TestWinnowUtils(unittest.TestCase):
    """ Test winnow_utils module """

    def test_get_indices_among_ones_of_overlapping_ones(self):
        """ Test the get_indices_among_ones_of_overlapping_ones() utility """

        more_ones_mask = [1, 1, 0, 1, 0, 0, 1]
        less_ones_mask = [1, 0, 0, 1, 0, 0, 0]
        self.assertEqual([0, 2], get_indices_among_ones_of_overlapping_ones(more_ones_mask, less_ones_mask))

        more_ones_mask = [0, 0, 0]
        less_ones_mask = [0, 0, 0]
        self.assertEqual([], get_indices_among_ones_of_overlapping_ones(more_ones_mask, less_ones_mask))

        more_ones_mask = [1, 1, 1]
        less_ones_mask = [1, 1, 1]
        self.assertEqual([0, 1, 2], get_indices_among_ones_of_overlapping_ones(more_ones_mask, less_ones_mask))

        more_ones_mask = [0, 0, 1]
        less_ones_mask = [0, 0, 1]
        self.assertEqual([0], get_indices_among_ones_of_overlapping_ones(more_ones_mask, less_ones_mask))

    def test_update_winnowed_channels(self):
        """ Test the update_winnowed_channels utility """
        original_mask = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
        new_mask = [1, 1, 0, 0, 1, 0, 1]
        update_winnowed_channels(original_mask, new_mask)
        self.assertEqual([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], original_mask)

        # Test that assertion is raised if length of new mask is different than number of ones in original mask
        with self.assertRaises(AssertionError):
            original_mask = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
            new_mask = [1, 1, 0, 0, 1, 0]
            update_winnowed_channels(original_mask, new_mask)
