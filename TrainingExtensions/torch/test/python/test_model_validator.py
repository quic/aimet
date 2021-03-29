# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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

import unittest
import torch

from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.model_validator import validation_checks
from aimet_torch.examples import test_models


class TestValidateModel(unittest.TestCase):
    """ Class for testing model validator """

    def test_model_validator(self):
        """ Check that model validator returns correct value """

        model = test_models.SequentialModel()
        rand_inp = torch.randn(1, 3, 8, 8)
        self.assertTrue(ModelValidator.validate_model(model, rand_inp))

        model = test_models.ModelWithReusedNodes()
        rand_inp = torch.randn(1, 3, 32, 32)
        self.assertFalse(ModelValidator.validate_model(model, rand_inp))


class TestValidationChecks(unittest.TestCase):
    """ Class for testing validation check functions """

    def test_validate_for_reused_modules(self):
        """ Validate the check for reused modules """

        model = test_models.ModelWithReusedNodes()
        rand_inp = torch.randn(1, 3, 32, 32)
        self.assertFalse(validation_checks.validate_for_reused_modules(model, rand_inp))

    def test_validate_for_missing_modules(self):
        """ Validate the check for ops with missing modules """

        model = test_models.ModelWithFunctionalOps()
        rand_inp = torch.randn(1, 3, 32, 32)
        self.assertFalse(validation_checks.validate_for_missing_modules(model, rand_inp))
