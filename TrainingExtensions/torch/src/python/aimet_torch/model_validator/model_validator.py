# /usr/bin/env python3.5
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

""" Utility for validating pytorch models prior to using AIMET features """

from typing import Tuple, Union, Callable
import torch

from aimet_common.utils import AimetLogger
import aimet_torch.model_validator.validation_checks as val_checks

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


class ModelValidator:
    """
    ModelValidator object for validating that AIMET features can be applied on the Pytorch model.
    """
    def __init__(self):
        self._validation_checks = [
            val_checks.validate_for_reused_modules,
            val_checks.validate_for_missing_modules
        ]

    def add_check(self, validation_check: Callable):
        """
        Add a validation check function to be used for validating the model. Validation check functions must take the
        model and model inputs as inputs, and output True if the model passes the check, and False otherwise.
        :param validation_check: Validation check function for validating the model.
        """
        self._validation_checks.append(validation_check)

    def validate_model(self, model: torch.nn.Module, model_input: Union[torch.Tensor, Tuple]) -> bool:
        """
        Validate the pytorch model by running all validation check functions and returning True if all pass, False
        otherwise.
        """
        is_valid_model = True
        for val_check in self._validation_checks:
            val_check_result = val_check(model, model_input)
            if not val_check_result:
                logger.info('Model validator %s check failed.', val_check)
            is_valid_model = is_valid_model and val_check(model, model_input)
        return is_valid_model
