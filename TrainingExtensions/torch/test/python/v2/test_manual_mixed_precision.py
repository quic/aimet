# /usr/bin/env python
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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

import pytest
import torch

from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.manual_mixed_precision import MixedPrecisionConfigurator
from .models_.test_models import SingleResidual

class TestManualMixedPrecisionConfigurator:

    def test_mp_1(self):
        """MMP Workflow """

        model = SingleResidual()
        input_shape = (1, 3, 32, 32)

        torch.manual_seed(0)
        input_tensor = torch.randn(*input_shape)

        # 1. Create QuantSim object
        sim = QuantizationSimModel(model, input_tensor)

        # 2. Create the MixedPrecisionConfigurator object by passing in the QuantSim object
        mp_configurator = MixedPrecisionConfigurator(sim)

        # 3. Make set_precision/set_model_input_precision/set_model_output_precision calls
        mp_configurator.set_precision(sim.model.conv1, 'Int16', 'Int16')

        # 4. Call apply() method by passing in the config file and strict flag
        mp_configurator.apply()
        assert mp_configurator

        # 5. compute encodings and export

    def test_mp_2(self):
        model = SingleResidual()
        input_shape = (1, 3, 32, 32)

        torch.manual_seed(0)
        input_tensor = torch.randn(*input_shape)

        sim = QuantizationSimModel(model, input_tensor)
        mp_configurator = MixedPrecisionConfigurator(sim)
        mp_configurator.set_precision(sim.model.conv1, 'Int16', 'Int16')
        with pytest.raises(ValueError):
            mp_configurator.set_precision(sim.model.maxpool, act_candidate='Int2')
