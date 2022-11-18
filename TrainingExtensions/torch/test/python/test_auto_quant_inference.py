# /usr/bin/env python3.6
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
from unittest.mock import MagicMock, patch
from torch.utils.data import Dataset, DataLoader

from aimet_torch import utils
from aimet_torch.auto_quant_inference import AutoQuantInfer
import aimet_torch.model_preparer as ModelPreparer
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.examples.test_models import ModelWithFunctionalReLU
from aimet_torch.quantsim import QuantizationSimModel


@pytest.fixture(scope="session")
def dummy_input():
    return torch.randn((1, 3, 32, 32))

@pytest.fixture(scope="session")
def unlabeled_data_loader(dummy_input):
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    dataset = MyDataset([dummy_input[0, :] for _ in range(10)])
    return DataLoader(dataset)


class TestAutoQuantInference:
    @patch('torch.cuda.is_available')
    def test_dummy_input_cuda_not_available(self, mock_cuda_available, dummy_input):
        # use_cuda is True but torch.cuda.is_available is False
        model = ModelWithFunctionalReLU().eval()

        mock_cuda_available.return_value = False
        auto_quant = AutoQuantInfer(model, MagicMock(), MagicMock(), dummy_input, use_cuda=True)
        # confirms that model and dummy_input are placed on CPU
        assert not auto_quant.dummy_input.is_cuda
        assert utils.get_device(auto_quant.fp32_model) == torch.device('cpu')

    @pytest.mark.cuda
    def test_dummy_input_conversion(self, dummy_input):
        model = ModelWithFunctionalReLU().eval()

        # use_cuda is True, inputs are on CPU
        auto_quant_use_cuda = AutoQuantInfer(model.cpu(), MagicMock(), MagicMock(), dummy_input.cpu(), use_cuda=True, cuda_device_num=4)
        # confirms that model and dummy_input are placed on GPU
        assert auto_quant_use_cuda.dummy_input.is_cuda
        assert utils.get_device(auto_quant_use_cuda.fp32_model) != torch.device('cpu')

        # use_cuda is false, inputs are on CUDA
        auto_quant_no_cuda = AutoQuantInfer(model.cuda(), MagicMock(), MagicMock(), dummy_input.cuda())
        # confirms that model and dummy_input are placed on CPU
        assert not auto_quant_no_cuda.dummy_input.is_cuda
        assert utils.get_device(auto_quant_no_cuda.fp32_model) == torch.device('cpu')

    def test_none_input(self):
        pytest.raises(ValueError, AutoQuantInfer, None, MagicMock(), MagicMock(), MagicMock())
        pytest.raises(ValueError, AutoQuantInfer, MagicMock(), None, MagicMock(), MagicMock())
        pytest.raises(ValueError, AutoQuantInfer, MagicMock(), MagicMock(), None, MagicMock())
        pytest.raises(ValueError, AutoQuantInfer, MagicMock(), MagicMock(), MagicMock(), None)

    @patch.object(ModelPreparer, 'prepare_model')
    @patch.object(ModelValidator, 'validate_model')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._apply_batchnorm_folding')
    def test_model_preparer_and_validator_fail(self, mock_bn_fold, mock_model_validator, mock_model_preparer, dummy_input):
        ''' Covers case where ModelValidator and ModelPreparer both fail with ignore_errors as True'''
        model = ModelWithFunctionalReLU().eval()
        mock_model_validator.side_effect = [False, False]
        mock_model_preparer.side_effect = ValueError
        mock_bn_fold.return_value = model, fold_all_batch_norms(model, tuple(dummy_input.shape))

        autoquant = AutoQuantInfer(model, MagicMock(), MagicMock(), dummy_input, ignore_errors=True)
        autoquant.inference()
        mock_model_preparer.assert_called()
        mock_model_validator.assert_called()
        assert mock_model_validator.call_count == 2
        mock_bn_fold.assert_called()

    @patch.object(ModelPreparer, 'prepare_model')
    @patch.object(ModelValidator, 'validate_model')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._apply_batchnorm_folding')
    def test_model_preparer_fail_strict(self, mock_bn_fold, mock_model_validator, mock_model_preparer, dummy_input):
        ''' Covers case where ModelValidator fails initially and ModelPreparer fails with ignore_errors as False'''
        model = ModelWithFunctionalReLU().eval()
        mock_model_validator.return_value = False
        mock_model_preparer.side_effect = ValueError

        autoquant = AutoQuantInfer(model, MagicMock(), MagicMock(), dummy_input)
        with pytest.raises(ValueError, match="Model validation and model preparation have failed. Please make the necessary changes to the model and run again."):
            autoquant.inference()
        mock_model_preparer.assert_called()
        mock_model_validator.assert_called()
        assert mock_model_validator.call_count == 1
        mock_bn_fold.assert_not_called()

    @patch.object(ModelPreparer, 'prepare_model')
    @patch.object(ModelValidator, 'validate_model')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._apply_batchnorm_folding')
    def test_model_validator_fail_twice_strict(self, mock_bn_fold, mock_model_validator, mock_model_preparer, dummy_input):
        ''' Covers case where ModelValidator fails twice and ModelPreparer passes with ignore_errors as False'''
        model = ModelWithFunctionalReLU().eval()
        mock_model_validator.side_effect = [False, False]

        autoquant = AutoQuantInfer(model, MagicMock(), MagicMock(), dummy_input)
        with pytest.raises(ValueError, match='Model validation has failed after model preparation. Please make the necesary changes to the model and run again.'):
            autoquant.inference()
        mock_model_preparer.assert_called()
        mock_model_validator.assert_called()
        assert mock_model_validator.call_count == 2
        mock_bn_fold.assert_not_called()

    @patch.object(ModelPreparer, 'prepare_model')
    @patch.object(ModelValidator, 'validate_model')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._apply_batchnorm_folding')
    def test_model_validator_fail_twice_ignore(self, mock_bn_fold, mock_model_validator, mock_model_preparer, dummy_input):
        ''' Covers case where ModelValidator fails twice and ModelPreparer passes with ignore_errors as True'''
        model = ModelWithFunctionalReLU().eval()
        mock_model_validator.side_effect = [False, False]
        mock_bn_fold.return_value = model, fold_all_batch_norms(model, tuple(dummy_input.shape))

        autoquant = AutoQuantInfer(model, MagicMock(), MagicMock(), dummy_input, ignore_errors=True)
        autoquant.inference()
        mock_model_preparer.assert_called()
        mock_model_validator.assert_called()
        assert mock_model_validator.call_count == 2
        mock_bn_fold.assert_called()

    @patch.object(ModelPreparer, 'prepare_model')
    @patch.object(ModelValidator, 'validate_model')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._apply_batchnorm_folding')
    def test_model_validator_resolved(self, mock_bn_fold, mock_model_validator, mock_model_preparer, dummy_input):
        ''' Covers the case where Model Validator fails initially then ModelPreparer resolves these errors'''
        model = ModelWithFunctionalReLU().eval()
        mock_model_validator.side_effect = [False, True]
        mock_bn_fold.return_value = model, fold_all_batch_norms(model, tuple(dummy_input.shape))

        autoquant = AutoQuantInfer(model, MagicMock(), MagicMock(), dummy_input)
        autoquant.inference()
        mock_model_preparer.assert_called()
        mock_model_validator.assert_called()
        assert mock_model_validator.call_count == 2
        mock_bn_fold.assert_called()

    def test_quantsim_inputs(self, dummy_input):
        model = ModelWithFunctionalReLU().eval()
        autoquant = AutoQuantInfer(model, MagicMock(), MagicMock(), dummy_input)

        # Bitwidth < 4 or bitwidth > 32
        pytest.raises(ValueError, autoquant.inference, param_bw=2)
        pytest.raises(ValueError, autoquant.inference, param_bw=64)
        pytest.raises(ValueError, autoquant.inference, output_bw=2)
        pytest.raises(ValueError, autoquant.inference, output_bw=64)

        # rounding mode
        pytest.raises(ValueError, autoquant.inference, rounding_mode="other")

        # quant scheme
        pytest.raises(ValueError, autoquant.inference, quant_scheme=None)

    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._validate_inputs')
    @patch.object(ModelPreparer, 'prepare_model')
    @patch.object(ModelValidator, 'validate_model')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._apply_batchnorm_folding')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._create_quantsim_and_encodings')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._evaluate_model_performance')
    def test_bn_fold_error(self, mock_model_eval, mock_create_quantsim, mock_bn_fold,
                           mock_validate_model, mock_prepare_model, mock_validate_inputs, dummy_input):

        model = ModelWithFunctionalReLU().eval()

        mock_validate_inputs.return_value = model.cpu(), dummy_input.cpu()
        mock_bn_fold.side_effect = ValueError

        autoquant = AutoQuantInfer(model, MagicMock(), MagicMock(), dummy_input)
        mock_validate_inputs.assert_called()

        pytest.raises(ValueError, autoquant.inference)
        mock_prepare_model.assert_not_called()
        mock_validate_model.assert_called()
        mock_bn_fold.assert_called()
        mock_model_eval.assert_not_called()
        mock_create_quantsim.assert_not_called()


    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._validate_inputs')
    @patch.object(ModelPreparer, 'prepare_model')
    @patch.object(ModelValidator, 'validate_model')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._apply_batchnorm_folding')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._create_quantsim_and_encodings')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._evaluate_model_performance')
    def test_quantsim_creation_error(self, mock_model_eval, mock_create_quantsim, mock_bn_fold,
                                     mock_validate_model, mock_prepare_model, mock_validate_inputs, dummy_input):

        model = ModelWithFunctionalReLU().eval()

        mock_validate_inputs.return_value = model.cpu(), dummy_input.cpu()
        mock_bn_fold.return_value = model, fold_all_batch_norms(model, tuple(dummy_input.shape))
        mock_create_quantsim.side_effect = ValueError

        autoquant = AutoQuantInfer(model, MagicMock(), MagicMock(), dummy_input)
        mock_validate_inputs.assert_called()

        pytest.raises(ValueError, autoquant.inference)
        mock_prepare_model.assert_not_called()
        mock_validate_model.assert_called()
        mock_bn_fold.assert_called()
        mock_create_quantsim.assert_called()
        mock_model_eval.assert_not_called()

    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._validate_inputs')
    @patch.object(ModelPreparer, 'prepare_model')
    @patch.object(ModelValidator, 'validate_model')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._apply_batchnorm_folding')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._create_quantsim_and_encodings')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._evaluate_model_performance')
    def test_end_to_end_success_cpu(self, mock_model_eval, mock_create_quantsim, mock_bn_fold,
                                    mock_validate_model, mock_prepare_model, mock_validate_inputs, dummy_input):

        model = ModelWithFunctionalReLU().eval()

        mock_validate_inputs.return_value = model.cpu(), dummy_input.cpu()
        mock_bn_fold.return_value = model, fold_all_batch_norms(model, tuple(dummy_input.shape))

        autoquant = AutoQuantInfer(model, MagicMock(), MagicMock(), dummy_input)
        mock_validate_inputs.assert_called()
        mock_bn_fold.assert_not_called()
        mock_model_eval.assert_not_called()
        mock_create_quantsim.assert_not_called()

        autoquant.inference()
        mock_prepare_model.assert_not_called()
        mock_validate_model.assert_called()
        mock_bn_fold.assert_called()
        mock_model_eval.assert_called()
        mock_create_quantsim.assert_called()

    @pytest.mark.cuda
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._validate_inputs')
    @patch.object(ModelPreparer, 'prepare_model')
    @patch. object(ModelValidator, 'validate_model')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._apply_batchnorm_folding')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._create_quantsim_and_encodings')
    @patch('aimet_torch.auto_quant_inference.AutoQuantInfer._evaluate_model_performance')
    def test_end_to_end_success_gpu(self, mock_model_eval, mock_create_quantsim, mock_bn_fold,
                                    mock_validate_model, mock_prepare_model, mock_validate_inputs, dummy_input):
        model = ModelWithFunctionalReLU().eval()

        mock_validate_inputs.return_value = model.cuda(), dummy_input.cuda()
        mock_bn_fold.return_value = model.cuda(), fold_all_batch_norms(model, tuple(dummy_input.shape))

        autoquant = AutoQuantInfer(model, MagicMock(), MagicMock(), dummy_input, use_cuda=True)
        mock_validate_inputs.assert_called()
        mock_bn_fold.assert_not_called()
        mock_model_eval.assert_not_called()
        mock_create_quantsim.assert_not_called()

        autoquant.inference()
        mock_prepare_model.assert_not_called()
        mock_validate_model.assert_called()
        mock_bn_fold.assert_called()
        mock_model_eval.assert_called()
        mock_create_quantsim.assert_called()

    def test_acceptance(self, unlabeled_data_loader, dummy_input):
        model = ModelWithFunctionalReLU().eval()

        mock_eval = MagicMock()
        mock_eval.return_value = 50.0

        auto_quant = AutoQuantInfer(model, unlabeled_data_loader, mock_eval, dummy_input)
        sim_model, accuracy = auto_quant.inference()
        assert ModelValidator.validate_model(auto_quant.fp32_model, model_input=dummy_input)
        assert auto_quant._bn_folding_applied
        assert isinstance(sim_model, QuantizationSimModel)


