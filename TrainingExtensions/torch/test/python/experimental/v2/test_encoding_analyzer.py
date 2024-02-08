# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 Qualcomm Innovation Center, Inc. All rights reserved.
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
import torch
import pytest
import numpy as np
import random
from aimet_torch.experimental.v2.quantization.encoding_analyzer import SqnrEncodingAnalyzer, PercentileEncodingAnalyzer, MinMaxEncodingAnalyzer, _HistogramObserver

@pytest.fixture(autouse=True)
def set_seed():
    random.seed(999)
    torch.random.manual_seed(999)

class TestEncodingAnalyzer():
    @pytest.fixture
    def encoding_analyzers(self):
        min_max_encoding_analyzer = MinMaxEncodingAnalyzer((1,))
        percentile_encoding_analyzer = PercentileEncodingAnalyzer((1,), 3)
        sqnr_encoding_analyzer = SqnrEncodingAnalyzer((1, ))
        encoding_analyzer_list = [min_max_encoding_analyzer, percentile_encoding_analyzer, sqnr_encoding_analyzer]
        yield encoding_analyzer_list

    def test_compute_encodings_with_negative_bitwidth(self, encoding_analyzers):
        for encoding_analyzer in encoding_analyzers:
            encoding_analyzer.update_stats(torch.randn(3, 4))
            with pytest.raises(ValueError):
                if isinstance(encoding_analyzer, PercentileEncodingAnalyzer):
                    encoding_analyzer.compute_encodings(bitwidth = 0, is_symmetric = False, percentile=99)
                else:
                    encoding_analyzer.compute_encodings(bitwidth = 0, is_symmetric = False)
    
    def test_reset_stats(self, encoding_analyzers):
        for encoding_analyzer in encoding_analyzers:
            encoding_analyzer.update_stats(torch.randn(3, 4))
            if isinstance(encoding_analyzer, MinMaxEncodingAnalyzer):
                assert encoding_analyzer.observer.stats.min
                assert encoding_analyzer.observer.stats.max
            else:
                assert all(x.min is not None for x in encoding_analyzer.observer.stats)
                assert all(x.max is not None for x in encoding_analyzer.observer.stats)
            
            encoding_analyzer.reset_stats()
            if isinstance(encoding_analyzer, MinMaxEncodingAnalyzer):
                assert not encoding_analyzer.observer.stats.min
                assert not encoding_analyzer.observer.stats.max
            else:
                assert all(x.min is None for x in encoding_analyzer.observer.stats)
                assert all(x.max is None for x in encoding_analyzer.observer.stats)
                
    def test_compute_encodings_with_no_stats(self, encoding_analyzers):
        for encoding_analyzer in encoding_analyzers:
            with pytest.raises(RuntimeError):
                if isinstance(encoding_analyzer, PercentileEncodingAnalyzer):
                    encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False, percentile = 99)
                else:
                    encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False)

    @pytest.mark.parametrize('dtype', [torch.float, torch.half])
    @pytest.mark.parametrize('symmetric', [True, False])
    def test_continuity(self, symmetric, dtype, encoding_analyzers):
        for encoding_analyzer in encoding_analyzers:
            normal_range = torch.arange(-128, 128).to(dtype) / 256
            eps = torch.finfo(dtype).eps

            if isinstance(encoding_analyzer, PercentileEncodingAnalyzer):
                percentile_val = 99
                min_1, max_1 = encoding_analyzer.compute_dynamic_encodings(normal_range * (1 - eps),
                                                                        bitwidth=8, is_symmetric=symmetric,
                                                                        percentile=percentile_val)
                min_2, max_2 = encoding_analyzer.compute_dynamic_encodings(normal_range,
                                                                        bitwidth=8, is_symmetric=symmetric,
                                                                        percentile=percentile_val)
                min_3, max_3 = encoding_analyzer.compute_dynamic_encodings(normal_range * (1 + eps),
                                                                    bitwidth=8, is_symmetric=symmetric,
                                                                    percentile=percentile_val)
            else:
                min_1, max_1 = encoding_analyzer.compute_dynamic_encodings(normal_range * (1 - eps),
                                                                        bitwidth=8, is_symmetric=symmetric)
                min_2, max_2 = encoding_analyzer.compute_dynamic_encodings(normal_range,
                                                                        bitwidth=8, is_symmetric=symmetric)
                min_3, max_3 = encoding_analyzer.compute_dynamic_encodings(normal_range * (1 + eps),
                                                                        bitwidth=8, is_symmetric=symmetric)

            assert min_3 <= min_2 <= min_1 <= max_1 <= max_2 <= max_3
            assert torch.allclose(max_1, max_2, atol=eps)
            assert torch.allclose(min_1, min_2, atol=eps)
            assert torch.allclose(max_2, max_3, atol=eps)
            assert torch.allclose(min_2, min_3, atol=eps)

class TestMinMaxEncodingAnalyzer():
    def test_compute_encodings_asymmetric(self):
        encoding_analyzer = MinMaxEncodingAnalyzer((1,))
        input_tensor =  torch.arange(start=0, end=26, step=0.5, dtype=torch.float)
        encoding_analyzer.update_stats(input_tensor)

        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False)
        assert torch.all(torch.isclose(asymmetric_min, torch.zeros(tuple(encoding_analyzer.observer.shape))))
        assert torch.all(torch.isclose(asymmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), 25.5)))

    def test_compute_encodings_signed_symmetric(self):
        encoding_analyzer = MinMaxEncodingAnalyzer((1,))
        input_tensor =  torch.arange(start=0, end=26, step=0.5, dtype=torch.float)
        encoding_analyzer.update_stats(input_tensor)

        symmetric_min, symmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = True)
        assert torch.all(torch.isclose(symmetric_min, torch.full(tuple(encoding_analyzer.observer.shape), -25.5)))
        assert torch.all(torch.isclose(symmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), 25.5)))

    @pytest.mark.parametrize("min_max_size", [[3,4], [2, 3, 1], [4], [1]])
    def test_update_stats_with_different_dimensions(self,min_max_size):
        for _ in range(4):
            encoding_analyzer = MinMaxEncodingAnalyzer(min_max_size)
            encoding_analyzer.update_stats(torch.randn(2, 3, 4))
            assert list(encoding_analyzer.observer.stats.min.shape) == min_max_size
            assert list(encoding_analyzer.observer.stats.max.shape) == min_max_size

    def test_update_stats_incompatible_dimension(self):
        encoding_analyzer = MinMaxEncodingAnalyzer([3, 4])
        with pytest.raises(RuntimeError):
            encoding_analyzer.update_stats(torch.randn(2, 3, 5))
   
    def test_compute_encodings_with_same_nonzero_tensor(self):
        encoding_analyzer = MinMaxEncodingAnalyzer((1,))
        encoding_analyzer.update_stats(torch.full((1,), 3.0))

        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False)
        assert torch.allclose(asymmetric_min, torch.full(tuple(encoding_analyzer.observer.shape), 0.0))
        assert torch.allclose(asymmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), 3.0))

        symmetric_min , symmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = True)
        assert torch.allclose(symmetric_min, torch.full(tuple(encoding_analyzer.observer.shape), -3.0))
        assert torch.allclose(symmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), 3.0))
    
    def test_compute_encodings_with_only_zero_tensor(self):
        encoding_analyzer = MinMaxEncodingAnalyzer((1,))
        encoding_analyzer.update_stats(torch.zeros(1,))

        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False)
        updated_min = torch.finfo(asymmetric_min.dtype).tiny * (2 ** (8 - 1))
        updated_max = torch.finfo(asymmetric_min.dtype).tiny * ((2 **(8 - 1)) - 1)
        assert torch.all(torch.eq(asymmetric_min,  torch.full(tuple(encoding_analyzer.observer.shape), -updated_min)))
        assert torch.all(torch.eq(asymmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), updated_max)))

        symmetric_min , symmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = True)
        updated_symmetric_min = min(-updated_min, -updated_max)
        updated_symmetric_max = max(updated_min, updated_max)
        assert torch.all(torch.eq(symmetric_min, torch.full(tuple(encoding_analyzer.observer.shape), updated_symmetric_min)))
        assert torch.all(torch.eq(symmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), updated_symmetric_max)))
    
    @pytest.mark.parametrize('symmetric', [True, False])
    def test_overflow(self, symmetric):
        encoding_analyzer = MinMaxEncodingAnalyzer((1,))
        float_input_min = (torch.arange(10) * torch.finfo(torch.float).tiny)
        encoding_analyzer.update_stats(float_input_min)
        min, max = encoding_analyzer.compute_encodings(bitwidth=8, is_symmetric=symmetric)
        scale = (max - min) / 255

        # Scale should be at least as large as torch.min
        assert scale != 0
        assert torch.allclose(scale, torch.tensor(torch.finfo(scale.dtype).tiny), atol=1e-10)

        float_input_max = (torch.arange(10) * torch.finfo(torch.float).max)
        encoding_analyzer.update_stats(float_input_max)
        min_1, max_1 = encoding_analyzer.compute_encodings(bitwidth=8, is_symmetric=symmetric)
        assert torch.all(torch.isfinite(min_1))
        assert torch.all(torch.isfinite(max_1))

class TestHistogramEncodingAnalyzer:
    @pytest.fixture
    def histogram_based_encoding_analyzers(self, request):
        min_max_shape = request.param[0]
        num_bins = request.param[1]

        percentile_encoding_analyzer = PercentileEncodingAnalyzer(min_max_shape, num_bins)
        #sqnr_encoding_analyzer = SqnrEncodingAnalyzer(min_max_shape, num_bins)
        encoding_analyzer_list = [percentile_encoding_analyzer]
        yield encoding_analyzer_list

    @pytest.mark.parametrize("histogram_based_encoding_analyzers", [((1,), 3)], indirect=True)
    def test_compute_encodings_with_same_nonzero_tensor(self, histogram_based_encoding_analyzers):
        for encoding_analyzer in histogram_based_encoding_analyzers:
            encoding_analyzer.update_stats(torch.full((1,), 3.0))

            stats = encoding_analyzer.observer.stats[0]
            cum_sum = torch.cumsum(stats.histogram, dim=0)
            index = torch.searchsorted(cum_sum, torch.quantile(cum_sum, 99/100))
            max_val = stats.bin_edges[index]

            if isinstance(encoding_analyzer, PercentileEncodingAnalyzer):
                asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False, percentile=99)
            else:
                asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False)
            
            assert torch.allclose(asymmetric_min, torch.full(tuple(encoding_analyzer.observer.shape), 0.0))
            assert torch.allclose(asymmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), max_val))

            if isinstance(encoding_analyzer, PercentileEncodingAnalyzer):
                symmetric_min , symmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = True, percentile = 99)
            else:
                symmetric_min , symmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = True)
            
            assert torch.allclose(symmetric_min, torch.full(tuple(encoding_analyzer.observer.shape), -1 * max_val))
            assert torch.allclose(symmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), max_val))
    
    @pytest.mark.parametrize("histogram_based_encoding_analyzers", [((1,), 3)], indirect=True)
    def test_compute_encodings_with_only_zero_tensor(self, histogram_based_encoding_analyzers):
        for encoding_analyzer in histogram_based_encoding_analyzers:
            encoding_analyzer.update_stats(torch.zeros(1,))

            stats = encoding_analyzer.observer.stats[0]
            cum_sum = torch.cumsum(stats.histogram, dim=0)
            index = torch.searchsorted(cum_sum, torch.quantile(cum_sum, 99/100))
            min_value = stats.bin_edges[index]

            if isinstance(encoding_analyzer, PercentileEncodingAnalyzer):
                asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False, percentile=99)
            else:
                asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False)

            assert torch.allclose(asymmetric_min,  torch.full(tuple(encoding_analyzer.observer.shape), min_value))
            assert torch.all(torch.eq(asymmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), 0)))

            if isinstance(encoding_analyzer, PercentileEncodingAnalyzer):
                symmetric_min , symmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = True, percentile = 99)
            else:
                symmetric_min , symmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = True)
            
            assert torch.allclose(symmetric_min, torch.full(tuple(encoding_analyzer.observer.shape), min_value))
            assert torch.allclose(symmetric_max, torch.full(tuple(encoding_analyzer.observer.shape), -1 * min_value))
        
    @pytest.mark.parametrize('num_bins', [-1, 0])
    def test_invalid_bin_value(self, num_bins):
        min_max_shape = (1,)
        
        with pytest.raises(ValueError):
            PercentileEncodingAnalyzer(num_bins = num_bins, shape=min_max_shape)
        
        with pytest.raises(ValueError):
            SqnrEncodingAnalyzer(min_max_shape, num_bins)

    @pytest.mark.cuda
    @pytest.mark.parametrize("histogram_based_encoding_analyzers", [((1,), 3)], indirect=True)
    @pytest.mark.parametrize('symmetric', [True, False])
    def test_cuda_inputs(self, histogram_based_encoding_analyzers, symmetric):
        for encoding_analyzer in histogram_based_encoding_analyzers:
            input_tensor = torch.tensor([2.0, 3.5, 4.2, 5.0])
            encoding_analyzer.update_stats(input_tensor.cuda())
            if isinstance(encoding_analyzer, PercentileEncodingAnalyzer):
                encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = symmetric, percentile=99)
            else:
                encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = symmetric)
            encoding_analyzer.update_stats(input_tensor.cuda())
            
            input_tensor_2 = input_tensor * 1.1 - 0.1 
            encoding_analyzer.update_stats(input_tensor_2.cuda())
            
            if isinstance(encoding_analyzer, PercentileEncodingAnalyzer):
                encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = symmetric, percentile=99)
            else:
                encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = symmetric)
            
            assert encoding_analyzer.observer.stats[0].histogram.is_cuda
            assert encoding_analyzer.observer.stats[0].bin_edges.is_cuda

    @pytest.mark.parametrize("histogram_based_encoding_analyzers", [((1,), 3)], indirect=True)
    def test_merge_stats_resize_histogram(self, histogram_based_encoding_analyzers):
        for encoding_analyzer in histogram_based_encoding_analyzers:
            input_tensor_1 = torch.tensor([2.0, 3.5, 4.2, 5.0])
            encoding_analyzer.update_stats(input_tensor_1)
            assert len(encoding_analyzer.observer.stats) == 1
            assert encoding_analyzer.observer.stats[0].min == 2
            assert encoding_analyzer.observer.stats[0].max == 5
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].histogram, torch.Tensor([1.0, 1.0, 2.0])))
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].bin_edges, torch.Tensor([2.0, 3.0, 4.0, 5.0])))
            
            # update max
            input_tensor_2 = torch.tensor([5.3, 6.4, 7.0, 8.0])
            encoding_analyzer.update_stats(input_tensor_2)
            assert len(encoding_analyzer.observer.stats) == 1
            assert encoding_analyzer.observer.stats[0].min == 2
            assert encoding_analyzer.observer.stats[0].max == 8
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].histogram, torch.Tensor([2.0, 3.0, 3.0])))
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].bin_edges, torch.Tensor([2.0, 4.0, 6.0, 8.0])))

            # update min
            input_tensor_3 = torch.tensor([-4.2, 0, 2.3, 4.5])
            encoding_analyzer.update_stats(input_tensor_3)
            assert len(encoding_analyzer.observer.stats) == 1
            assert encoding_analyzer.observer.stats[0].min == -4.2
            assert encoding_analyzer.observer.stats[0].max == 8
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].histogram, torch.Tensor([1, 4, 7])))
            assert torch.allclose(encoding_analyzer.observer.stats[0].bin_edges, torch.Tensor([-4.2000, -0.133333, 3.933333, 8.000]))
    
    @pytest.mark.parametrize("histogram_based_encoding_analyzers", [((1,), 3)], indirect=True)
    def test_merge_stats_resize_histogram_with_ambiguous_bins(self, histogram_based_encoding_analyzers):
        for encoding_analyzer in histogram_based_encoding_analyzers:
            input_tensor_1 = torch.tensor([-4.2, 2.4, 7.0, 8.0])
            encoding_analyzer.update_stats(input_tensor_1)
            assert len(encoding_analyzer.observer.stats) == 1
            assert encoding_analyzer.observer.stats[0].min == -4.2
            assert encoding_analyzer.observer.stats[0].max == 8
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].histogram, torch.Tensor([1.0, 1.0, 2.0])))
            assert torch.allclose(encoding_analyzer.observer.stats[0].bin_edges, torch.Tensor([-4.2000, -0.133333, 3.933333, 8.000]))

            input_tensor_2 = torch.tensor([-6.7, -2.5, 7.2, 10.3])
            # hist is [2, 0, 2] for this tensor only
            encoding_analyzer.update_stats(input_tensor_2) 
            assert len(encoding_analyzer.observer.stats) == 1
            assert encoding_analyzer.observer.stats[0].min == -6.7
            assert encoding_analyzer.observer.stats[0].max == 10.3
            '''
            Ambiguity lies when mapping 1st and 3rd bins ex: values in [-4.2, -0.133) could map to [-6.7, -1.033) or [-1.033, 4.633)
            '''
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].histogram, torch.Tensor([3.0, 1.0, 4.0])))
            assert torch.allclose(encoding_analyzer.observer.stats[0].bin_edges, torch.Tensor([-6.7000, -1.03333,  4.63333, 10.3000]))
    
    @pytest.mark.parametrize("histogram_based_encoding_analyzers", [((1,), 3)], indirect=True)
    def test_merge_stats_resize_histogram_with_bin_splitting(self, histogram_based_encoding_analyzers):
        for encoding_analyzer in histogram_based_encoding_analyzers:
            input_tensor_1 = torch.tensor([1, 7, 5.3, 6, 5.7, 6.8, 6.2, 2.8, 3.9])
            encoding_analyzer.update_stats(input_tensor_1)
            assert len(encoding_analyzer.observer.stats) == 1
            assert encoding_analyzer.observer.stats[0].min == 1
            assert encoding_analyzer.observer.stats[0].max == 7
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].histogram, torch.Tensor([2.0, 1.0, 6.0])))
            assert torch.allclose(encoding_analyzer.observer.stats[0].bin_edges, torch.Tensor([1, 3, 5, 7]))

            input_tensor_2 = torch.tensor([0, 9, 7.8, 2.5, 4.6, 6.2, 8.8])
            encoding_analyzer.update_stats(input_tensor_2)
            assert len(encoding_analyzer.observer.stats) == 1
            assert encoding_analyzer.observer.stats[0].min == 0
            assert encoding_analyzer.observer.stats[0].max == 9
            # 6 values from the source's histograms 3rd bucket are split in half into the destination's 2nd and 3rd bucket
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].histogram, torch.Tensor([4.0, 5.0, 7.0])))
            assert torch.allclose(encoding_analyzer.observer.stats[0].bin_edges, torch.Tensor([0, 3, 6, 9]))
    
    @pytest.mark.parametrize("histogram_based_encoding_analyzers", [((1,), 1)], indirect=True)
    def test_histogram_with_one_bin(self, histogram_based_encoding_analyzers):
        for encoding_analyzer in histogram_based_encoding_analyzers:
            input_tensor_1 = torch.tensor([1, 7, 5.3, 6, 5.7, 6.8, 6.2, 2.8, 3.9])
            encoding_analyzer.update_stats(input_tensor_1)
            assert encoding_analyzer.observer.stats[0].min == 1
            assert encoding_analyzer.observer.stats[0].max == 7
    
    @pytest.mark.parametrize("histogram_based_encoding_analyzers", [((1,), 3)], indirect=True)
    def test_merge_stats_without_resizing(self, histogram_based_encoding_analyzers):
        for encoding_analyzer in histogram_based_encoding_analyzers:
            input_tensor_1 = torch.tensor([2.0, 3.5, 4.2, 5.0])
            encoding_analyzer.update_stats(input_tensor_1)
            assert len(encoding_analyzer.observer.stats) == 1
            assert encoding_analyzer.observer.stats[0].min == 2
            assert encoding_analyzer.observer.stats[0].max == 5
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].histogram, torch.Tensor([1, 1, 2])))
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].bin_edges, torch.Tensor([2, 3, 4, 5])))
            
            # same min, max values
            input_tensor_2 = torch.tensor([2.0, 3.3, 4.8, 5])
            encoding_analyzer.update_stats(input_tensor_2)
            assert len(encoding_analyzer.observer.stats) == 1
            assert encoding_analyzer.observer.stats[0].min == 2
            assert encoding_analyzer.observer.stats[0].max == 5
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].histogram, torch.Tensor([2, 2, 4])))
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].bin_edges, torch.Tensor([2, 3, 4, 5])))

            # min and max within current range
            input_tensor_3 = torch.tensor([3.1, 3.3, 3.7, 3.9])
            encoding_analyzer.update_stats(input_tensor_3)
            assert len(encoding_analyzer.observer.stats) == 1
            assert encoding_analyzer.observer.stats[0].min == 2
            assert encoding_analyzer.observer.stats[0].max == 5
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].histogram, torch.Tensor([2, 6, 4])))
            assert torch.all(torch.eq(encoding_analyzer.observer.stats[0].bin_edges, torch.Tensor([2, 3, 4, 5])))

    
    @pytest.mark.cuda
    @pytest.mark.parametrize('symmetric', [True, False])
    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    @pytest.mark.parametrize('shape', [(4,), (3, 1), (2, 1, 1), (3, 4), (2, 3, 1), (2, 1, 4), (2, 3, 4)])
    def test_compute_encodings_multidimensional(self, symmetric, device, shape):
        x = torch.arange(24, dtype=torch.float).view(2, 3, 4).to(device)
        encoding_analyzer = PercentileEncodingAnalyzer(shape=shape)
        encoding_analyzer.update_stats(x)
        encoding_min, encoding_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric=symmetric, percentile=99)
        assert encoding_min.shape == shape
        assert encoding_max.shape == shape
        if device == 'cuda':
            assert encoding_min.is_cuda
            assert encoding_max.is_cuda
        else:
            assert not encoding_min.is_cuda
            assert not encoding_max.is_cuda

        #TODO: Add SQNR here   
    
    def test_collect_stats_multidimensional(self):
        x = torch.arange(24, dtype=torch.float).view(2, 3, 4)
        shape = (4,)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(4):
            assert torch.equal(histograms[i].min,  x[:,:,i].min())
            assert torch.equal(histograms[i].max,  x[:,:,i].max())

        shape = (3, 1)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(3):
            assert torch.equal(histograms[i].min,  x[:,i,:].min())
            assert torch.equal(histograms[i].max,  x[:,i,:].max())

        shape = (2, 1, 1)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(2):
            assert torch.equal(histograms[i].min,  x[i,:,:].min())
            assert torch.equal(histograms[i].max,  x[i,:,:].max())
    

        shape = (3, 4)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(12):
            j = i // 4
            k = i % 4
            assert torch.equal(histograms[i].min,  x[:,j,k].min())
            assert torch.equal(histograms[i].max,  x[:,j,k].max())

        shape = (2, 3, 1)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(6):
            j = i // 3
            k = i % 3
            assert torch.equal(histograms[i].min,  x[j,k,:].min())
            assert torch.equal(histograms[i].max,  x[j,k,:].max())

        shape = (2, 1, 4)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(8):
            j = i // 4
            k = i % 4
            assert torch.equal(histograms[i].min,  x[j,:,k].min())
            assert torch.equal(histograms[i].max,  x[j,:,k].max())

        shape = (2, 3, 4)
        observer = _HistogramObserver(shape, num_bins=5)
        histograms = observer.collect_stats(x)
        for i in range(24):
            j = i // 12
            k = (i // 4) % 3
            m = i % 4
            assert torch.equal(histograms[i].min,  x[j,k,m].min())
            assert torch.equal(histograms[i].max,  x[j,k,m].max()) 
    
    def test_histogram_during_merging(self):
        observer = _HistogramObserver((1,), num_bins=10)
        input = torch.arange(-50, 51, dtype=torch.float)
        old_histogram = observer.collect_stats(input)
        observer.merge_stats(old_histogram, input)

        input = torch.arange(-50, 51, dtype=torch.float) * 1.5
        new_histogram = observer.collect_stats(input)
        observer.merge_stats(new_histogram, input)

        merged_histogram = observer.stats[0]
        assert list(merged_histogram.histogram) == [10, 15, 25, 25, 25, 25, 25, 26, 15, 11]
        assert list(merged_histogram.bin_edges) == [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]

        #                                       (old_histogram)
        #
        #                   10    10    10    10    10    10    10    10    10    11
        #                 |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
        #                -50 | -40   -30   -20 | -10    0     10 |  20    30    40 |  50
        #                    |        |        |        |        |        |        |
        #                    |        |        |        |        |        |        |
        #                    |        |        |        |        |        |        |
        #              (+5)  | (+15)  | (+15)  | (+15)  | (+15)  | (+15)  | (+16)  |  (+5)
        #      10       10   |   10   |   10   |   10   |   10   |   10   |   10   |   10       11
        #  |--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
        # -75      -60      -45      -30      -15       0        15       30       45       60       75
        #
        #                                       (new_histogram)
            

class TestPercentileEncodingAnalyzer():  
    @pytest.mark.parametrize("percentile_value", [-1, 49, 5, 101])
    def test_invalid_percentile_value(self, percentile_value):
        with pytest.raises(ValueError):
            encoding_analyzer = PercentileEncodingAnalyzer((1,), 3)
            encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False, percentile=percentile_value)
    
    def test_compute_encodings_asymmetric_normalized(self):
        encoding_analyzer = PercentileEncodingAnalyzer((1,), 3)
        mean = std_dev = 2
        input_tensor = np.random.normal(mean, std_dev, size=(100000))
        encoding_analyzer.update_stats(torch.from_numpy(input_tensor))

        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False, percentile=99)
        
        # 99% of the population is within 2 1/2 standard deviations of the mean
        assert asymmetric_min > mean - std_dev * 2.5
        assert asymmetric_max < mean + std_dev * 2.5  
    
    def test_compute_encodings_asymmetric_sequential(self):
        encoding_analyzer = PercentileEncodingAnalyzer((1,), 500)
        input_tensor =  torch.arange(start=0, end=1001, step=1, dtype=torch.float)
        encoding_analyzer.update_stats(input_tensor)

        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False, percentile=99)
        
        # encoding max is the histogram bin edge which contains 99% percentile (990.02)
        assert asymmetric_min == 0
        assert asymmetric_max == 990.0
        
    def test_compute_encodings_signed_symmetric_normalized(self):
        encoding_analyzer = PercentileEncodingAnalyzer((1,), 3)
        mean = std_dev = 2
        input_tensor = np.random.normal(mean, std_dev, size=(10000))
        encoding_analyzer.update_stats(torch.from_numpy(input_tensor))

        symmetric_min, symmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = True, percentile=99)
        largest_absolute_value = max(abs(element) for element in input_tensor)
        assert symmetric_min > -largest_absolute_value
        assert symmetric_max < largest_absolute_value
    
    def test_compute_encodings_signed_symmetric_sequential(self):
        encoding_analyzer = PercentileEncodingAnalyzer((1,), 500)
        input_tensor =  torch.arange(start=0, end=1001, step=1, dtype=torch.float)
        encoding_analyzer.update_stats(input_tensor)
        
        symmetric_min, symmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = True, percentile=99)
        assert symmetric_min == -990.0
        assert symmetric_max == 990.0
    
    def test_compute_encodings_100_percentile(self):
        encoding_analyzer = PercentileEncodingAnalyzer((1,), 3)
        mean = std_dev = 2
        input_tensor = np.random.normal(mean, std_dev, size=(10000))
        encoding_analyzer.update_stats(torch.from_numpy(input_tensor))

        symmetric_min, symmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = True, percentile=100)
        largest_absolute_value = max(abs(element) for element in input_tensor)
        assert abs(symmetric_min) <= largest_absolute_value
        assert symmetric_max <= largest_absolute_value

        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = 8, is_symmetric = False, percentile=100)
        assert np.allclose(asymmetric_min.item(), min(input_tensor))
        assert np.allclose(asymmetric_max.item(), max(input_tensor))

    
    def test_compute_encodings_50_percentile(self):
        encoding_analyzer = PercentileEncodingAnalyzer((1,), 3)
        input_tensor =  torch.arange(start=0, end=1001, step=1, dtype=torch.float)
        encoding_analyzer.update_stats(input_tensor)
        bw = 8

        symmetric_min, symmetric_max = encoding_analyzer.compute_encodings(bitwidth = bw, is_symmetric = True, percentile=50)
        
        stats = encoding_analyzer.observer.stats[0]
        cum_sum = torch.cumsum(stats.histogram, dim=0)
        index = torch.searchsorted(cum_sum, torch.quantile(cum_sum, 50/100))
        mid_value = stats.bin_edges[index]
        
        assert symmetric_min == -1 * mid_value
        assert symmetric_max == mid_value

        asymmetric_min, asymmetric_max = encoding_analyzer.compute_encodings(bitwidth = bw, is_symmetric = False, percentile=50)
        assert asymmetric_min == 0
        assert asymmetric_max == mid_value


class TestSqnrEncodingAnalyzer:

    def test_computed_encodings_uniform_dist(self):
        """
        Given: Update stats on an equally spaced input (no outliers)
        When: Compute encodings
        Then: computed encodings should cover then entire input range
        """
        x = torch.arange(-100, 101) / 100
        encoding_analyzer = SqnrEncodingAnalyzer(shape=(1, ), gamma=1)
        # Expected min/max isn't exactly -1, 1 due to offset rounding
        observed_delta = torch.tensor(2 / 255.0)
        observed_offset = torch.round(-1 / observed_delta)
        observed_min = observed_offset * observed_delta
        observed_max = observed_min + 255 * observed_delta
        encoding_analyzer.update_stats(x)
        qmin, qmax = encoding_analyzer.compute_encodings(bitwidth=8, is_symmetric=False)
        assert torch.equal(qmin, observed_min.view((1, )))
        assert torch.equal(qmax, observed_max.view((1, )))

    def test_computed_encodings_with_outliers(self):
        """
        Given: Update stats on input with a severe outliers
        When: Compute encodings
        Then: Min/max range should be less than the observed min/max
        """
        # Aim for an optimal max_encoding of 3/4 the outlier value
        expected_delta = torch.Tensor([0.1])
        outlier_val = (255 * expected_delta) / 0.75
        """
        Some math required to determine the number of non-outliers to make expected_delta optimal:
            Let n = number of non outliers
            Let m = the value of the outlier
            Assume uniform distribution
            MSE_non_outlier = Var(Uniform(0, 0.5 delta)) = delta^2 / 12
            MSE_outlier = (m - 255 * delta)^2 (in unsigned symmetric)
            MSE = n * delta ^ 2 / 12 + (m - 255 * delta)^2
            d_MSE / d_delta = n * delta / 6  + 2 * 255^2 * delta - 510 * m
        Optimal delta at d_MSE / d_delta = 0, solve for n
            n = 12 * 255 * m / delta - 12 * 255 ^ 2    
        """
        # In this case, n should evaluate to 210,100
        n = round(12 * 255 * outlier_val.item() / expected_delta.item() - 12 * (255 ** 2))
        encoding_analyzer = SqnrEncodingAnalyzer((1, ), gamma=1.0)
        x = torch.rand((1, n)) * 255 * expected_delta
        encoding_analyzer.update_stats(x)
        outlier = torch.Tensor([outlier_val]).view(1, 1)
        encoding_analyzer.update_stats(outlier)
        qmin, qmax = encoding_analyzer.compute_encodings(8, is_symmetric=False)
        expected_min = torch.Tensor([0])
        expected_max = expected_delta * 255
        assert torch.allclose(qmin, expected_min)
        assert torch.allclose(qmax, expected_max)

    def test_compute_encodings_shape(self):
        """
        Given: Encoding analyzer with an arbitrary shape
        When: Compute encodings
        Then: Encodings have shape == encoding_analyzer.shape
        """
        x = torch.randn(5, 3, 5, 100)
        shape = (5, 1, 5, 1)

        encoding_analyzer = SqnrEncodingAnalyzer(shape=shape)
        histograms = encoding_analyzer.observer.collect_stats(x)
        best_min, best_max = encoding_analyzer.compute_encodings_from_stats(histograms, 8, False)
        assert best_min.shape == shape
        assert best_max.shape == shape

    def test_clamp_delta_offset_candidates(self):
        """
        Given: A set of delta offset candidates
        When: Some combinations of delta/offset extend beyond the observed min/max ranged
        Then: Clamp the delta values such that the min/max encodings fall within the observed min/max range
        """
        encoding_analyzer = SqnrEncodingAnalyzer(shape=(1,))
        num_steps = 255
        deltas = torch.Tensor([[1 / 4, 3 / 4, 5 / 4]])
        offsets = torch.Tensor([-255, -128, 0])[None, :]
        should_clamp = torch.Tensor([
            [False, False, False],
            [True,  False, True],
            [True,  True,  True],
        ]).to(torch.bool)
        deltas_clamp, offsets_clamp = encoding_analyzer._clamp_delta_offset_values(torch.Tensor([-128]),
                                                                                   torch.Tensor([127]),
                                                                                   num_steps,
                                                                                   deltas,
                                                                                   offsets)
        assert torch.all(torch.where(should_clamp,
                                     deltas_clamp < deltas[:, :, None],
                                     deltas_clamp == deltas[:, :, None]))
        min_after_clamp = deltas_clamp * offsets_clamp
        max_after_clamp = min_after_clamp + deltas_clamp * num_steps
        assert torch.all(min_after_clamp > -128.5)
        assert torch.all(max_after_clamp < 127.5)


    def test_pick_test_candidates_asymmetric(self):
        """
        Given: An initialized encoding analyzer and min/max observations
        When: Encoder selects the search space for asymmetric delta/offset combinations
        """
        min_obs = torch.Tensor([-128])
        max_obs = torch.Tensor([127])
        observed_offset = -128
        observed_scale = 1.0
        num_steps = 255
        encoding_analyzer = SqnrEncodingAnalyzer(shape=(1,), asymmetric_delta_candidates=5, offset_candidates=7)
        deltas, offsets = encoding_analyzer._pick_test_candidates_asymmetric(min_obs, max_obs, num_steps)
        """
        Then: 1) The number of candidates should be equal to num_encodings * num_delta_candidates * num_offset_candidates
        """
        assert deltas.shape == (1, 5, 7)
        assert offsets.shape == (1, 5, 7)
        """
        Then: 2) Unclamped delta values should be equally spaced in the range max_delta * [ 1/(n-1)... n/(n-1) ]
              3) Offset candidates should contain evenly spaced steps between -num_steps and 0
              4) Offset candidates should contain the observed offset
        """
        # Steps of 1 / (5 - 1) = 1 / 4
        for value in [1/4, 1/2, 3/4, 1]:
            assert value in deltas
        # Steps of 255/5 = 51
        for value in [-255, -204, -153, -102, -51, 0]:
            assert value in offsets
        assert observed_offset in offsets
        assert observed_scale in deltas
        """
        Then: 5) None of the candidates should represent values outside of [observed_min, observed_max]
        """
        min_val = torch.round(offsets * deltas)
        max_val = min_val + deltas * num_steps
        # Allow some room for offset rounding
        assert torch.all(min_val > -128.5)
        assert torch.all(max_val < 127.5)

    def test_pick_test_candidates_symmetric(self):
        """
        Given: An initialized encoding analyzer and min/max observations
        When: Encoder selects the search space for symmetric delta/offset combinations
        """
        min_obs = torch.Tensor([-100])
        max_obs = torch.Tensor([127.5])
        num_steps = 255
        encoding_analyzer = SqnrEncodingAnalyzer(shape=(1,), symmetric_delta_candidates=5)
        deltas, offsets = encoding_analyzer._pick_test_candidates_symmetric(min_obs, max_obs, num_steps)
        """
        Then: 1) The number of candidates should be equal to num_encodings * num_delta_candidates * num_offset_candidates
        """
        assert deltas.shape == (1, 5, 1)
        """
        Then: 2) Delta values should be equally spaced in the range max_delta * [ 1/(n-1)... n/(n-1) ]
              3) Only offset candidate should be -128
        """
        # Steps of 1 / (5 - 1) = 1 / 4
        for value in [1/4, 1/2, 3/4, 1]:
            assert value in deltas
        assert torch.all(offsets == -128)
        assert offsets.numel() == 1


    def test_estimate_quantization_noise(self):
        """
        Given: 1) A histogram with sufficient granularity
               2) A set of candidate delta/offset encodings
        When: Estimate the total quantization noise for each delta/offset candidate
        Then: The estimated quantization noise should be very close to the actual quantization noise
        """
        # Create random input
        input = torch.randn(2, 1000)
        # Add some outliers
        input[0][0] = 100; input[0][1] = -50
        encoding_analyzer = SqnrEncodingAnalyzer(shape=(2, 1))
        # Get Histogram inputs
        histograms = encoding_analyzer.observer.collect_stats(input)
        # Create a wide range of delta/offsets to search
        deltas = torch.Tensor([0.001, 0.1, 0.5, 1.0])[None, :, None].expand(2, 4, 6)
        offsets = torch.Tensor([-255, -204, -153, -102, -51, 0]).expand_as(deltas)
        # Compute the actual MSE for each encoding candidate:
        delta_bc = deltas[:, :, :, None]
        offset_bc = offsets[:, :, :, None]
        input_qdq = input[:, None, None, :].div(delta_bc).sub(offset_bc).round().clamp(0, 255).add(offset_bc).mul(delta_bc)
        q_error_exp = torch.pow(input[:, None, None, :] - input_qdq, 2).sum(dim=-1)
        # Estimate the MSE from the observer histogram
        q_error_est = encoding_analyzer._estimate_clip_and_quant_noise(histograms, deltas, offsets, 255, gamma=1.0)
        # Estimated and measured errors should be very close
        assert torch.allclose(q_error_exp, q_error_est, rtol=0.01)

    def test_select_best_delta_offset(self):
        """
        Given: A set of candidate delta offsets and observed histograms
        When: Call encoding_analyzer._select_best_candidates
        Then: Return the (delta, offset) pair which results in the lowest SQNR of the candidates
        """
        # Channels are in range ([-0.5, 0.5], [0, 2])
        input = (torch.rand(2, 1000) - torch.Tensor([[0.5], [0.0]])) * torch.Tensor([[1.], [2.]])
        encoding_analyzer = SqnrEncodingAnalyzer(shape=(2, 1), gamma=1.0)
        # Get Histogram inputs
        histograms = encoding_analyzer.observer.collect_stats(input)
        # best delta offset: delta=[1 / 255.0, 2 / 255.0], offset=[-128, 0]
        deltas = torch.Tensor([1/510.0, 1/255.0, 2/255.0])[None, :, None].expand(2, 3, 3)
        offsets = torch.Tensor([-200, -128, 0]).expand_as(deltas)
        # Find the best delta/offsets based on the candidates
        best_delta, best_offset = encoding_analyzer._select_best_candidates(deltas, offsets, histograms, 255)
        assert torch.equal(best_delta, torch.Tensor([1/255.0, 2/255.0]).view(2, 1))
        assert torch.equal(best_offset, torch.Tensor([-128, 0]).view(2, 1))
