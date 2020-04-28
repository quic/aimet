//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2017, Qualcomm Innovation Center, Inc. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
//
//==============================================================================

#ifndef ISVD_HPP
#define ISVD_HPP

#include <vector>

namespace DlCompression
{
enum COMPRESS_LAYER_TYPE
{
    // All other layer types. These are not compressed.
    LAYER_TYPE_OTHER,
    // Convolution Layer
    LAYER_TYPE_CONV,
    // Fully connected Layer
    LAYER_TYPE_FC
};

enum NETWORK_COST_METRIC
{
    // Optimize based on memory footprint
    COST_TYPE_MEMORY,
    // Optimize based on processing cycles
    COST_TYPE_MAC
};

enum SVD_COMPRESS_TYPE
{
    // No Compression
    TYPE_NONE,
    // Single SVD compression
    TYPE_SINGLE,
    // Successive SVD compression
    TYPE_SUCCESSIVE
};

/**
 * @brief The layer attribute container.
 * @param mode Compression type (SVD/SSVD)
 * @param layerName Layer name
 * @param layerType layer type
 * @param shape The shape of the weight matrix for layers
 *  that have weights.
 * @param blobs The collection of weights and biases of layers
 *  copied from the original network.
 * @activation_dims The dimensions of output activations from this layer.
 *  Must be set to <1, 1> for FC layers.
 *  Blobs in layerAttrib are stored with the following convention:
 *  weights --> blobs[0];  bias --> blobs[1]
 *  @param candidate_ranks Set of potential ranks to evaluate
 *  compression performance.
 * @param bestRanks Set of optimal/preferred ranks to compress layer
 * @param inputChannelMean (Optional) The average of each input channel.
 */
template <typename DTYPE>
struct LayerAttributes
{
    std::vector<int> shape;
    std::vector<std::vector<DTYPE>> blobs;
    std::pair<int, int> activation_dims;
    std::vector<std::vector<unsigned int>> candidateRanks;
    std::vector<unsigned int> bestRanks;
    SVD_COMPRESS_TYPE mode;
    COMPRESS_LAYER_TYPE layerType;
    std::vector<DTYPE> inputChannelMean;
    std::vector<float> compressionRate;
};

/**
 * @brief Interface layer for the SVD implementation class
 */
template <typename DTYPE>
class ISVD
{
public:
    /**
     * @brief Set the preferred list of ranks for compression analysis.
     * @param numCandidateRanks Number of potential ranks with which
     * to try layer compression across the network.
     * Evaluate all possible combination of rank values for compressing
     * all applicable layers in the network, and store potential ranks
     * to be used for analysimg network performance after compression.
     * @return Actual number of candidate ranks specified by SVD.
     */
    virtual int SetCandidateRanks(int numCandidateRanks) = 0;

    /**
     * @brief Get candidate ranks for a given layer and a specified rank index
     * @param layer_name Layer name
     * @param rankIndex The particular set of ranks for the layer.
     */
    virtual std::vector<unsigned int>& GetCandidateRanks(const std::string& layer_name, int rankIndex) = 0;

    /**
     * @brief Print candidate ranks of all layers for a given rank index.
     * @param rankIndex Common Index specifying a set of ranks across all layers.
     * @param useBestRanks Print only best ranks of a layer
     * The function displays the candidate ranks for compression across
     * all layers of the network which is usefu diagnostic information
     * when evaluating network performance with multiple candidates.
     * The function also displays the best ranks selected for each layer
     * which is useful information during the final compression stage.
     */
    virtual void PrintCandidateRanks(int rankIndex, bool useBestRanks) = 0;

    /**
     * @brief Determine layer type for SVD Compression (CONV/FC).
     * @param layer_type String form of the layer type.
     * The function returns the enum equivalent of the string
     * denoting the layer type.
     */
    virtual COMPRESS_LAYER_TYPE GetLayerType(const std::string& layer_type) = 0;

    /**
     * @brief Return a list of all layer names.
     */
    virtual const std::vector<std::string> GetLayerNames() const = 0;

    /**
     * @brief Determine Compression type (SVD/SSVD).
     * @param layer_type Layer type (CONV/FC).
     * @param svd_pass_type String form of compression type
     * The function returns the enum equivalent of the string
     * denoting the desired compression type (SVD/SSVD) based on
     * the layer type.
     */
    virtual SVD_COMPRESS_TYPE GetCompressionType(COMPRESS_LAYER_TYPE layer_type,
                                                 const std::string& svd_pass_type) const = 0;

    /**
     * @brief Determine Compression type (SVD/SSVD) of stored layer.
     * @param layer_name Name of stored layer.
     * This overloaded function returns the compression type of a
     * layer already stored in the layer map.
     */
    virtual SVD_COMPRESS_TYPE GetCompressionType(const std::string& layer_name) const = 0;

    /**
     * @brief Store the cost metric for evaluating degree of compression.
     * @param metric The cost type: can be MEMORY/MACs.
     */
    virtual void SetCostMetric(NETWORK_COST_METRIC metric) = 0;

    /**
     * @brief Store layer attributes.
     * @param layerName 'key' used to store in map
     * @param layerAttrib 'value' stored against the key in map
     */
    virtual void StoreLayerAttributes(const std::string& layerName, const LayerAttributes<DTYPE>& layerAttrib) = 0;

    /**
     * @brief Retrieve layer attributes.
     * @param layer_name Name of layer to look in map
     * @return LayerMap for the corresponding layer name.
     */
    virtual LayerAttributes<DTYPE>* GetLayerAttributes(const std::string& layer_name) = 0;

    /**
     * @brief Compute the total cost of the network
     * based on the cost metric.
     * computed accounting for contributions from all compressible layers.
     */
    virtual void ComputeNetworkCost() = 0;

    /**
     * @brief Get relative compression score with a given rank index
     * @param rank_index Index of the vector of candidate ranks.
     * @param useBestRanks Use LayerAttributes --> bestRanks to
     * compute compression score (used during direct compression)
     * instead of ranks indexed by rank_index (used in analysis phase).
     * The method the relative compression for each layer of the network
     * as a measure of the fraction of removed values that is achieved
     * by the specified set of ranks.
     */
    virtual DTYPE GetCompressionScore(int rank_index, bool useBestRanks, size_t networkCostMem,
                                      size_t networkCostMac) = 0;

    /**
     * @brief Split layer weight matrix into residual sub-matrices using SVD/SSVD.
     * @param layer_name Name of layer to split.
     * @param splitWeights The vector of resultant weights of split layers
     * @param weightSizes Sizes of weight vectors
     * @param rank(s) to be used for compression.
     * The method splits a matrix M[mxn] using SVD and truncates the
     * resulting [U, SVt] matrices using rank r of sizes [mxr], [rxn] respectively.
     * The method is applied successively on the [SVt] matrix in the case of
     * successive SVD.
     * ALGORITHM:
     ********************************************************************
     * SVD on Inner Product Layer
     * Split a [mxn] matrix into two sub-matrices of size
     * [mxr] and [rxn]
     * where
     * m = number of input channels
     * n = number of output channels
     * r = compression rank
     ********************************************************************
     * SVD on Convolution layer
     * Split a [mxnxkxk] matrix into two sub-matrices of size
     * [mxrx1x1] and [rxnxkxk]
     * where
     * m = number of input channels
     * n = number of output channels
     * k = kernel size (for square kernels, but applicable to
     *                  non-square kernels as well)
     * r = compression rank
     ********************************************************************
     * SSVD on Convolution layer (not applicable to other layer types)
     * Split a [mxnxkxk] matrix into three (or more) sub-matrices of size
     * [mxrx1x1], [rxsxkxk] and [sxnx1x1]
     * where
     * m = number of input channels
     * n = number of output channels
     * k = kernel size (for square kernels, but applicable to
     *                  non-square kernels as well)
     * r,s = <input, output> compression ranks
     ********************************************************************
     */
    virtual std::vector<std::vector<DTYPE>>& SplitLayerWeights(const std::string& layer_name,
                                                               std::vector<std::vector<DTYPE>>& splitWeights,
                                                               const std::vector<unsigned int>& weightSizes,
                                                               const std::vector<unsigned int>& ranks) = 0;
    virtual void SplitLayerWeights(const std::string& layer_name, std::vector<DTYPE*> splitWeights,
                                   const std::vector<unsigned int>& weightSizes,
                                   const std::vector<unsigned int>& ranks)                             = 0;

    /**
     * @brief Split layer bias vector into residual sub-vectors using SVD/SSVD.
     * @param layer_name Name of layer to split.
     * @param splitBiases The vector of resultant biases of split layers
     * @param biasSizes Sizes of bias vectors
     * @param rank(s) to be used for compression.
     * The method operates on biases of the SVD-split layers by
     * truncating the bias vector of the first layer to size 'r' (rank) and
     * zeroing out its contents and simply copies the original bias vector
     * for the second layer. It applies this procedure successively for
     * subsequent layers for SSVD compression.
     * ALGORITHM:
     ********************************************************************
     *                               SVD
     * Split a bias vector b into two sub-vectors of the form:
     * b1: A zero-vector of size of the input rank
     * b2: The copy of the original bias vector
     ********************************************************************
     *                              SSVD
     * Split a bias vector b into three sub-vectors of the form:
     * b1: A zero-vector of size of the input rank (r)
     * b2: A zero-vector of size of the output rank (s)
     * b3: The copy of the original bias vector
     ********************************************************************
     */
    virtual std::vector<std::vector<DTYPE>>& SplitLayerBiases(const std::string& layer_name,
                                                              std::vector<std::vector<DTYPE>>& splitBiases,
                                                              const std::vector<unsigned int>& biasSizes,
                                                              const std::vector<unsigned int>& ranks) = 0;
    virtual void SplitLayerBiases(const std::string& layer_name, std::vector<DTYPE*> splitBiases,
                                  const std::vector<unsigned int>& biasSizes,
                                  const std::vector<unsigned int>& ranks)                             = 0;

    /**
     * @brief Store rank(s) of all layers corresponding to an index into an internal ref table.
     * @param rankIndex Common Index specifying a set of ranks across all layers
     * contributing to a finite approximation residual.
     * This method works during network performance analysis with
     * multiple candidate ranks.
     */
    virtual void StoreBestRanks(const int rankIndex) = 0;

    /**
     * @brief Store rank(s) of a single layer corresponding to layer_name in map.
     * @param layer_name Name of layer to look in map.
     * @param bestRanks Set of ranks that are found to compress the layer best.
     * This method works best in the non-evaluation mode where best ranks
     * for a layer are pre-determined.
     */
    virtual void StoreBestRanks(const std::string& layer_name, const std::vector<unsigned int>& bestRanks) = 0;

    virtual ~ISVD() {};
};

template <typename DTYPE>
ISVD<DTYPE>* GetSVDInstance();

}   // End of namespace DlCompression

#endif   // ISVD_HPP
