//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2016-2017, Qualcomm Innovation Center, Inc. All rights reserved.
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

#include <cmath>
#include <iomanip>
#include <memory>
#include <vector>

#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>

#endif   // USE_OPENCV

#include "DlCompression/ISVD.hpp"
#include "gtest/gtest.h"

using namespace DlCompression;

enum INIT_METHOD
{
    INIT_RANDOM,     // randomly initialize data
    INIT_IDENTITY,   // initialize to an identity matrix
    INIT_FIXED       // initialize with fixed content
};

template <typename TypeParam>
class DlCompressionSVDTest : public ::testing::Test
{
protected:
    void CreateLayer(COMPRESS_LAYER_TYPE layerType, SVD_COMPRESS_TYPE mode, INIT_METHOD init_method,
                     std::vector<int>& shape, bool use_bias)
    {
        m_layerAttrib.layerType = layerType;
        m_layerAttrib.mode      = mode;
        m_layerAttrib.shape     = shape;
        std::vector<TypeParam> weights;
        std::vector<TypeParam> bias;
        std::vector<TypeParam> average_input;
        int M = 0, N = 0, k_h = 1, k_w = 1;
        N                                   = m_layerAttrib.shape[0];
        M                                   = m_layerAttrib.shape[1];
        m_layerAttrib.activation_dims.first = m_layerAttrib.activation_dims.second = 1;

        if (LAYER_TYPE_CONV == layerType)
        {
            ASSERT_EQ(m_layerAttrib.shape.size(), 4);
            k_h                                  = m_layerAttrib.shape[2];
            k_w                                  = m_layerAttrib.shape[3];
            m_layerAttrib.activation_dims.first  = m_layerAttrib.shape[2];   // top height
            m_layerAttrib.activation_dims.second = m_layerAttrib.shape[3];   // top width
        }

#ifdef USE_OPENCV
        int sz = N * M * k_h * k_w;   // Layer size
        cv::Mat A;
        // Fill layer weights
        switch (init_method)
        {
        case INIT_IDENTITY:
            ASSERT_EQ(M, N);
            A = cv::Mat::eye(M, M, CV_32F);
            break;
        case INIT_FIXED:
            A = cv::Mat::ones(1, sz, CV_32F) * 0.64;
            break;
        case INIT_RANDOM:
        default:
            A = cv::Mat(1, sz, CV_32F);
            cv::randu(A, cv::Scalar(-1), cv::Scalar(1));
            break;
        }
        weights.assign((TypeParam*) A.datastart, (TypeParam*) A.dataend);

        if (use_bias)
        {
            // Fill layer bias.
            // Always use random bias.
            cv::Mat B(N, 1, CV_32F);
            cv::randu(B, cv::Scalar(-1), cv::Scalar(1));
            bias.assign((TypeParam*) B.datastart, (TypeParam*) B.dataend);

            // Fill average inputs.
            cv::Mat C(M, 1, CV_32F);
            cv::randu(C, cv::Scalar(-1), cv::Scalar(1));
            average_input.assign((TypeParam*) C.datastart, (TypeParam*) C.dataend);
        }
#else
        // Silence compiler warnings
        (void) init_method;
#endif
        m_layerAttrib.blobs.push_back(weights);
        if (use_bias)
        {
            m_layerAttrib.blobs.push_back(bias);
            m_layerAttrib.inputChannelMean = average_input;
        }
    }

#ifdef USE_OPENCV

    void PrintMatrixElements(const std::string& matName, cv::Mat& Mat, int numRows, int numCols)
    {
        std::cout << std::endl << std::endl << std::endl;
        std::cout << "Printing " << numRows << " rows and " << numCols << " columns of " << matName << std::endl;
        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                std::cout << std::setprecision(6) << std::setw(8) << std::fixed << Mat.at<TypeParam>(i, j) << "\t";
            }
            std::cout << std::endl;
        }
    }

    // Check whether a matrix is identity matrix [I]
    bool IsIdentityMatrix(cv::Mat& Mat, int rows, int cols)
    {
        EXPECT_EQ(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if ((i == j) && Mat.at<TypeParam>(i, j) != 1.0)
                {
                    std::cout << "Returning false for Mat(i, j) = (" << i << ", " << j
                              << ") = " << Mat.at<TypeParam>(i, j) << std::endl;
                    return false;
                }

                else if ((i != j) && Mat.at<TypeParam>(i, j) != 0)
                {
                    std::cout << "Returning false for Mat(i, j) = (" << i << ", " << j
                              << ") = " << Mat.at<TypeParam>(i, j) << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

#endif

    // Container for layer attributes
    LayerAttributes<TypeParam> m_layerAttrib;
};

TYPED_TEST_CASE(DlCompressionSVDTest, ::testing::Types<float>);

/* Sanity Test: Test SVD compression with invalid rank
 *  The rank r of a matrix M (m,n) is bound by
 *  r <= min (m, n).
 *  The compression library should not allow compression
 *  when a rank outside this range is supplied.
 */
TYPED_TEST(DlCompressionSVDTest, SANITY_TestInvalidRankInput)
{
#ifdef USE_OPENCV
    std::string layerName = "ip1";
    std::vector<int> shape;
    int M = 20;   // rows of FC layer
    int N = 40;   // cols of FC layer
    shape.push_back(M);
    shape.push_back(N);

    // Create rank vector
    std::vector<unsigned int> ranks;
    unsigned int r = 60;
    ranks.push_back(r);
    // Create weights vector
    std::vector<float*> splitWeights;
    std::vector<unsigned int> weightSizes;

    this->CreateLayer(LAYER_TYPE_FC, TYPE_SINGLE, INIT_RANDOM, shape, false);
    // Create an instance of the SVD object
    ISVD<float>* svdObj = GetSVDInstance<float>();
    svdObj->StoreLayerAttributes(layerName, this->m_layerAttrib);

    unsigned int weight1_size = M * r;
    std::unique_ptr<float> weights1(new float[weight1_size]);
    splitWeights.push_back(weights1.get());
    weightSizes.push_back(weight1_size);

    unsigned int weight2_size = r * N;
    std::unique_ptr<float> weights2(new float[weight2_size]);
    splitWeights.push_back(weights2.get());
    weightSizes.push_back(weight2_size);

    // Compress layer matrix with invalid rank
    EXPECT_THROW({ svdObj->SplitLayerWeights(layerName, splitWeights, weightSizes, ranks); }, std::runtime_error);
#endif
}

/* Sanity Test: Test Identity Matrix SVD
 *  Test if SVD compression of an identity matrix [I]
 *  results in sub-matrices with of the form [I|0]
 *  where the non-zero portions of the matrices is
 *  also an identity matrix.
 */
TYPED_TEST(DlCompressionSVDTest, SANITY_TestIdentityMatrixSVD)
{
#ifdef USE_OPENCV
    std::string layerName = "ip1";
    std::vector<int> shape;
    int M = 40;   // rows of FC layer
    int N = 40;   // cols of FC layer
    shape.push_back(M);
    shape.push_back(N);

    // Create rank vector
    std::vector<unsigned int> ranks;
    unsigned int r = 30;
    ranks.push_back(r);
    // Create weights vector
    std::vector<float*> splitWeights;
    std::vector<unsigned int> weightSizes;

    this->CreateLayer(LAYER_TYPE_FC, TYPE_SINGLE, INIT_IDENTITY, shape, false);
    // Create an instance of the SVD object
    ISVD<float>* svdObj = GetSVDInstance<float>();
    svdObj->StoreLayerAttributes(layerName, this->m_layerAttrib);

    unsigned int weight1_size = M * r;
    std::unique_ptr<float> weights1(new float[weight1_size]);
    splitWeights.push_back(weights1.get());
    weightSizes.push_back(weight1_size);

    unsigned int weight2_size = r * N;
    std::unique_ptr<float> weights2(new float[weight2_size]);
    splitWeights.push_back(weights2.get());
    weightSizes.push_back(weight2_size);

    // Compress layer matrix
    svdObj->SplitLayerWeights(layerName, splitWeights, weightSizes, ranks);
    // Split weights are in transposed form.
    // Store them in proper form in intermediate matrices.
    cv::Mat WeightMat_0(r, M, CV_32F, (float*) (splitWeights[0]));
    cv::Mat WeightMat_1(N, r, CV_32F, (float*) (splitWeights[1]));
    // Expect that these matrices are of the form
    // [I|0] with rank 'r'.
    WeightMat_0 = WeightMat_0(cv::Range(0, r), cv::Range(0, r));
    WeightMat_1 = WeightMat_1(cv::Range(0, r), cv::Range(0, r));
    ASSERT_TRUE(this->IsIdentityMatrix(WeightMat_0, r, r));
    ASSERT_TRUE(this->IsIdentityMatrix(WeightMat_1, r, r));
#endif
}

/* Sanity Tests: Test Matrix reconstruction
 *  Test if a matrix reconstructed from its components obtained
 *  from an SVD operation with low-rank compression is "close"
 *  to the original in the Frobenius norm sense.
 *  The proximity of the reconstructed matrix to the original depends
 *  on the relation between the rank of the original matrix and that used
 *  to compress it. A rank value that is closer to the original rank leads
 *  to less aggressive compression and remains closer to the original,
 *  whereas an arbitrarily chosen compression rank that is much lower
 *  than the original rank leads to large divergence in the resulting matrix.
 */

/* Sanity Test: Test Exact reconstruction
 *  Create a very-low rank matrix (such as a uniform matrix
 *  with a repeated entries that is rank-1) and compress
 *  using an arbitrarily low rank (can be as low as the rank
 *  of the original matrix).
 *  The test expects the matrix resulting from reconstruction
 *  of the SVD components is an almost exact match to the
 *  original.
 */
TYPED_TEST(DlCompressionSVDTest, SANITY_TestExactReconstruction)
{
#ifdef USE_OPENCV
    float errorThreshold  = 1e-4;
    std::string layerName = "ip1";
    std::vector<int> shape;
    int M = 40;   // rows of FC layer
    int N = 30;   // cols of FC layer
    // Store dims in 'Caffe' style
    shape.push_back(N);
    shape.push_back(M);

    // Create rank vector
    std::vector<unsigned int> ranks;
    // selected rank can match the true rank of the
    // original matrix to achieve loss-less compression.
    unsigned int r = 10;
    ranks.push_back(r);
    // Create weights vector
    std::vector<float*> splitWeights;
    std::vector<unsigned int> weightSizes;

    this->CreateLayer(LAYER_TYPE_FC, TYPE_SINGLE, INIT_FIXED, shape, false);
    cv::Mat original_Matrix(N, M, CV_32F, this->m_layerAttrib.blobs[0].data());
    // Create an instance of the SVD object
    ISVD<float>* svdObj = GetSVDInstance<float>();
    svdObj->StoreLayerAttributes(layerName, this->m_layerAttrib);

    unsigned int weight1_size = M * r;
    std::unique_ptr<float> weights1(new float[weight1_size]);
    splitWeights.push_back(weights1.get());
    weightSizes.push_back(weight1_size);

    unsigned int weight2_size = r * N;
    std::unique_ptr<float> weights2(new float[weight2_size]);
    splitWeights.push_back(weights2.get());
    weightSizes.push_back(weight2_size);

    // Compress layer matrix
    svdObj->SplitLayerWeights(layerName, splitWeights, weightSizes, ranks);
    // Split weights are in transposed form.
    // Store them in proper form in intermediate matrices.
    cv::Mat WeightMat_0(r, M, CV_32F, (float*) (splitWeights[0]));
    cv::Mat WeightMat_1(N, r, CV_32F, (float*) (splitWeights[1]));

    // Reconstruct the original matrix by representing matrices
    // in their true mathematical form by nullifying the transpose
    // applied by the SVD algorithm.
    cv::Mat Reconstructed_Matrix = WeightMat_0.t() * WeightMat_1.t();
    // this->PrintMatrixElements("Reconstructed_Matrix", Reconstructed_Matrix,
    //    Reconstructed_Matrix.rows, Reconstructed_Matrix.cols);
    // Estimate the tensor approximation residual (TAR)
    float recon_error = cv::norm(Reconstructed_Matrix.t(), original_Matrix, (cv::NORM_RELATIVE | cv::NORM_L2));
    ASSERT_LE(recon_error, errorThreshold);
#endif
}

/* Sanity Test: Test Lossy reconstruction
 *  Create a high-rank matrix by initializing with random values
 *  and compress using a low rank value to result in lossy
 *  compression.
 *  The test expects that compression with
 *  higher rank values results in lower reconstruction errors.
 */
TYPED_TEST(DlCompressionSVDTest, SANITY_TestLossyReconstruction)
{
#ifdef USE_OPENCV
    float errorThreshold  = 5e-1;
    std::string layerName = "ip1";
    std::vector<int> shape;
    int M = 160;   // rows of FC layer
    int N = 120;   // cols of FC layer
    // Store dims in 'Caffe' style
    shape.push_back(N);
    shape.push_back(M);

    // Create rank vector
    std::vector<unsigned int> ranks;
    // selected a fairly small rank to perform compression
    unsigned int r = 60;
    ranks.push_back(r);
    // Create weights vector
    std::vector<float*> splitWeights;
    std::vector<unsigned int> weightSizes;

    this->CreateLayer(LAYER_TYPE_FC, TYPE_SINGLE, INIT_RANDOM, shape, false);
    cv::Mat original_Matrix(N, M, CV_32F, this->m_layerAttrib.blobs[0].data());
    // Create an instance of the SVD object
    ISVD<float>* svdObj = GetSVDInstance<float>();
    svdObj->StoreLayerAttributes(layerName, this->m_layerAttrib);

    unsigned int weight1_size = M * r;
    std::unique_ptr<float> weights1(new float[weight1_size]);
    splitWeights.push_back(weights1.get());
    weightSizes.push_back(weight1_size);

    unsigned int weight2_size = r * N;
    std::unique_ptr<float> weights2(new float[weight2_size]);
    splitWeights.push_back(weights2.get());
    weightSizes.push_back(weight2_size);

    // Compress layer matrix
    svdObj->SplitLayerWeights(layerName, splitWeights, weightSizes, ranks);
    // Split weights are in transposed form.
    // Store them in proper form in intermediate matrices.
    cv::Mat WeightMat_0(r, M, CV_32F, (float*) (splitWeights[0]));
    cv::Mat WeightMat_1(N, r, CV_32F, (float*) (splitWeights[1]));

    // Reconstruct the original matrix by representing matrices
    // in their true mathematical form by nullifying the transpose
    // applied by the SVD algorithm.
    cv::Mat Reconstructed_Matrix = WeightMat_0.t() * WeightMat_1.t();
    // Estimate the tensor approximation residual (TAR)
    float recon_error = cv::norm(Reconstructed_Matrix.t(), original_Matrix, (cv::NORM_RELATIVE | cv::NORM_L2));
    ASSERT_LE(recon_error, errorThreshold);
#endif
}

/* Sanity Test: Test single pass SVD with FC layer
 *  Create an FC layer with standard layer parameters
 *  and examine rank selection with different criteria.
 *  The test expects that the compression scores produced
 *  by the library for a given rank value match the theoretical
 *  expected amount of compression.
 */
TYPED_TEST(DlCompressionSVDTest, PERFORMANCE_TestRankSelectionFCSingle)
{
#ifdef USE_OPENCV
    std::string layerName = "ip1";
    std::vector<int> shape;
    int M                = 1024;   // rows of FC layer
    int N                = 10;     // cols of FC layer
    float errorThreshold = 5e-1;
    // Store dims in 'Caffe' style
    shape.push_back(N);
    shape.push_back(M);
    this->CreateLayer(LAYER_TYPE_FC, TYPE_SINGLE, INIT_RANDOM, shape, false);

    // Perform SVD rank analysis with different metrics.
    std::vector<NETWORK_COST_METRIC> metric_types = {COST_TYPE_MEMORY, COST_TYPE_MAC};

    for (auto metric: metric_types)
    {
        std::shared_ptr<ISVD<float> > svdObj = std::shared_ptr<ISVD<float> >(GetSVDInstance<float>());
        svdObj->SetCostMetric(metric);
        svdObj->StoreLayerAttributes(layerName, this->m_layerAttrib);
        // Compute network cost
        svdObj->ComputeNetworkCost();

        size_t layerSize              = M * N;
        int optimalRankIndex          = 12;   // Pick a rank index as optimal
        float optimalCompressionScore = 0.0;

        // Set a pre-determined number of candidate ranks
        int numCandidateRanks = svdObj->SetCandidateRanks(20);
        for (int rankIndex = 0; rankIndex < numCandidateRanks; ++rankIndex)
        {
            //  svdObj->PrintCandidateRanks (rankIndex, false);
            std::vector<unsigned int> rank = svdObj->GetCandidateRanks(layerName, rankIndex);
            // Get relative compression score at each rank.
            float compressionScore = svdObj->GetCompressionScore(rankIndex, false, 0, 0);
            // Store the 'optimal' compression score
            if (rankIndex == optimalRankIndex)
                optimalCompressionScore = compressionScore;

            // Compare theoretical compression score.
            unsigned int r                 = rank.at(0);
            size_t reducedLayerSize        = (M + N) * r;
            float expectedCompressionScore = float(layerSize - reducedLayerSize) / float(layerSize);
            ASSERT_LE(fabs(compressionScore - expectedCompressionScore), errorThreshold);
        }
        // Now store optimal ranks for the network
        svdObj->StoreBestRanks(optimalRankIndex);
        // Get compression score with bestRanks
        float bestCompressionScore = svdObj->GetCompressionScore(0, true, 0, 0);
        // Compare with optimal compression score
        ASSERT_LE(fabs(bestCompressionScore - optimalCompressionScore), errorThreshold);
    }
#endif
}

/* Sanity Test: Test single pass SVD with CONV layer
 *  Create a CONV layer with standard layer parameters
 *  and examine rank selection with different criteria with
 *  single pass SVD.
 *  CONV layers, unlike FC layers are 4D tensors and require
 *  careful handling and transposition methods to achieve
 *  proper SVD compression.
 *  The test expects that the compression scores produced
 *  by the library for a given rank value match the theoretical
 *  expected amount of compression.
 */
TYPED_TEST(DlCompressionSVDTest, PERFORMANCE_TestRankSelectionConvSingle)
{
#ifdef USE_OPENCV
    std::string layerName = "conv1";
    std::vector<int> shape;
    float errorThreshold = 5e-1;
    int M                = 32;   // rows of Conv layer
    int N                = 64;   // cols of Conv layer
    int k_h              = 5;    // kernel height
    int k_w              = 5;    // kernel width
    int Nkk              = N * k_h * k_w;
    // Store dims in 'Caffe' style
    shape.push_back(N);
    shape.push_back(M);
    shape.push_back(k_h);
    shape.push_back(k_w);
    this->CreateLayer(LAYER_TYPE_CONV, TYPE_SINGLE, INIT_RANDOM, shape, false);

    // Perform SVD rank analysis with different metrics.
    std::vector<NETWORK_COST_METRIC> metric_types = {COST_TYPE_MEMORY, COST_TYPE_MAC};

    for (auto metric: metric_types)
    {
        std::shared_ptr<ISVD<float> > svdObj = std::shared_ptr<ISVD<float> >(GetSVDInstance<float>());
        svdObj->SetCostMetric(metric);
        svdObj->StoreLayerAttributes(layerName, this->m_layerAttrib);
        // Compute network cost
        svdObj->ComputeNetworkCost();

        size_t layerSize              = M * Nkk;
        int optimalRankIndex          = 12;   // Pick a rank index as optimal
        float optimalCompressionScore = 0.0;

        int numCandidateRanks = svdObj->SetCandidateRanks(20);
        for (int rankIndex = 0; rankIndex < numCandidateRanks; ++rankIndex)
        {
            //  svdObj->PrintCandidateRanks (rankIndex, false);
            std::vector<unsigned int> ranks = svdObj->GetCandidateRanks(layerName, rankIndex);
            // Get relative compression score at each rank.
            float compressionScore = svdObj->GetCompressionScore(rankIndex, false, 0, 0);
            // Store the 'optimal' compression score
            if (rankIndex == optimalRankIndex)
                optimalCompressionScore = compressionScore;

            // Compare theoretical compression score.
            unsigned int r                 = ranks.at(0);
            size_t reducedLayerSize        = (M + Nkk) * r;
            float expectedCompressionScore = float(layerSize - reducedLayerSize) / float(layerSize);
            ASSERT_LE(fabs(compressionScore - expectedCompressionScore), errorThreshold);
        }
        // Now store optimal ranks for the network
        svdObj->StoreBestRanks(optimalRankIndex);
        // Get compression score with bestRanks
        float bestCompressionScore = svdObj->GetCompressionScore(0, true, 0, 0);
        // Compare with optimal compression score
        ASSERT_LE(fabs(bestCompressionScore - optimalCompressionScore), errorThreshold);
    }
#endif
}

/* Sanity Test: Test successive pass SVD with CONV layer
 *  Create a CONV layer with standard layer parameters
 *  and examine rank selection with different criteria with
 *  successive pass SVD.
 *  CONV layers, unlike FC layers are 4D tensors and require
 *  careful handling and transposition methods to achieve
 *  proper SSVD compression.
 *  The test expects that the compression scores produced
 *  by the library for a given pair of SSVD ranks match the
 *  theoretical expected amount of compression.
 */
TYPED_TEST(DlCompressionSVDTest, PERFORMANCE_TestRankSelectionConvSuccessive)
{
#ifdef USE_OPENCV
    std::string layerName = "conv2";
    std::vector<int> shape;
    float errorThreshold = 5e-1;
    int M                = 32;   // rows of Conv layer
    int N                = 64;   // cols of Conv layer
    int k_h              = 5;    // kernel height
    int k_w              = 5;    // kernel width
    int Nkk              = N * k_h * k_w;

    // Store dims in 'Caffe' style
    shape.push_back(N);
    shape.push_back(M);
    shape.push_back(k_h);
    shape.push_back(k_w);
    this->CreateLayer(LAYER_TYPE_CONV, TYPE_SUCCESSIVE, INIT_RANDOM, shape, false);

    size_t layerSize              = M * Nkk;
    int optimalRankIndex          = 8;   // Pick a rank index as optimal
    float optimalCompressionScore = 0.0;

    // Perform SSVD rank analysis with different metrics.
    std::vector<NETWORK_COST_METRIC> metric_types = {COST_TYPE_MEMORY, COST_TYPE_MAC};

    for (auto metric: metric_types)
    {
        std::shared_ptr<ISVD<float> > svdObj = std::shared_ptr<ISVD<float> >(GetSVDInstance<float>());
        svdObj->SetCostMetric(metric);
        svdObj->StoreLayerAttributes(layerName, this->m_layerAttrib);
        // Compute network cost
        svdObj->ComputeNetworkCost();

        int numCandidateRanks = svdObj->SetCandidateRanks(20);
        for (int rankIndex = 0; rankIndex < numCandidateRanks; ++rankIndex)
        {
            //  svdObj->PrintCandidateRanks (rankIndex, false);
            std::vector<unsigned int> ranks = svdObj->GetCandidateRanks(layerName, rankIndex);
            // Get relative compression score at each rank.
            float compressionScore = svdObj->GetCompressionScore(rankIndex, false, 0, 0);
            // Store the 'optimal' compression score
            if (rankIndex == optimalRankIndex)
                optimalCompressionScore = compressionScore;

            // Compare theoretical compression score.
            ASSERT_EQ(ranks.size(), 2);
            unsigned int r                 = ranks.at(0);
            unsigned int s                 = ranks.at(1);
            size_t reducedLayerSize        = M * r + r * s * k_h * k_w + s * N;
            float expectedCompressionScore = float(layerSize - reducedLayerSize) / float(layerSize);
            ASSERT_LE(fabs(compressionScore - expectedCompressionScore), errorThreshold);
        }
        // Now store optimal ranks for the network
        svdObj->StoreBestRanks(optimalRankIndex);
        // Get compression score with bestRanks
        float bestCompressionScore = svdObj->GetCompressionScore(0, true, 0, 0);
        // Compare with optimal compression score
        ASSERT_LE(fabs(bestCompressionScore - optimalCompressionScore), errorThreshold);
    }
#endif
}

/* Sanity Test: Test the mean shift correction (MSC) for an FC layer with bias.
 *  We split both the weights and the bias of this layer. The weights will
 *  incur some error. The library will try to make up for this by adapting the
 *  bias.
 */
TYPED_TEST(DlCompressionSVDTest, SANITY_TestMscFCSingle)
{
#ifdef USE_OPENCV
    float errorThreshold  = 1e-4;
    std::string layerName = "ip1";
    std::vector<int> shape;
    int M = 40;   // rows of FC layer
    int N = 30;   // cols of FC layer
    // Store dims in 'Caffe' style
    shape.push_back(N);
    shape.push_back(M);

    // Create rank vector
    std::vector<unsigned int> ranks;
    // selected a fairly small rank to perform compression
    unsigned int r = 20;
    ranks.push_back(r);
    // Create weights vector
    std::vector<float*> splitWeights;
    std::vector<unsigned int> weightSizes;
    // Create bias vector
    std::vector<float*> splitBiases;
    std::vector<unsigned int> biasSizes;

    this->CreateLayer(LAYER_TYPE_FC, TYPE_SINGLE, INIT_RANDOM, shape, true);
    cv::Mat original_Matrix(N, M, CV_32F, this->m_layerAttrib.blobs[0].data());
    cv::Mat original_Bias(N, 1, CV_32F, this->m_layerAttrib.blobs[1].data());
    cv::Mat average_Inputs(M, 1, CV_32F, this->m_layerAttrib.inputChannelMean.data());
    // Create an instance of the SVD object
    ISVD<float>* svdObj = GetSVDInstance<float>();
    svdObj->StoreLayerAttributes(layerName, this->m_layerAttrib);

    // Create the data structure for the split weights.
    unsigned int weight1_size = M * r;
    std::unique_ptr<float> weights1(new float[weight1_size]);
    splitWeights.push_back(weights1.get());
    weightSizes.push_back(weight1_size);

    unsigned int weight2_size = r * N;
    std::unique_ptr<float> weights2(new float[weight2_size]);
    splitWeights.push_back(weights2.get());
    weightSizes.push_back(weight2_size);

    // Split the weights.
    svdObj->SplitLayerWeights(layerName, splitWeights, weightSizes, ranks);

    // Split weights are in transposed form.
    // Store them in proper form in intermediate matrices.
    cv::Mat WeightMat_0(r, M, CV_32F, (float*) (splitWeights[0]));
    cv::Mat WeightMat_1(N, r, CV_32F, (float*) (splitWeights[1]));

    // Reconstruct the original matrix by representing matrices
    // in their true mathematical form by nullifying the transpose
    // applied by the SVD algorithm.
    cv::Mat Reconstructed_Matrix = WeightMat_0.t() * WeightMat_1.t();

    // Create the data structure for the split biases.
    unsigned int bias1_size = r;
    std::vector<float> bias1(bias1_size);
    splitBiases.push_back(bias1.data());
    biasSizes.push_back(bias1_size);

    unsigned int bias2_size = N;
    std::vector<float> bias2(bias2_size);
    splitBiases.push_back(bias2.data());
    biasSizes.push_back(bias2_size);

    // Split layer bias.
    svdObj->SplitLayerBiases(layerName, splitBiases, biasSizes, ranks);

    // Copy the result into cv matrices.
    cv::Mat BiasMat_0(r, 1, CV_32F, (float*) (splitBiases[0]));
    cv::Mat BiasMat_1(N, 1, CV_32F, (float*) (splitBiases[1]));

    // Compute the expected bias, including the MSC correction.
    cv::Mat Bias0_Expected = cv::Mat::zeros(r, 1, CV_32F);
    // The second's layer bias gets computed using the MSC method. A correction
    // term is added to the original bias, to make up for the reconstrution
    // error. We know the reconstruction error as well as the average of this
    // layer's input channels. Using these two matrices, we can compute the bias
    // correction.
    cv::Mat Bias1_Expected = (original_Matrix.t() - Reconstructed_Matrix).t() * average_Inputs + original_Bias;
    // Check that the split biases have low L2 error.
    float bias0_error = cv::norm(Bias0_Expected, BiasMat_0, (cv::NORM_RELATIVE | cv::NORM_L2));
    float bias1_error = cv::norm(Bias1_Expected, BiasMat_1, (cv::NORM_RELATIVE | cv::NORM_L2));
    EXPECT_EQ(bias0_error, 0);
    EXPECT_LE(bias1_error, errorThreshold);

    // Run some actual data through this layer, and check if our bias correction
    // improves the accuracy.
    // As layer input, we use some data which looks similar, but not identical,
    // to the average layer input.
    cv::Mat Delta(M, 1, CV_32F);
    cv::randu(Delta, cv::Scalar(-0.1), cv::Scalar(0.1));
    cv::Mat Layer_Input = average_Inputs + Delta;
    // First we compute the reference output of this layer.
    cv::Mat Output_Reference = Layer_Input.t() * original_Matrix.t() + original_Bias.t();
    // Now compute the output of the compressed layer, without mean shift
    // correction.
    cv::Mat Output_No_Msc = Layer_Input.t() * WeightMat_0.t() * WeightMat_1.t() + original_Bias.t();
    // Now compute the output of the compressed layer, with mean shift
    // correction.
    cv::Mat Output_With_Msc = Layer_Input.t() * WeightMat_0.t() * WeightMat_1.t() + BiasMat_1.t();
    // Compute the error with and without MSC, using L2 norm.
    float error_no_msc   = cv::norm(Output_Reference, Output_No_Msc, (cv::NORM_RELATIVE | cv::NORM_L2));
    float error_with_msc = cv::norm(Output_Reference, Output_With_Msc, (cv::NORM_RELATIVE | cv::NORM_L2));
    // Verify that MSC improves the accuracy.
    float expected_improvement_factor = 5;
    EXPECT_LE(error_with_msc, error_no_msc / expected_improvement_factor);
#endif
}
