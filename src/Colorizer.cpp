#include "Colorizer.h"
#include <numeric>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/core/eigen.hpp>
// Constructors

using namespace colorizer;
Colorizer::Colorizer() = default;

Colorizer::Colorizer(const cv::Mat &reference_img) {
    setReferenceImg(reference_img);
}


// Setters & Getters
void Colorizer::setReferenceImg(const cv::Mat &reference_img){
    this->reference_img = reference_img;
    cv::cvtColor(reference_img, preprocessed_ref_img, cv::COLOR_BGR2Lab);
    superpixels_ref = createSuperPixels(preprocessed_ref_img);
    superpixels_ref->getLabels(ref_superpixels_labels);
    std::vector<std::vector<cv::Scalar>> ref_superpixels_features_vec = extractFeatures(preprocessed_ref_img, ref_superpixels_labels, superpixels_ref->getNumberOfSuperpixels());
    // std::vector<cv::Scalar> ref_superpixels_features_vec_flat;
    // for (int i = 0; i < ref_superpixels_features_vec.size(); i++) {
    //     for (int j = 0; j < ref_superpixels_features_vec[i].size(); j++) {
    //         ref_superpixels_features_vec_flat.push_back(ref_superpixels_features_vec[i][j]);
    //     }
    // }
    cv::Mat features_matrix(superpixels_ref->getNumberOfSuperpixels(), 172, CV_64F);
    // for (int i = 0; i < 172; i++) {
    //     cv::Mat sample = cv::Mat(1, 172, CV_64F, ref_superpixels_features_vec_flat.data()).clone();
    //     features_matrix.push_back(sample);
    // }

    assert(ref_superpixels_features_vec.size() == superpixels_ref->getNumberOfSuperpixels());

    for (int i = 0; i < superpixels_ref->getNumberOfSuperpixels(); i++) {
        if(i == 574)
            int p =1;
        for (int j = 0; j < 172; j++) {
            if(j == 171)
                int b= 0;
            features_matrix.at<cv::Scalar>(i,j) = ref_superpixels_features_vec.at(i).at(j);
        }
    }
    auto mat_val00 = features_matrix.ptr<cv::Scalar>(0,0);
    auto mat_val01 = features_matrix.ptr<cv::Scalar>(0,1);
    auto mat_val10 = features_matrix.ptr<cv::Scalar>(1,0);
    auto mat_val11 = features_matrix.ptr<cv::Scalar>(1,1);
    auto vec_val00 = ref_superpixels_features_vec[0][0];
    auto vec_val01 = ref_superpixels_features_vec[0][1];
    auto vec_val10 = ref_superpixels_features_vec[1][0];
    auto vec_val11 = ref_superpixels_features_vec[1][1];
    auto mat_val = features_matrix.ptr<cv::Scalar>(2,171);
    auto vec_val = ref_superpixels_features_vec[2][171];
    int a = 0;
//    int a = 0;
//    ref_superpixels_features = features_matrix;
}
void Colorizer::setSuperPixelAlgorithm(const std::string &superpixel_algorithm){
    superpixel_algo = evaluateAlgo(superpixel_algorithm);
}


cv::ximgproc::SLICType Colorizer::evaluateAlgo(const std::string &algorithm)
{
    if (algorithm == "SLIC")
        return cv::ximgproc::SLIC;
    if(algorithm == "SLICO")
        return cv::ximgproc::SLICO;
    if(algorithm == "MSLIC")
        return cv::ximgproc::MSLIC;
    else
        return cv::ximgproc::SLIC;
}

cv::Mat Colorizer::blurImage(const cv::Mat & input_img) {
    cv::Mat output_img;
    cv::GaussianBlur(input_img, output_img, cv::Size(3,3),0);
    return output_img;
}

int Colorizer::colorizeGreyScale(const cv::Mat &input_img, cv::Mat &output_img) {
    std::size_t num_superpixels;
    if (superpixels_ref.empty()) {
        setReferenceImg(reference_img);
    }
    if (input_img.empty()) {
        return -1;
    }
    if (superpixels_ref.empty()) {
        return -1;
    }
//    if (input_img.channels() > 1) {
//        return -1;
//    }
    auto input_superpixels = createSuperPixels(input_img);
    cv::Mat superpixels_labels;
    input_superpixels->getLabels(superpixels_labels);
    num_superpixels = input_superpixels->getNumberOfSuperpixels();
    std::vector<std::vector<cv::Scalar>> target_feature_vecs = extractFeatures(input_img, superpixels_labels, num_superpixels);
    cv::Mat target_feature_matrix(num_superpixels, 172, CV_64F);
    for (int i = 0; i < num_superpixels; i++) {
        for (int j = 0; j < 172; j++) {
            target_feature_matrix.at<cv::Scalar>(i,j) = target_feature_vecs[i][j];
        }
    }
    cv::Mat outputLabels;
    auto target_ref_matches = cascadeFeatureMatching(target_feature_matrix, superpixels_labels, num_superpixels);
    applyColorTransfer(input_img, superpixels_labels, num_superpixels, target_ref_matches, output_img);
    return 0;
}

cv::Ptr<cv::ximgproc::SuperpixelLSC> Colorizer::createSuperPixels(cv::Mat input_img, uint region_size, float ruler) {
    cv::Mat output_labels;
    cv::Mat blurred_img = blurImage(input_img);
        if (blurred_img.channels() > 1)
        cv::cvtColor(blurred_img, blurred_img, cv::COLOR_BGR2Lab);
    auto superpixels = cv::ximgproc::createSuperpixelLSC(blurred_img, region_size, ruler);
    superpixels->iterate();
    return superpixels;
}

std::vector<std::vector<cv::Scalar>>
Colorizer::extractFeatures(const cv::Mat &input_img, const cv::Mat &input_superpixels, const std::size_t num_superpixels) {
    std::vector<cv::Scalar> average_intensities = computeAverageIntensities(input_img, input_superpixels, num_superpixels);
    std::vector<cv::Scalar> average_stddevs = computeAverageStdDev(input_img, input_superpixels, num_superpixels);
    std::vector<std::set<unsigned int>> neighbourhoods = findSuperPixelNeighbours(input_superpixels, num_superpixels);
    std::vector<cv::Scalar> average_neighbour_intensities = computeAverageNeighbourIntensities(neighbourhoods, num_superpixels, average_intensities);
    std::vector<cv::Scalar> average_neighbour_stddevs = computeAverageNeighbourStdDev(neighbourhoods, num_superpixels, average_stddevs);
    std::vector<std::vector<cv::Scalar>> gaborFeatures = returnGaborFeatures(input_img, input_superpixels, num_superpixels);
    std::vector<std::vector<cv::Scalar>> surfFeatures = returnSURFFeatures(input_img, input_superpixels, num_superpixels);
    std::vector<std::vector<cv::Scalar>> featureMatrix(num_superpixels, std::vector<cv::Scalar>(172));
    for (int i = 0; i < num_superpixels; i++) {
        for(int j = 0; j < 40; j++)
            featureMatrix[i][j] = gaborFeatures[i][j];
        for(int j = 40; j < 168; j++)
            featureMatrix[i][j] = surfFeatures[i][j-40];
        featureMatrix[i][168] = average_intensities[i];
        featureMatrix[i][169] = average_stddevs[i];
        featureMatrix[i][170] = average_neighbour_intensities[i];
        featureMatrix[i][171] = average_neighbour_stddevs[i];
    }
    return featureMatrix;
}

std::vector<unsigned int> Colorizer::cascadeFeatureMatching(const cv::Mat &target_features, const cv::Mat &target_superpixels, const std::size_t target_num_superpixels) {
    float weights[4] = {0.2, 0.5, 0.2 , 0.1};
    std::vector<unsigned int> target_ref_matches_map(target_num_superpixels, 0);
    for (int i = 0; i < target_num_superpixels; i++) {
        cv::Mat target_feature = target_features.row(i);
        int indexes[6] = {0, 40, 168, 169, 170, 171};
        std::vector<unsigned int> ref_superpixels_indexes(superpixels_ref->getNumberOfSuperpixels());
        std::iota(ref_superpixels_indexes.begin(), ref_superpixels_indexes.end(), 0);
        for (int j = 1; j < 6; j++) {
            matchFeatures(target_feature.colRange(indexes[j-1], indexes[j]), ref_superpixels_features.colRange(indexes[j-1], indexes[j]), ref_superpixels_indexes);
            ref_superpixels_indexes = std::vector<unsigned int>(ref_superpixels_indexes.begin(), ref_superpixels_indexes.begin() + ref_superpixels_indexes.size()/2);
        }
        int best_match_index = *std::min_element(ref_superpixels_indexes.begin(), ref_superpixels_indexes.end(), [&](unsigned int a, unsigned int b) {
            float costA, costB;
            costA = weights[0] * cv::norm(target_feature.colRange(0, 40) - ref_superpixels_features.colRange(0, 40).row(a)) + weights[1] * cv::norm(target_feature.colRange(40, 168) - ref_superpixels_features.colRange(40, 168).row(a)) + weights[2] * cv::norm(target_feature.colRange(168, 170) - ref_superpixels_features.colRange(168, 170).row(a)) + weights[3] * cv::norm(target_feature.colRange(170, 172) - ref_superpixels_features.colRange(170, 172).row(a));
            costB = weights[0] * cv::norm(target_feature.colRange(0, 40) - ref_superpixels_features.colRange(0, 40).row(b)) + weights[1] * cv::norm(target_feature.colRange(40, 168) - ref_superpixels_features.colRange(40, 168).row(b)) + weights[2] * cv::norm(target_feature.colRange(168, 170) - ref_superpixels_features.colRange(168, 170).row(b)) + weights[3] * cv::norm(target_feature.colRange(170, 172) - ref_superpixels_features.colRange(170, 172).row(b));
            return costA < costB;
        });
        target_ref_matches_map[i] = ref_superpixels_indexes[best_match_index];
    }
    return target_ref_matches_map;
}

void Colorizer::applyColorTransfer(const cv::Mat input_img, const cv::Mat &input_superpixels, const unsigned int &num_superpixels, const std::vector<unsigned int> &target_ref_matches, cv::Mat &output_img) {
    output_img = cv::Mat::zeros(input_img.size(), input_img.type());
    cv::Mat ref_cie_img;
    cv::cvtColor(reference_img, ref_cie_img, cv::COLOR_BGR2Lab);
    cv::Mat input_img_cie;
    cv::cvtColor(input_img, input_img_cie, cv::COLOR_GRAY2BGR);
    cv::cvtColor(input_img_cie, input_img_cie, cv::COLOR_BGR2Lab);
    for (int i = 0; i < num_superpixels; i++) {
        cv::Rect target_superpixel_rect = cv::boundingRect(input_superpixels == i);
        cv::Rect ref_superpixel_rect = cv::boundingRect(ref_superpixels_labels == target_ref_matches[i]);
        cv::Point target_superpixel_centroid = computeCentroids(input_superpixels, i);
        cv::Point ref_superpixel_centroid = computeCentroids(ref_superpixels_labels, target_ref_matches[i]);
        input_img_cie.at<cv::Vec3b>(target_superpixel_centroid)[1] = ref_cie_img.at<cv::Vec3b>(ref_superpixel_centroid)[1];
        input_img_cie.at<cv::Vec3b>(target_superpixel_centroid)[2] = ref_cie_img.at<cv::Vec3b>(ref_superpixel_centroid)[2];   
    }
    
    cv::cvtColor(input_img_cie, output_img, cv::COLOR_Lab2BGR);
}

cv::Point2i Colorizer::computeCentroids(const cv::Mat &superpixels, const unsigned int &label) {
    cv::Mat mask = (superpixels == label);
    cv::Mat centroids;
    cv::Moments M = cv::moments(mask, true);
    return cv::Point2i(static_cast<int>(M.m10/M.m00),static_cast<int>(M.m01/M.m00));
}

void Colorizer::matchFeatures(const cv::Mat &target_features, const cv::Mat &ref_features, std::vector<unsigned int> &ref_superpixels) {
    std::sort(ref_superpixels.begin(), ref_superpixels.end(), [&](unsigned int a, unsigned int b) {
        return cv::norm(target_features - ref_features.row(a)) < cv::norm(target_features - ref_features.row(b));
    });    
}

std::vector<cv::Scalar> Colorizer::computeAverageIntensities(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels) {
    std::vector<cv::Scalar> average_intensities(num_superpixels);
    for (std::size_t i = 0; i < num_superpixels; i++) {
        cv::Mat mask = (labels == i);
        cv::Scalar mean = cv::mean(input_img, mask);
        average_intensities[i] = mean;
    }
    return average_intensities;
}

void Colorizer::computePixelStdDev(const cv::Mat &input_img, cv::Mat &stddev_img){
    cv::Mat mean, sqmean;    
    cv::boxFilter(input_img, mean, CV_32F, cv::Size(5,5), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    cv::sqrBoxFilter(input_img, sqmean, CV_32F, cv::Size(5,5), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    cv::sqrt(sqmean - mean.mul(mean), stddev_img);
}

std::vector<cv::Scalar> Colorizer::computeAverageStdDev(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels) {
    std::vector<cv::Scalar> average_stddevs(num_superpixels);
    cv::Mat stddev_img;
    computePixelStdDev(input_img, stddev_img);
    for (std::size_t i = 0; i < num_superpixels; i++) {
        cv::Mat mask = (labels == i);
        cv::Scalar mean = cv::mean(stddev_img, mask);
        average_stddevs[i] = mean;
    }
    return average_stddevs;
}

std::vector<std::set<unsigned int>> Colorizer::findSuperPixelNeighbours(const cv::Mat &labels, const std::size_t num_superpixels) {
    std::vector<std::set<unsigned int>> neighbours(num_superpixels);

    for (std::size_t label = 0; label < num_superpixels; ++label) {
        cv::Mat mask = (labels == label);
        cv::Mat dilatedMask;
        cv::dilate(mask, dilatedMask, cv::Mat());

        cv::Mat neighborMask = dilatedMask & ~mask;
        cv::Mat neighbourLabels;
        labels.copyTo(neighbourLabels, neighborMask);

        for (int i = 0; i < neighbourLabels.rows; ++i) {
            for (int j = 0; j < neighbourLabels.cols; ++j) {
                unsigned int neighbourLabel = neighbourLabels.at<int>(i, j);
                if (neighbourLabel != label) {
                    neighbours[label].insert(neighbourLabel);
                }
            }
        }
    }
    return neighbours;
}

std::vector<cv::Scalar> Colorizer::computeAverageNeighbourIntensities(const std::vector<std::set<unsigned int>> &neighbourhoods, const std::size_t num_superpixels, std::vector<cv::Scalar> avgIntensities) {
    std::vector<cv::Scalar> average_neighbour_intensities(num_superpixels);
    for (std::size_t label = 0; label < num_superpixels; label++) {
        std::set<unsigned int> neighbours = neighbourhoods[label];
        cv::Scalar totalIntensity = cv::Scalar::all(0);
        for (int neighbour : neighbours) {
            totalIntensity += avgIntensities[neighbour];
        }
        average_neighbour_intensities[label] = totalIntensity / static_cast<int>(neighbours.size());
    }
    return average_neighbour_intensities;
}

std::vector<cv::Scalar> Colorizer::computeAverageNeighbourStdDev(const std::vector<std::set<unsigned int>> &neighbourhoods, const std::size_t num_superpixels, std::vector<cv::Scalar> avgStdDev) {
    std::vector<cv::Scalar> average_neighbour_stddevs(num_superpixels);
    for (std::size_t label = 0; label < num_superpixels; label++) {
        std::set<unsigned int> neighbours = neighbourhoods[label];
        cv::Scalar totalStdDev = cv::Scalar::all(0);
        for (std::size_t neighbour : neighbours) {
            totalStdDev += avgStdDev[neighbour];
        }
        average_neighbour_stddevs[label] = totalStdDev / static_cast<int>(neighbours.size());
    }
    return average_neighbour_stddevs;
}

void Colorizer::applyFeatureKernel(const cv::Mat &input_img, const cv::Mat &kenel, cv::Mat &output_img) {
    cv::filter2D(input_img, output_img, -1, kenel);
}

std::vector<cv::Scalar> Colorizer::computeAverageFeatureKernel(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels, const cv::Mat &kernel) {
    std::vector<cv::Scalar> average_feature_kernels(num_superpixels);
    for (std::size_t i = 0; i < num_superpixels; i++) {
        cv::Mat mask = (labels == i);
        cv::Mat output_img;
        applyFeatureKernel(input_img, kernel, output_img);
        cv::Scalar mean = cv::mean(output_img, mask);
        average_feature_kernels[i] = mean;
    }
    return average_feature_kernels;
}

std::vector<std::vector<cv::Scalar>>
Colorizer::returnGaborFeatures(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels) {
    std::vector<std::vector<cv::Scalar>> gaborFeatures(num_superpixels, std::vector<cv::Scalar>(40));
    double thetas[8] = {0, M_1_PI/8, 2*M_1_PI/8, 3*M_1_PI/8, 4*M_1_PI/8, 5*M_1_PI/8, 6*M_1_PI/8, 7*M_1_PI/8};
    double lambdas[5] = {0, 1, 2, 3, 4};
    int count = 0;
    for(const auto &theta : thetas) {
        for(const auto &lambda : lambdas) {
            cv::Mat gaborKernel = cv::getGaborKernel(cv::Size(5,5), 1, theta, lambda, 1, 0, CV_32F);
            auto val = computeAverageFeatureKernel(input_img, labels, num_superpixels, gaborKernel);
            for (int i = 0; i < num_superpixels; i++){
                gaborFeatures[i][count] = val[i];
            }
        }
    }
    return gaborFeatures;
}

std::vector<cv::KeyPoint>  Colorizer::applySURF(const cv::Mat &input_img, const cv::Mat &mask, cv::Mat &descriptors) {
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
    surf->setHessianThreshold(400);
    surf->setExtended(true);
    surf->setUpright(true);
    std::vector<cv::KeyPoint> keypoints;
    surf->detectAndCompute(input_img, mask, keypoints, descriptors);
    return keypoints;
}

std::vector<std::vector<cv::Scalar>> Colorizer::returnSURFFeatures(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels) {
    std::vector<std::vector<cv::Scalar>> surfFeatures(num_superpixels, std::vector<cv::Scalar>(128));
    for (int i = 0; i < num_superpixels; i++) {
        cv::Mat mask = (labels == i);
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> keypoints = applySURF(input_img, mask, descriptors);
        for (int j = 0; j < 128; j++) {
            if(descriptors.empty() && keypoints.empty()){
                surfFeatures[i][j] = cv::Scalar(0,0,0,0);
            } else {
            cv::Scalar mean = cv::mean(descriptors.col(j));
            surfFeatures[i][j] = mean;
            }
        }
    }
    return surfFeatures;
}

cv::Mat Colorizer::sumAbsDiff(const cv::Mat &img1, const cv::Mat &img2) {
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    std::vector<cv::Mat> channels(3);
    cv::split(diff, channels);
    cv::Mat sumAbsDiff = channels[0] + channels[1] + channels[2];
    return sumAbsDiff;
}

cv::Mat Colorizer::getColorExact(const cv::Mat &color_img, const cv::Mat &yuv_img) {
    int n = yuv_img.rows;
    int m = yuv_img.cols;
    int img_size = n * m;
    cv::Mat nI = yuv_img.clone();
    Eigen::MatrixXi indices_matrix(n,m);
    // cv::Mat indices_matrix(n,m,CV_32S);
    int counter = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            indices_matrix(i,j) = counter++;
            nI.at<cv::Scalar>(i,j)[1] = 0;
            nI.at<cv::Scalar>(i,j)[2] = 0;
        }
    }
    cv::Mat nonZeroLocations; 
    cv::findNonZero(color_img, nonZeroLocations);
    std::vector<int> lblInds;
    for (int i = 0; i < nonZeroLocations.total(); i++) {
        lblInds.push_back(nonZeroLocations.at<cv::Point>(i).y * color_img.cols + nonZeroLocations.at<cv::Point>(i).x);
    }
    int wd = 1;
    int nr_of_px_in_wd = (2*wd+1)*(2*wd+1);
    int max_nr = img_size * nr_of_px_in_wd;
    int consts_len = 0, len = 0;

    std::vector<int> row_inds = std::vector<int>(max_nr, 0);
    std::vector<int> col_inds = std::vector<int>(max_nr, 0);
    std::vector<double> vals(max_nr, 0);
    Eigen::VectorXd gvals = Eigen::VectorXd::Zero(nr_of_px_in_wd);

    // int length = 0;
    // int pixel_nr = 0;

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            consts_len = consts_len + 1;
            if(!color_img.at<cv::Scalar>(i,j)[0] && !color_img.at<cv::Scalar>(i,j)[1] && !color_img.at<cv::Scalar>(i,j)[2]) {
                int tlen = 0;
                // Eigen::VectorXd window_vals = Eigen::VectorXd::Zero(nr_of_px_in_wd);
                for (int ii = -wd; ii <= wd; ii++) {
                    for (int jj = -wd; jj <= wd; jj++) {
                        if (ii != i && jj != j) {
                            len++; tlen++;
                            row_inds[len - 1] = consts_len;
                            col_inds[len - 1] = indices_matrix(i+ii,j+jj);
                            gvals(tlen - 1) = yuv_img.at<cv::Scalar>(i+ii,j+jj)[0];
                            // window_vals(window_index) = yuv_img.at<cv::Scalar>(i+ii,j+jj)[0];
                            // length++;
                            // window_index++;
                        }
                    }
                }
                double t_val = yuv_img.at<cv::Scalar>(i,j)[0];
                // window_vals(window_index) = center;
                gvals(tlen) = t_val;
                double mean = gvals.head(tlen+1).mean();
                double c_var = (gvals.head(tlen+1).array() - mean).square().mean();
                double csig = c_var * 0.6;
                double mgv = (gvals.head(tlen).array() - t_val).square().minCoeff();
                if(csig < (-mgv / std::log(0.01)))
                    csig = -mgv / std::log(0.01);
                if(csig < 0.000002)
                    csig = 0.000002;
                
                gvals.head(tlen) = (-((gvals.head(tlen).array() - t_val).square()) / csig).exp();
                gvals.head(tlen) /= gvals.head(tlen).sum();
                for (int k = len - tlen; k < len; k++) {
                    vals[k] = -gvals(k - len + tlen);
                }
            }
            len++;
            row_inds[len-1] = consts_len;
            col_inds[len-1] = indices_matrix(i,j);
            vals[len - 1] = 1;
        }
    }

    vals.resize(len);
    row_inds.resize(len);
    col_inds.resize(len);

    Eigen::SparseMatrix<double> A(consts_len, img_size);
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < len; i++) {
        A.insert(row_inds[i], col_inds[i]) = vals[i];
    }
    Eigen::VectorXd b = Eigen::VectorXd::Zero(A.rows());
    std::vector<cv::Mat> yuv_channels;
    cv::split(yuv_img, yuv_channels);

    for (int t = 1; t < 3; t++){
        cv::Mat curIm = yuv_channels[t];
        for (int i = 0; i < lblInds.size(); i++)
            b(lblInds[i]) = curIm.at<double>(lblInds[i]);
        Eigen::VectorXd new_vals = Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>(A).solve(b);
        cv::Mat new_vals_mat(n, m, CV_64F, new_vals.data());
        cv::merge(std::vector<cv::Mat>{nI, new_vals_mat}, nI);
    }
    cv::cvtColor(nI, nI, cv::COLOR_Lab2BGR);
    return nI;
}

// cv::Mat Colorizer::getVolColor(const cv::Mat &color_img, const cv::Mat &yuv_img, float dy_pr = 0.0f, float dx_pr = 0.0f, float idy_pr = 0.0f, float idx_pr = 0.0f, int in_itr_num = 5, int out_itr_num = 1){
//     int winSize=int(dy_pr+0.5);
//     int deg=int(dx_pr+0.5);
//     Eigen::Tensor<float, 3, Eigen::RowMajor> yuv_img_tensor;
//     cv::cv2eigen(yuv_img, yuv_img_tensor);
//     auto sizes = yuv_img_tensor.dimensions();
//     int sizel;
//     sizel=yuv_img.dims;
//     assert(sizel>2);
//     int n=yuv_img.rows;
//     int m=yuv_img.cols;
//     int k,max_d, max_d1,max_d2, in_itr_num, out_itr_num, itr;
//     int x,y,z;

//     if (sizel>3){
//     k=sizes[3];
//     }else{
//     k=1;
//     }
//     max_d1=int(floor(log(n)/log(2)-2)+0.1);
//     max_d2=int(floor(log(m)/log(2)-2)+0.1);
//     if (max_d1>max_d2){
//     max_d=max_d2;
//     }else{
//     max_d=max_d1;
//     }
//     //   double *lblImg_pr, *img_pr;
//     double * res_pr;
//     double **res_prv;
//     double *dx_pr,*dy_pr,*idx_pr,*idy_pr;
//     //   lblImg_pr=mxGetPr(prhs[0]);

//     //   img_pr=mxGetPr(prhs[1]);
//     Tensor3d D,G,I;
//     Tensor3d Dx,Dy,iDx,iDy;
//     MG smk;
//     G.set(m,n,k);
//     D.set(m,n,k);
//     I.set(m,n,k);

//     if (in_itr_num!=5){
//     in_itr_num=int(in_itr_num+0.5);
//     }
//     if (out_itr_num!=2){
//     out_itr_num=int(out_itr_num+0.5);
//     }

//     Dx.set(m,n,k-1);
//     Dy.set(m,n,k-1);
//     iDx.set(m,n,k-1);
//     iDy.set(m,n,k-1);
//     for ( z=0; z<(k-1); z++){
//     for ( y=0; y<n; y++){
//         for ( x=0; x<m; x++){
//     Dx(x,y,z)=dx_pr; dx_pr++;
//     Dy(x,y,z)=dy_pr; dy_pr++;
//     iDx(x,y,z)=idx_pr; idx_pr++;
//     iDy(x,y,z)=idy_pr; idy_pr++;
//         }
//     }
//     }

//     int dims[4];
//     dims[0]=m; dims[1]=n; dims[2]=3; dims[3]=k;
//     // output_img=mxCreateNumericArray(4,dims,  mxDOUBLE_CLASS ,mxREAL);
//     Eigen::Tensor<double, 4, Eigen::RowMajor> output_img_tensor(m,n,3,k);
//     res_pr=mxGetPr(plhs[0]);
//     res_prv=new double*[k];
//     for (z=0; z<k; z++){
//     res_prv[z]=res_pr+n*m*3*z;
//     }


//     for ( z=0; z<k; z++){
//     for ( y=0; y<n; y++){
//         for ( x=0; x<m; x++){
//     I(x,y,z)=lblImg_pr[x+m*y+z*n*m];
//     G(x,y,z)=img_pr[x+y*m+z*m*n*3];
//     I(x,y,z)=!I(x,y,z);     
//         }
//     }
//     }

//     for ( z=0; z<k; z++){
//     for ( y=0; y<n; y++){
//         for ( x=0; x<m; x++){
//     (*res_prv[z])=G(x,y,z);
//     res_prv[z]++;
//         }
//     }
//     }
//     smk.set(m,n,k,max_d);
//     smk.setI(I) ;
//     smk.setG(G);
//     smk.setFlow(Dx,Dy,iDx,iDy);

//     for (int t=1; t<3; t++){
//         for ( z=0; z<k; z++){
//         for ( y=0; y<n; y++){
//         for ( x=0; x<m; x++){
//         D(x,y,z)=img_pr[x+y*m+n*m*t+z*m*n*3];
//         smk.P()(x,y,z)=img_pr[x+y*m+n*m*t+z*m*n*3];
//         D(x,y,z)*=(!I(x,y,z));
//         }
//         }
//         }
        
//         smk.Div() = D ;
        

//         Tensor3d tP2;

        
//         if (k==1){
//         for (itr=0; itr<out_itr_num; itr++){
//         smk.setDepth(max_d);
//         Field_MGN(&smk, in_itr_num, 2) ;
//         smk.setDepth(ceil(max_d/2));
//         Field_MGN(&smk, in_itr_num, 2) ;
//         smk.setDepth(2);
//         Field_MGN(&smk, in_itr_num, 2) ;
//         smk.setDepth(1);
//         Field_MGN(&smk, in_itr_num, 4) ;
//         }
//         } else{
//         for (itr=0; itr<out_itr_num; itr++){
//         smk.setDepth(2);
//         Field_MGN(&smk, in_itr_num, 2) ;
//         smk.setDepth(1);
//         Field_MGN(&smk, in_itr_num, 4) ;
//     }
//         }
        

//         tP2=smk.P();

//         for ( z=0; z<k; z++){
//         for ( y=0; y<n; y++){
//         for ( x=0; x<m; x++){
//         (*res_prv[z])=tP2(x,y,z);
//         res_prv[z]++;
//         }
//         }
//         }
//     }    
// }