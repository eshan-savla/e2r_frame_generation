#include "Colorizer.h"
#include <numeric>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <boost/algorithm/clamp.hpp>
// Constructors
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
    ref_superpixels_features = extractFeatures(preprocessed_ref_img, ref_superpixels_labels, superpixels_ref->getNumberOfSuperpixels());
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
    if (input_img.channels() > 1) {
        return -1;
    }
    auto input_superpixels = createSuperPixels(input_img);
    cv::Mat superpixels_labels;
    input_superpixels->getLabels(superpixels_labels);
    num_superpixels = input_superpixels->getNumberOfSuperpixels();
    cv::Mat target_feature_matrix = extractFeatures(input_img, superpixels_labels, num_superpixels);
    cv::Mat outputLabels;
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

cv::Mat Colorizer::extractFeatures(const cv::Mat &input_img, const cv::Mat &input_superpixels, const std::size_t num_superpixels) {
    std::vector<cv::Scalar> average_intensities = computeAverageIntensities(input_img, input_superpixels, num_superpixels);
    std::vector<cv::Scalar> average_stddevs = computeAverageStdDev(input_img, input_superpixels, num_superpixels);
    std::vector<std::set<unsigned int>> neighbourhoods = findSuperPixelNeighbours(input_superpixels, num_superpixels);
    std::vector<cv::Scalar> average_neighbour_intensities = computeAverageNeighbourIntensities(neighbourhoods, num_superpixels, average_intensities);
    std::vector<cv::Scalar> average_neighbour_stddevs = computeAverageNeighbourStdDev(neighbourhoods, num_superpixels, average_stddevs);
    std::vector<std::vector<cv::Scalar>> gaborFeatures = returnGaborFeatures(input_img, input_superpixels, num_superpixels);
    std::vector<std::vector<cv::Scalar>> surfFeatures = returnSURFFeatures(input_img, input_superpixels, num_superpixels);
    cv::Mat featureMatrix = cv::Mat(num_superpixels, 172, CV_32F);
    for (std::size_t i = 0; i < num_superpixels; i++) {
        featureMatrix.at<std::vector<cv::Scalar>>(i, 0) = gaborFeatures[i];
        featureMatrix.at<std::vector<cv::Scalar>>(i, 40) = surfFeatures[i];
        featureMatrix.at<cv::Scalar>(i, 168) = average_intensities[i];
        featureMatrix.at<cv::Scalar>(i, 169) = average_stddevs[i];
        featureMatrix.at<cv::Scalar>(i, 170) = average_neighbour_intensities[i];
        featureMatrix.at<cv::Scalar>(i, 171) = average_neighbour_stddevs[i];
    }
    
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

std::vector<std::vector<cv::Scalar>> Colorizer::returnGaborFeatures(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels) {
    cv::Mat gaborFeatures(40, num_superpixels, CV_32F);
    double thetas[8] = {0, M_1_PI/8, 2*M_1_PI/8, 3*M_1_PI/8, 4*M_1_PI/8, 5*M_1_PI/8, 6*M_1_PI/8, 7*M_1_PI/8};
    int lambdas[5] = {0, 1, 2, 3, 4};
    int count = 0;
    for(const auto &theta : thetas) {
        for(const auto &lambda : lambdas) {
            cv::Mat gaborKernel = cv::getGaborKernel(cv::Size(5,5), 1, theta, 1/lambda, 1, 0, CV_32F);
            gaborFeatures.at<std::vector<cv::Scalar>>(count++) = computeAverageFeatureKernel(input_img, labels, num_superpixels, gaborKernel);
        }
    }
    // cv::Mat gaborFeaturesTransposed = gaborFeatures.t();
    return cv::Mat(gaborFeatures.t());
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
    cv::Mat surfFeatures(num_superpixels, 128, CV_32F);
    for (std::size_t i = 0; i < num_superpixels; i++) {

        cv::Mat mask = (labels == i);
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> keypoints = applySURF(input_img, mask, descriptors);
        for (std::size_t j = 0; j < 128; j++) {
            cv::Scalar mean = cv::mean(descriptors.col(j));
            surfFeatures.at<cv::Scalar>(i, j) = mean;
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
}

cv::Mat Colorizer::getColorExact(const cv::Mat &color_img, const cv::Mat &yuv_img) {
    int n = yuv_img.rows;
    int m = yuv_img.cols;
    int img_size = n * m;
    // cv::Mat nI = yuv_img.clone();
    Eigen::MatrixXi indices_matrix(n,m);
    // cv::Mat indices_matrix(n,m,CV_32S);
    int counter = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            indices_matrix(i,j) = counter++;
        }
    }
    int wd = 1;
    int nr_of_px_in_wd = (2*wd+1)*(2*wd+1);
    int max_nr = img_size * nr_of_px_in_wd;

    std::vector<int> row_inds = std::vector<int>(max_nr, 0);
    std::vector<int> col_inds = std::vector<int>(max_nr, 0);
    Eigen::VectorXd vals = Eigen::VectorXd::Zero(max_nr);

    int length = 0;
    int pixel_nr = 0;

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            if(!color_img.at<cv::Scalar>(i,j)[0] && !color_img.at<cv::Scalar>(i,j)[1] && !color_img.at<cv::Scalar>(i,j)[2]) {
                int window_index = 0;
                Eigen::VectorXd window_vals = Eigen::VectorXd::Zero(nr_of_px_in_wd);
                for (int ii = -wd; ii <= wd; ii++) {
                    for (int jj = -wd; jj <= wd; jj++) {
                        if (ii != i && jj != j) {
                            row_inds[length] = pixel_nr;
                            col_inds[length] = indices_matrix(i+ii,j+jj);
                            window_vals(window_index) = yuv_img.at<cv::Scalar>(i+ii,j+jj)[0];
                            length++;
                            window_index++;
                        }
                    }
                }
                double center = yuv_img.at<cv::Scalar>(i,j)[0];
                window_vals(window_index) = center;
                double mean = window_vals.head(window_index+1).mean();
                double variance = (window_vals.head(window_index+1).array() - mean).square().mean();
                double sigma = variance * 0.6;
                double mgv = ((window_vals.head(window_index+1).array() - center).square()).minCoeff();
                if(sigma < (-mgv / std::log(0.01)))
                    sigma = -mgv / std::log(0.01);
                if(sigma < 0.000002)
                    sigma = 0.000002;
                
                window_vals.head(window_index) = Eigen::exp(-(window_vals.head(window_index).array() - center).square() / sigma);
                window_vals.head(window_index) = window_vals.head(window_index).array() / (window_vals.head(window_index).sum());
                vals.segment(length-window_index, window_index) = -window_vals.head(window_index);
            }
            row_inds[length] = pixel_nr;
            col_inds[length] = indices_matrix(i,j);
            vals(length) = 1;
            length++;
            pixel_nr++;
        }
    }

    vals = vals.head(length);
    col_inds = std::vector<int>(col_inds.begin(), col_inds.begin() + length);
    row_inds = std::vector<int>(row_inds.begin(), row_inds.begin() + length);

    Eigen::SparseMatrix<double> A(pixel_nr, img_size);
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < length; i++) {
        triplets.push_back(Eigen::Triplet<double>(row_inds[i], col_inds[i], vals(i)));
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::VectorXd b = Eigen::VectorXd::Zero(A.rows());

}