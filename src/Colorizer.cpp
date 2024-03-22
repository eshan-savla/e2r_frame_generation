#include "Colorizer.h"
#include <numeric>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/core/eigen.hpp>
// Constructors

using namespace colorizer;
Colorizer::Colorizer() = default;

/**
 * @brief Constructs a Colorizer object with the given reference image.
 *
 * This constructor initializes a Colorizer object with the provided reference image.
 *
 * @param reference_img The reference image used for colorization.
 */
Colorizer::Colorizer(const cv::Mat &reference_img) {
    setReferenceImg(reference_img);
}


// Setters & Getters
/**
 * @brief Sets the reference image for colorization.
 * 
 * This function sets the reference image that will be used for colorization.
 * The reference image should be passed as a `cv::Mat` object.
 * 
 * @param reference_img The reference image to be set.
 */
void Colorizer::setReferenceImg(const cv::Mat &reference_img){
    this->reference_img = reference_img;
    cv::cvtColor(reference_img, preprocessed_ref_img, cv::COLOR_BGR2Lab); // Convert to Lab color space
    superpixels_ref = createSuperPixels(preprocessed_ref_img, 40); // Creates superpixels of average size of 40 pixels
    superpixels_ref->getLabels(ref_superpixels_labels); // Get labels of image pixels. Each pixel is assigned a superpixel label.
    int num_suppix = superpixels_ref->getNumberOfSuperpixels();
    std::cout << "Computing features for reference image... \n";
    std::vector<cv::Scalar> ref_superpixels_features_vec = extractFeatures(preprocessed_ref_img, ref_superpixels_labels, num_suppix);
    cv::Mat features_matrix(superpixels_ref->getNumberOfSuperpixels(), 172, CV_64FC4, ref_superpixels_features_vec.data()); // 172 dimensional feature Matrix containing intensity, std-dev, gabor and surf features.
    ref_superpixels_features = features_matrix;
}

cv::Mat Colorizer::blurImage(const cv::Mat & input_img) {
    cv::Mat output_img;
    cv::GaussianBlur(input_img, output_img, cv::Size(3,3),0);
    return output_img;
}

int Colorizer::colorizeGreyScale(const cv::Mat &input_img, cv::Mat &output_img) {
    int num_superpixels;
    if (superpixels_ref.empty()) {
        setReferenceImg(reference_img);
    }
    if (input_img.empty()) {
        return -1;
    }
    if (superpixels_ref.empty()) {
        return -1;
    }
    auto input_superpixels = createSuperPixels(input_img, 40);
    cv::Mat superpixels_labels;
    input_superpixels->getLabels(superpixels_labels); // Get labels of image pixels. Each pixel is assigned a superpixel label.
    num_superpixels = static_cast<int>(input_superpixels->getNumberOfSuperpixels());
    std::cout << "Computing features for input image... \n";
    std::vector<cv::Scalar> target_feature_vecs = extractFeatures(input_img, superpixels_labels, num_superpixels);
    cv::Mat target_feature_matrix(num_superpixels, 172, CV_64FC4, target_feature_vecs.data()); // 172 dimensional feature Matrix containing intensity, std-dev, gabor and surf features.
    cv::Mat outputLabels;
    std::cout << "Matching features... \n";
    std::vector<int> target_ref_matches = cascadeFeatureMatching(target_feature_matrix, superpixels_labels, num_superpixels);// Match features of input image with reference image.
    int input_img_channels = input_img.channels();
    int target_feature_matrix_channels = target_feature_matrix.channels();
    assert(input_img.channels() == target_feature_matrix.channels());
    int ref_num_superpixels = superpixels_ref->getNumberOfSuperpixels();
    std::cout << "Applying color transfer... \n";
    cv::Mat scribbled_img = applyColorTransfer(input_img, superpixels_labels, num_superpixels, target_ref_matches); // Apply color transfer to input image. Transfer avg color of reference superpixel to centre of b/w superpixel as scribbles.
    output_img = scribbled_img;
    return 0;
}

/**
 * Creates superpixels using the SuperpixelLSC algorithm.
 *
 * @param input_img The input image on which superpixels will be created.
 * @param region_size The desired size of each superpixel region.
 * @param ruler The parameter controlling the compactness of the superpixels.
 * @return A pointer to the created SuperpixelLSC object.
 */
cv::Ptr<cv::ximgproc::SuperpixelLSC> Colorizer::createSuperPixels(const cv::Mat &input_img, uint region_size, float ruler) {
    cv::Mat output_labels;
    cv::Mat blurred_img = blurImage(input_img);
        if (blurred_img.channels() > 1)
        cv::cvtColor(blurred_img, blurred_img, cv::COLOR_BGR2Lab);
    auto superpixels = cv::ximgproc::createSuperpixelLSC(blurred_img, region_size, ruler);
    superpixels->iterate();
    return superpixels;
}

std::vector<cv::Scalar>
/**
 * Extracts features from the input image and input superpixels.
 *
 * @param input_img The input image.
 * @param input_superpixels Superpixels labels for each image pixel.
 * @param num_superpixels The number of superpixels.
 */
Colorizer::extractFeatures(const cv::Mat &input_img, const cv::Mat &input_superpixels, const std::size_t num_superpixels) {
    std::vector<cv::Scalar> average_intensities = computeAverageIntensities(input_img, input_superpixels, num_superpixels);
    std::cout << "Computed average intensities" << std::endl;
    std::vector<cv::Scalar> average_stddevs = computeAverageStdDev(input_img, input_superpixels, num_superpixels);
    std::cout << "Computed average standard deviations" << std::endl;
    std::vector<std::set<unsigned int>> neighbourhoods = findSuperPixelNeighbours(input_superpixels, num_superpixels);
    std::cout << "Computed superpixel neighbourhoods" << std::endl;
    std::vector<cv::Scalar> average_neighbour_intensities = computeAverageNeighbourIntensities(neighbourhoods, num_superpixels, average_intensities);
    std::cout << "Computed average neighbour intensities" << std::endl;
    std::vector<cv::Scalar> average_neighbour_stddevs = computeAverageNeighbourStdDev(neighbourhoods, num_superpixels, average_stddevs);
    std::cout << "Computed average neighbour standard deviations" << std::endl;
    std::vector<std::vector<cv::Scalar>> gaborFeatures = returnGaborFeatures(input_img, input_superpixels, num_superpixels);
    std::cout << "Computed Gabor features" << std::endl;
    std::vector<std::vector<cv::Scalar>> surfFeatures = returnSURFFeatures(input_img, input_superpixels, num_superpixels);
    std::cout << "Computed SURF features" << std::endl;
    std::vector<cv::Scalar> featureMatrix;
    for (int i = 0; i < num_superpixels; i++) {
        for(int j = 0; j < 40; j++)
            featureMatrix.push_back(gaborFeatures[i][j]);
        for(int j = 40; j < 168; j++)
            featureMatrix.push_back(surfFeatures[i][j-40]);
        featureMatrix.push_back(average_intensities[i]);
        featureMatrix.push_back(average_stddevs[i]);
        featureMatrix.push_back(average_neighbour_intensities[i]);
        featureMatrix.push_back(average_neighbour_stddevs[i]);
    }

    return featureMatrix;
}

std::vector<int>
/**
 * Performs cascade feature matching on the given target features and target superpixels.
 *
 * @param target_features The target features to match.
 * @param target_superpixels Superpixels labels for each target image pixel.
 * @param target_num_superpixels The number of target superpixels.
 */
Colorizer::cascadeFeatureMatching(const cv::Mat &target_features, const cv::Mat &target_superpixels, const int target_num_superpixels) {
    double weights[4] = {0.2, 0.5, 0.2 , 0.1}; // Weights for feature matching for each feature type: intensity, std-dev, gabor and surf.
    std::vector<int> target_ref_matches_map(target_num_superpixels, 0);
    for (int i = 0; i < target_num_superpixels; i++) {
        cv::Mat target_feature = target_features.row(i);
        int indexes[6] = {0, 40, 168, 169, 170, 171}; // Indexes for each feature type in the feature matrix.
        std::vector<int> ref_superpixels_indexes(superpixels_ref->getNumberOfSuperpixels()); // Vector to store reference superpixel indexes matched with respective target superpixel.
        std::iota(ref_superpixels_indexes.begin(), ref_superpixels_indexes.end(), 0);
        for (int j = 1; j < 6; j++) {
            matchFeatures(target_feature.colRange(indexes[j-1], indexes[j]), ref_superpixels_features.colRange(indexes[j-1], indexes[j]), ref_superpixels_indexes); // Match features for each feature type.
            ref_superpixels_indexes = std::vector<int>(ref_superpixels_indexes.begin(), ref_superpixels_indexes.begin() + float(ref_superpixels_indexes.size())/2); // Keep only top 50% matches.
        }
        int best_match_index = *std::min_element(ref_superpixels_indexes.begin(), ref_superpixels_indexes.end(), [&](int a, int b) { // Find best match from the remaining matches.
            float costA, costB;
            costA = weights[0] * cv::norm(target_feature.colRange(0, 40) - ref_superpixels_features.colRange(0, 40).row(a)) + weights[1] * cv::norm(target_feature.colRange(40, 168) - ref_superpixels_features.colRange(40, 168).row(a)) + weights[2] * cv::norm(target_feature.colRange(168, 170) - ref_superpixels_features.colRange(168, 170).row(a)) + weights[3] * cv::norm(target_feature.colRange(170, 172) - ref_superpixels_features.colRange(170, 172).row(a));
            costB = weights[0] * cv::norm(target_feature.colRange(0, 40) - ref_superpixels_features.colRange(0, 40).row(b)) + weights[1] * cv::norm(target_feature.colRange(40, 168) - ref_superpixels_features.colRange(40, 168).row(b)) + weights[2] * cv::norm(target_feature.colRange(168, 170) - ref_superpixels_features.colRange(168, 170).row(b)) + weights[3] * cv::norm(target_feature.colRange(170, 172) - ref_superpixels_features.colRange(170, 172).row(b));
            return costA < costB;
        });
        target_ref_matches_map[i] = best_match_index; // Store best match reference superpixel for target superpixel.
    }
    return target_ref_matches_map;
}

/**
 * Applies color transfer to the input image based on the input superpixels and target reference matches.
 *
 * @param input_img The input image to apply color transfer to.
 * @param input_superpixels Superpixels labels for each pixel of input b/w image.
 * @param num_superpixels The number of superpixels in the input image.
 * @param target_ref_matches The best reference superpixel matches of reference image to input superpixels.
 * @return The color transferred image.
 */
cv::Mat Colorizer::applyColorTransfer(const cv::Mat &input_img, const cv::Mat &input_superpixels,
                                      const unsigned int &num_superpixels, const std::vector<int> &target_ref_matches) {
    cv::Mat output_img = cv::Mat::zeros(input_img.size(), input_img.type());
    cv::cvtColor(reference_img, ref_img_lab, cv::COLOR_BGR2Lab);
    cv::Mat input_img_cpy;
    cv::cvtColor(input_img, input_img_cpy, cv::COLOR_BGRA2BGR);
    cv::Mat input_img_cie;
    cv::cvtColor(input_img, input_img_cie, cv::COLOR_BGRA2BGR);
    cv::cvtColor(input_img_cie, input_img_cie, cv::COLOR_BGR2Lab);
    int pos[] = {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6}; // Array defining size of neighbourhood to transfer color. Magnification based size of array
    for (int i = 0; i < num_superpixels; i++) {
        cv::Vec3d average_color = computeAverageColor(input_superpixels, i);
        cv::Point target_superpixel_centroid = computeCentroids(input_superpixels, i);
        for(const auto po:pos){
            for(const auto po2:pos){
                int x = target_superpixel_centroid.x + po;
                int y = target_superpixel_centroid.y + po2;
                if(x < 0 || x >= input_img.cols || y < 0 || y >= input_img.rows)
                    continue;
                // Only transfer chromatic values of reference superpixel to input superpixel. Keep luminance of input superpixel.
                input_img_cie.at<cv::Vec3b>(y, x)[1] = average_color[1];
                input_img_cie.at<cv::Vec3b>(y, x)[2] = average_color[2];
            }
        }
    }
    cv::cvtColor(input_img_cie, output_img, cv::COLOR_Lab2BGR);
    return output_img;
}

/**
 * Computes the centroids of the given superpixels for a specific label.
 *
 * @param superpixels The input matrix containing the superpixels.
 * @param label The label of the superpixels for which centroids need to be computed.
 * @return The computed centroids as a cv::Point2i object.
 */
cv::Point2i Colorizer::computeCentroids(const cv::Mat &superpixels, const int &label) {
    cv::Mat mask;
    mask = (superpixels == label);
    if(cv::countNonZero(mask) == 0)
        return cv::Point2i(0,0);
    cv::Moments M = cv::moments(mask, true);
    return {static_cast<int>(M.m10/M.m00),static_cast<int>(M.m01/M.m00)};
}

/**
 * Matches features between target and reference images.
 *
 * This function takes in target and reference feature matrices and matches the features between them by calculating euclidian distance between feautres.
 * The matched features are stored in the `ref_superpixels` vector.
 *
 * @param target_features The feature matrix of the target image.
 * @param ref_features The feature matrix of the reference image.
 * @param ref_superpixels The vector to store the matched features.
 */
void Colorizer::matchFeatures(const cv::Mat &target_features, const cv::Mat &ref_features, std::vector<int> &ref_superpixels) {
    std::sort(ref_superpixels.begin(), ref_superpixels.end(), [&](unsigned int a, unsigned int b) {
        return cv::norm(target_features - ref_features.row(a)) < cv::norm(target_features - ref_features.row(b));
    });    
}

/**
 * Computes the average intensities for each superpixel in the input image.
 *
 * @param input_img The input image.
 * @param labels The labels matrix indicating the superpixel labels for each pixel.
 * @param num_superpixels The number of superpixels in the image.
 * @return A vector of cv::Scalar containing the average intensities for each superpixel.
 */
std::vector<cv::Scalar> Colorizer::computeAverageIntensities(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels) {
    std::vector<cv::Scalar> average_intensities(num_superpixels);
    for (std::size_t i = 0; i < num_superpixels; i++) {
        cv::Mat mask = (labels == i);
        cv::Scalar mean = cv::mean(input_img, mask);
        average_intensities[i] = mean;
    }
    return average_intensities;
}

/**
 * Computes the standard deviation of each pixel in the input image.
 *
 * @param input_img The input image for which to compute the pixel standard deviation.
 * @param stddev_img The output image where the computed pixel standard deviation will be stored.
 */
void Colorizer::computePixelStdDev(const cv::Mat &input_img, cv::Mat &stddev_img){
    cv::Mat mean, sqmean;    
    cv::boxFilter(input_img, mean, CV_32F, cv::Size(5,5), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    cv::sqrBoxFilter(input_img, sqmean, CV_32F, cv::Size(5,5), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
    cv::sqrt(sqmean - mean.mul(mean), stddev_img);
}

/**
 * Computes the average and standard deviation of color values for each superpixel in the input image.
 *
 * @param input_img The input image.
 * @param labels The label image indicating the superpixel labels.
 * @param num_superpixels The number of superpixels in the image.
 * @return A vector of cv::Scalar objects representing the average and standard deviation of color values for each superpixel.
 */
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

/**
 * Finds the neighboring superpixels for each superpixel in the given image.
 *
 * @param labels The labels matrix containing superpixel label for each image.
 * @param num_superpixels The total number of superpixels in the image.
 * @return A vector of sets, where each set represents the neighbors of a superpixel.
 */
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

/**
 * Computes the average intensities of the neighbors for each superpixel.
 *
 * @param neighbourhoods A vector of sets representing the neighborhoods of each superpixel.
 * @param num_superpixels The total number of superpixels.
 * @param avgIntensities A vector of cv::Scalar representing the average intensities of each superpixel.
 * @return A vector of cv::Scalar representing the computed average intensities of the neighbors for each superpixel.
 */
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

/**
 * Computes the average standard deviation of the neighbors for each superpixel.
 *
 * @param neighbourhoods A vector of sets representing the neighborhoods of each superpixel.
 * @param num_superpixels The total number of superpixels.
 * @param avgStdDev A vector of cv::Scalar representing the average standard deviation for each superpixel.
 * @return A vector of cv::Scalar representing the computed average standard deviation for each superpixel.
 */
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

/**
 * Applies a feature kernel to the input image and stores the result in the output image.
 *
 * @param input_img The input image to apply the feature kernel to.
 * @param kenel The feature kernel to apply.
 * @param output_img The output image to store the result.
 */
void Colorizer::applyFeatureKernel(const cv::Mat &input_img, const cv::Mat &kenel, cv::Mat &output_img) {
    cv::filter2D(input_img, output_img, -1, kenel);
}

/**
 * Computes the average feature kernel for a given input image, labels, number of superpixels, and kernel.
 *
 * @param input_img The input image.
 * @param labels The labels matrix containing superpixel label of each pixel.
 * @param num_superpixels The number of superpixels.
 * @param kernel The kernel matrix.
 * @return A vector of cv::Scalar representing the computed average feature kernel.
 */
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
/**
 * Calculates and returns the Gabor features for the given input image.
 *
 * @param input_img The input image for which Gabor features need to be calculated.
 * @param labels The labels matrix containing superpixel label of each pixel.
 * @param num_superpixels The number of superpixels in the input image.
 * @return The calculated Gabor features.
 */
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

/**
 * Applies the SURF algorithm to detect keypoints and compute descriptors on the input image.
 *
 * @param input_img The input image on which SURF algorithm will be applied.
 * @param mask The optional mask specifying where to look for keypoints.
 * @param descriptors The computed descriptors for the detected keypoints.
 * @return A vector of keypoints detected in the input image.
 */
std::vector<cv::KeyPoint>  Colorizer::applySURF(const cv::Mat &input_img, const cv::Mat &mask, cv::Mat &descriptors) {
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
    surf->setHessianThreshold(400);
    surf->setExtended(true);
    surf->setUpright(true);
    std::vector<cv::KeyPoint> keypoints;
    surf->detectAndCompute(input_img, mask, keypoints, descriptors);
    return keypoints;
}

/**
 * Returns the SURF features for the given input image, labels, and number of superpixels.
 *
 * @param input_img The input image for which SURF features are to be computed.
 * @param labels The labels corresponding to the superpixels in the input image.
 * @param num_superpixels The number of superpixels in the input image.
 * @return A vector of vectors of cv::Scalar representing the SURF features for each superpixel.
 */
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

/**
 * Transfers colors from a scribbled image to a black and white image. This method is incomplete and is only here as reference for future work
 *
 * @param bw_image The black and white image.
 * @param scribbled_image The scribbled image containing color information.
 * @param output_img The output image with transferred colors.
 */
void Colorizer::transferColors(const cv::Mat &bw_image, const cv::Mat &scribbled_image, cv::Mat &output_img) {

    cv::Mat original = bw_image.clone();
    cv::Mat marked = scribbled_image.clone();
    original.convertTo(original, CV_32FC3, 1.0/255.0);
    marked.convertTo(marked, CV_32FC3, 1.0/255.0);
    int bw_channels = original.channels();
    int scr_channels = marked.channels();
    auto bw_type = original.type();
    auto scr_type = marked.type();
    cv::Mat colorIm_ = sumAbsDiff(original, marked);
    cv::Mat colorIm(colorIm_.rows, colorIm_.cols, CV_8U);
    for(int i = 0; i < colorIm.rows; i++) {
        for(int j = 0; j < colorIm.cols; j++) {
            colorIm.at<uchar>(i,j) = colorIm_.at<double>(i,j) > 0.01 ? 1 : 0;
            if(colorIm_.at<double>(i,j) > 0.01) {
                double val_ = colorIm_.at<double>(i,j);
                uchar val = colorIm.at<uchar>(i,j);
                int a = 0;
            }
        }
    }
    cv::Mat ntscIm = marked.clone();
    cv::mixChannels(&original, 1, &ntscIm, 1, new int[2]{0, 0}, 1);

    int max_d = floor(log(std::min(ntscIm.rows, ntscIm.cols))/log(2)-2);
    int iu = floor(ntscIm.rows / pow(2, max_d - 1)) * pow(2, max_d - 1);
    int ju = floor(ntscIm.cols / pow(2, max_d - 1)) * pow(2, max_d - 1);
    colorIm = colorIm(cv::Range(0, iu), cv::Range(0, ju));
    ntscIm = ntscIm(cv::Range(0, iu), cv::Range(0, ju));
    auto ntsc_type = ntscIm.type();
    output_img = getColorExact(colorIm, ntscIm);
}

/**
 * Computes the average color of a superpixel in the given image.
 *
 * @param superpixel The superpixel region represented as a cv::Mat.
 * @param label The label of the superpixel.
 * @return The average color of the superpixel as a cv::Vec3d.
 */
cv::Vec3d Colorizer::computeAverageColor(const cv::Mat &superpixel, int label) {
    cv::Mat mask = (superpixel == label);
    cv::Scalar mean = cv::mean(ref_img_lab, mask);
    return cv::Vec3b(mean[0], mean[1], mean[2]);
}

/**
 * Calculates the sum of absolute differences between two input images.
 *
 * @param img1 The first input image.
 * @param img2 The second input image.
 * @return The resulting image containing the sum of absolute differences.
 */
cv::Mat Colorizer::sumAbsDiff(const cv::Mat &img1, const cv::Mat &img2) {
    cv::Mat diff(img1.size(), img1.type());
    cv::absdiff(img1, img2, diff);
    std::vector<cv::Mat> channels;
    cv::split(diff, channels);
    cv::Mat sumAbsDiff = channels[0] + channels[1] + channels[2];
    int ty = sumAbsDiff.type();
    return sumAbsDiff;
}

/**
 * Spreads color in input image with color scribbles. This method implementation is incomplete and is only here as reference for future work.
 *
 * @param color_img Boolean matrix flagging which pixel is colored.
 * @param yuv_img The lab b/w image with color scribbles.
 * @return The color from the color image.
 */
cv::Mat Colorizer::getColorExact(const cv::Mat &color_img, const cv::Mat &yuv_img) {
    int n = yuv_img.rows;
    int m = yuv_img.cols;
    int img_size = n * m;
    cv::Mat nI = yuv_img.clone();
    Eigen::MatrixXi indices_matrix(n,m);
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
    int c_total = color_img.total();
    int non_z = nonZeroLocations.total();
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

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            consts_len = consts_len + 1;
            if(!color_img.at<uchar>(i,j)) {
                int tlen = 0;
                for (int ii = std::max(0, i -wd); ii <= std::min(i + wd, n - 1); ii++) {
                    for (int jj = std::max(0, j - wd); jj <= std::min(j + wd, m - 1); jj++) {
                        if (ii != i && jj != j) {
                            len++; tlen++;
                            row_inds[len - 1] = consts_len;
                            col_inds[len - 1] = indices_matrix(ii,jj);
                            gvals(tlen - 1) = yuv_img.at<cv::Scalar>(ii,jj)[0];
                        }
                    }
                }
                auto t_vals = yuv_img.at<cv::Vec3f >(i,j);
                float t_val = yuv_img.at<cv::Vec3f>(i,j)[0];
                gvals(tlen) = static_cast<double>(t_val);
                double mean = gvals.head(tlen+1).mean();
                double c_var = (gvals.head(tlen+1).array() - mean).square().mean();
                double csig = c_var * 0.6;
                double mgv = (gvals.head(tlen).array() - t_val).square().minCoeff();
                if(csig < (-mgv / std::log(0.01)))
                    csig = -mgv / std::log(0.01);
                if(csig < 0.000002)
                    csig = 0.000002;
                
                gvals.head(tlen) = (-((gvals.head(tlen).array() - t_val).square()) / csig).exp();
                gvals.head(tlen) = gvals.head(tlen)/gvals.head(tlen).sum();
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

    vals.resize(len + 1);
    row_inds.resize(len + 1);
    col_inds.resize(len + 1);
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < len; i++) {
        if(i == int(len/2))
            int b = 0;
        triplets.push_back(Eigen::Triplet<double>(row_inds[i], col_inds[i], vals[i]));
    }
    Eigen::SparseMatrix<double> A(triplets.size(), triplets.size());

    A.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::VectorXd b = Eigen::VectorXd::Zero(A.rows());
    std::vector<cv::Mat> yuv_channels;
    cv::split(yuv_img, yuv_channels);

    for (int t = 1; t < 3; t++){
        cv::Mat curIm = yuv_channels[t];
        for (const int &lblInd : lblInds)
            b(lblInd) = curIm.at<double>(lblInd);
        Eigen::VectorXd new_vals = Eigen::BiCGSTAB<Eigen::SparseMatrix<double>>(A).solve(b);
        cv::Mat new_vals_mat(n, m, CV_64FC3, new_vals.data());
        cv::merge(std::vector<cv::Mat>{nI, new_vals_mat}, nI);
    }
    nI.convertTo(nI, CV_32FC3, 255);
    nI.convertTo(nI, CV_8UC3);
    cv::cvtColor(nI, nI, cv::COLOR_Lab2BGR);
    return nI;
}