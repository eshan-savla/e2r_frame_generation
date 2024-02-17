#include "Colorizer.h"


// Constructors
Colorizer::Colorizer() = default;

Colorizer::Colorizer(const cv::Mat &reference_img) : reference_img(reference_img) {
    cv::Mat blurred_img = blurImage(reference_img);
    cv::cvtColor(blurred_img, preprocessed_ref_img, cv::COLOR_BGR2Lab);
}


// Setters & Getters
void Colorizer::setReferenceImg(const cv::Mat &reference_img){
    this->reference_img = reference_img;
    cv::Mat blurred_img = blurImage(reference_img);
    cv::cvtColor(blurred_img, preprocessed_ref_img, cv::COLOR_BGR2Lab);
    superpixels_ref = createSuperPixels(reference_img);
    superpixels_ref->getLabels(superpixels_labels);
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
}

cv::Mat Colorizer::blurImage(const cv::Mat & input_img) {
    cv::Mat output_img;
    cv::GaussianBlur(input_img, output_img, cv::Size(3,3),0);
    return output_img;
}

cv::Ptr<cv::ximgproc::SuperpixelSLIC> Colorizer::createSuperPixels(cv::Mat input_img, uint region_size, float ruler) {
    cv::Mat output_labels;
    cv::Mat blurred_img = blurImage(input_img);
        if (blurred_img.channels() > 1)
        cv::cvtColor(blurred_img, blurred_img, cv::COLOR_BGR2Lab);
    superpixels_ref = cv::ximgproc::createSuperpixelSLIC(blurred_img, superpixel_algo, region_size, ruler);
    superpixels_ref->iterate();
    return superpixels_ref;
}

void Colorizer::extractFeatures(const cv::Mat &input_img, const cv::Mat &input_superpixels, const std::size_t num_superpixels, cv::Mat &output_img) {
    std::vector<cv::Scalar> average_intensities = computeAverageIntensities(input_img, input_superpixels, num_superpixels);
}

std::vector<cv::Scalar> Colorizer::computeAverageIntensities(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels) {
    std::vector<cv::Scalar> average_intensities(num_superpixels);
    for (int i = 0; i < num_superpixels; i++) {
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
    cv::Mat stddevImage = cv::Mat(mean.size(), CV_32F);
    cv::sqrt(sqmean - mean.mul(mean), stddevImage);
}

std::vector<cv::Scalar> Colorizer::computeAverageStdDev(const cv::Mat &stddev_img, const cv::Mat &labels, const std::size_t num_superpixels) {
    std::vector<cv::Scalar> average_stddevs(num_superpixels);
    for (int i = 0; i < num_superpixels; i++) {
        cv::Mat mask = (labels == i);
        cv::Scalar mean = cv::mean(stddev_img, mask);
        average_stddevs[i] = mean;
    }
    return average_stddevs;
}

std::vector<std::set<int>> Colorizer::findSuperPixelNeighbours(const cv::Mat &labels, const std::size_t num_superpixels) {
    std::vector<std::set<int>> neighbours(num_superpixels);

    for (int label = 0; label < num_superpixels; ++label) {
        cv::Mat mask = (labels == label);
        cv::Mat dilatedMask;
        cv::dilate(mask, dilatedMask, cv::Mat());

        cv::Mat neighborMask = dilatedMask & ~mask;
        cv::Mat neighbourLabels;
        labels.copyTo(neighbourLabels, neighborMask);

        for (int i = 0; i < neighbourLabels.rows; ++i) {
            for (int j = 0; j < neighbourLabels.cols; ++j) {
                int neighbourLabel = neighbourLabels.at<int>(i, j);
                if (neighbourLabel != label) {
                    neighbours[label].insert(neighbourLabel);
                }
            }
        }
    }

    return neighbours;
}

std::vector<cv::Scalar> Colorizer::computeAverageNeighbourIntensities(const cv::Mat &input_img, const std::vector<std::set<int>> &neighbourhoods, const std::size_t num_superpixels, std::vector<cv::Scalar> avgIntensities) {
    std::vector<cv::Scalar> average_neighbour_intensities(num_superpixels);
    for (int label = 0; label < num_superpixels; label++) {
        std::set<int> neighbours = neighbourhoods[label];
        cv::Scalar totalIntensity = cv::Scalar::all(0);
        for (int neighbour : neighbours) {
            totalIntensity += avgIntensities[neighbour];
        }
        average_neighbour_intensities[label] = totalIntensity / static_cast<int>(neighbours.size());
    }
}

std::vector<cv::Scalar> Colorizer::computeAverageNeighbourStdDev(const cv::Mat &stddev_img, const std::vector<std::set<int>> &neighbourhoods, const std::size_t num_superpixels, std::vector<cv::Scalar> avgStdDev) {
    std::vector<cv::Scalar> average_neighbour_stddevs(num_superpixels);
    for (int label = 0; label < num_superpixels; label++) {
        std::set<int> neighbours = neighbourhoods[label];
        cv::Scalar totalStdDev = cv::Scalar::all(0);
        for (int neighbour : neighbours) {
            totalStdDev += avgStdDev[neighbour];
        }
        average_neighbour_stddevs[label] = totalStdDev / static_cast<int>(neighbours.size());
    }
}

void Colorizer::applyFeatureKernel(const cv::Mat &input_img, const cv::Mat &kenel, cv::Mat &output_img) {
    cv::filter2D(input_img, output_img, -1, kenel);
}

std::vector<cv::Scalar> Colorizer::computeAverageFeatureKernel(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels, const cv::Mat &kernel) {
    std::vector<cv::Scalar> average_feature_kernels(num_superpixels);
    for (int i = 0; i < num_superpixels; i++) {
        cv::Mat mask = (labels == i);
        cv::Mat output_img;
        applyFeatureKernel(input_img, kernel, output_img);
        cv::Scalar mean = cv::mean(output_img, mask);
        average_feature_kernels[i] = mean;
    }
    return average_feature_kernels;
}

std::vector<std::vector<cv::Scalar>> Colorizer::returnGaborFeatures(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels) {
    std::vector<std::vector<cv::Scalar>> gaborFeatures(40, std::vector<cv::Scalar>(num_superpixels));
    double thetas[8] = {0, M_1_PI/8, 2*M_1_PI/8, 3*M_1_PI/8, 4*M_1_PI/8, 5*M_1_PI/8, 6*M_1_PI/8, 7*M_1_PI/8};
    int lambdas[5] = {0, 1, 2, 3, 4};
    int count = 0;
    for(const auto &theta : thetas) {
        for(const auto &lambda : lambdas) {
            cv::Mat gaborKernel = cv::getGaborKernel(cv::Size(5,5), 1, theta, 1/lambda, 1, 0, CV_32F);
            std::vector<cv::Scalar> gaborFeatures[count++] = computeAverageFeatureKernel(input_img, labels, num_superpixels, gaborKernel);
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
    cv::Mat surfFeatures(128, num_superpixels, CV_32F);
    for (int i = 0; i < num_superpixels; i++) {

        cv::Mat mask = (labels == i);
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> keypoints = applySURF(input_img, mask, descriptors);
        for (int j = 0; j < 128; j++) {
            cv::Scalar mean = cv::mean(descriptors.col(j));
            surfFeatures.at<cv::Scalar>(j, i) = mean;
            }
    }
    return surfFeatures;
}