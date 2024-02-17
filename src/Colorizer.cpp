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
