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

std::vector<cv::Scalar> Colorizer::computeAverageNeighbourIntensities(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels, std::vector<cv::Scalar> avgIntensities) {
    std::vector<cv::Scalar> average_neighbour_intensities(num_superpixels);
    cv::Mat mask = (labels == labels);
    cv::Mat dilatedMask;
    cv::dilate(mask, dilatedMask, cv::Mat());

    cv::Mat neighborMask = dilatedMask & ~mask;
    cv::Mat neighborLabels;
    labels.copyTo(neighborLabels, neighborMask);
    for (int i = 0; i < num_superpixels; i++) {
        cv::Mat mask = (neighborLabels == i);
        cv::Mat dilatedMask;
        cv::dilate(mask, dilatedMask, cv::Mat());

        cv::Mat neighborMask = dilatedMask & ~mask;
        cv::Mat neighborLabels;
        labels.copyTo(neighborLabels, neighborMask);
        cv::Scalar totalIntensity = cv::Scalar::all(0);
        int count = 0;
        for (int i = 0; i < neighborLabels.rows; ++i)
        {
            for (int j = 0; j < neighborLabels.cols; ++j)
            {
                int neighborLabel = neighborLabels.at<int>(i, j);
                if (neighborLabel != i && neighborLabel != 0)
                {
                    totalIntensity += avgIntensities[neighborLabel];
                    ++count;
                }
            }
        }
        average_neighbour_intensities[i] = totalIntensity / count;
    }
    

}