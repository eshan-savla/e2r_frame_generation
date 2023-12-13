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

cv::Mat Colorizer::createSuperPixels(cv::Mat input_img, uint region_size, float ruler) {
    cv::Mat output_labels;
    cv::Mat blurred_img = blurImage(input_img);
        if (blurred_img.channels() < 3)
        cv::cvtColor(blurred_img,blurred_img,cv::COLOR_GRAY2BGR, 3);
    cv::cvtColor(blurred_img, blurred_img, cv::COLOR_BGR2Lab);
    auto superpixels = cv::ximgproc::createSuperpixelSLIC(blurred_img, superpixel_algo, region_size, ruler);
    superpixels->iterate();
    superpixels->getLabels(output_labels);
    return output_labels;
}