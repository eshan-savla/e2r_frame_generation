#ifndef IMGCOLORIZER_COLORIZER_H
#define IMGCOLORIZER_COLORIZER_H

#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/slic.hpp>

class Colorizer
{
private:
    cv::Mat reference_img;
    cv::Mat preprocessed_ref_img;
    struct cv::Ptr<cv::ximgproc::SuperpixelSLIC> superpixels_ref;
    cv::Mat superpixels_labels;
    cv::ximgproc::SLICType superpixel_algo;
    static cv::ximgproc::SLICType evaluateAlgo(const std::string &algorithm);
    cv::Mat blurImage(const cv::Mat &input_img);
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> createSuperPixels(cv::Mat input_img, uint region_size = 10, float ruler = 10.0f);
    void extractFeatures(const cv::Mat &input_img, const cv::Mat &input_superpixels, const std::size_t num_superpixels, cv::Mat &output_img);
    static std::vector<cv::Scalar> computeAverageIntensities(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels);
    static std::vector<cv::Scalar> computeAverageNeighbourIntensities(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels, std::vector<cv::Scalar> avgIntensities);
    

public:
    Colorizer(/* args */);
    Colorizer(const cv::Mat &reference_img);
    void setReferenceImg(const cv::Mat &reference_img);
    void setSuperPixelAlgorithm(const std::string &superpixel_algorithm);
    int colorizeGreyScale(const cv::Mat &input_img, cv::Mat &output_img);
    ~Colorizer();
};

#endif
