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
    cv::ximgproc::SLICType superpixel_algo;
    static cv::ximgproc::SLICType evaluateAlgo(const std::string &algorithm);
    cv::Mat blurImage(const cv::Mat &input_img);
    cv::Mat createSuperPixels(cv::Mat input_img, uint region_size = 10, float ruler = 10.0f);
    

public:
    Colorizer(/* args */);
    Colorizer(const cv::Mat &reference_img);
    void setReferenceImg(const cv::Mat &reference_img);
    void setSuperPixelAlgorithm(const std::string &superpixel_algorithm);
    int colorizeGreyScale(const cv::Mat &input_img, cv::Mat &output_img);
    ~Colorizer();
};

#endif
