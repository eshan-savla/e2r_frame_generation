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
    std::string superpixel_algo;
    cv::ximgproc::SuperpixelSLIC createSuperPixels(cv::Mat input_img, uint region_size = 10, float ruler = 10.0f);

public:
    Colorizer(/* args */);
    void setReferenceImg(const cv::Mat &reference_img);
    int colorizeGreyScale(const cv::Mat &input_img, cv::Mat &output_img);
    ~Colorizer();
};

#endif
