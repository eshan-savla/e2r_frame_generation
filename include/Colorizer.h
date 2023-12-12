#ifndef IMGCOLORIZER_COLORIZER_H
#define IMGCOLORIZER_COLORIZER_H

#include <stdio.h>
#include <string>
#include <opencv4/opencv2/opencv.hpp>


class Colorizer
{
private:
    cv::Mat reference_img;

public:
    Colorizer(/* args */);
    
    ~Colorizer();
};

#endif
