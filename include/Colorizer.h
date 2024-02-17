#ifndef IMGCOLORIZER_COLORIZER_H
#define IMGCOLORIZER_COLORIZER_H

#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/lsc.hpp>
#include <opencv2/ximgproc/slic.hpp>
#include <opencv2/xfeatures2d.hpp>

class Colorizer
{
private:
    cv::Mat reference_img;
    cv::Mat preprocessed_ref_img;
    struct cv::Ptr<cv::ximgproc::SuperpixelLSC> superpixels_ref;
    cv::Mat superpixels_labels;
    cv::ximgproc::SLICType superpixel_algo;
    static cv::ximgproc::SLICType evaluateAlgo(const std::string &algorithm);
    cv::Mat blurImage(const cv::Mat &input_img);
    cv::Ptr<cv::ximgproc::SuperpixelLSC> createSuperPixels(cv::Mat input_img, uint region_size = 40, float ruler = 10.0f);
    void extractFeatures(const cv::Mat &input_img, const cv::Mat &input_superpixels, const std::size_t num_superpixels, cv::Mat &output_img);
    static std::vector<cv::Scalar> computeAverageIntensities(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels);
    static std::vector<std::set<int>> findSuperPixelNeighbours(const cv::Mat &labels, const std::size_t num_superpixels);
    static std::vector<cv::Scalar> computeAverageNeighbourIntensities(const cv::Mat &input_img, const std::vector<std::set<int>> &neighbourhoods, const std::size_t num_superpixels, std::vector<cv::Scalar> avgIntensities);
    static void computePixelStdDev(const cv::Mat &input_img, cv::Mat &stddev_img);
    static std::vector<cv::Scalar> computeAverageStdDev(const cv::Mat &stddev_img, const cv::Mat &labels, const std::size_t num_superpixels);
    static std::vector<cv::Scalar> computeAverageNeighbourStdDev(const cv::Mat &stddev_img, const std::vector<std::set<int>> &neighbourhoods, const std::size_t num_superpixels, std::vector<cv::Scalar> avgStdDev);
    static void applyFeatureKernel(const cv::Mat &input_img, const cv::Mat &kenel, cv::Mat &output_img);
    static std::vector<cv::Scalar> computeAverageFeatureKernel(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels, const cv::Mat &kernel);
    std::vector<std::vector<cv::Scalar>> returnGaborFeatures(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels);
    std::vector<cv::KeyPoint> static applySURF(const cv::Mat &input_img, const cv::Mat &mask, cv::Mat &descriptors);
    static std::vector<std::vector<cv::Scalar>> returnSURFFeatures(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels);

public:
    Colorizer(/* args */);
    Colorizer(const cv::Mat &reference_img);
    void setReferenceImg(const cv::Mat &reference_img);
    void setSuperPixelAlgorithm(const std::string &superpixel_algorithm);
    int colorizeGreyScale(const cv::Mat &input_img, cv::Mat &output_img);
    ~Colorizer();
};

#endif
