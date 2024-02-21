#ifndef IMGCOLORIZER_COLORIZER_H
#define IMGCOLORIZER_COLORIZER_H

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/lsc.hpp>
#include <opencv2/ximgproc/slic.hpp>
#include <opencv2/xfeatures2d.hpp>
namespace colorizer{
    class Colorizer
    {
    private:
        cv::Mat reference_img, preprocessed_ref_img, ref_superpixels_labels, ref_superpixels_features;
        struct cv::Ptr<cv::ximgproc::SuperpixelLSC> superpixels_ref;
        cv::ximgproc::SLICType superpixel_algo;
        static cv::ximgproc::SLICType evaluateAlgo(const std::string &algorithm);
        cv::Mat blurImage(const cv::Mat &input_img);
        cv::Ptr<cv::ximgproc::SuperpixelLSC> createSuperPixels(cv::Mat input_img, uint region_size = 40, float ruler = 10.0f);
        std::vector<std::vector<cv::Scalar>>
        extractFeatures(const cv::Mat &input_img, const cv::Mat &input_superpixels, const std::size_t num_superpixels);
        std::vector<unsigned int> cascadeFeatureMatching(const cv::Mat &target_features, const cv::Mat &target_superpixels, const std::size_t target_num_superpixels);
        void applyColorTransfer(const cv::Mat input_img, const cv::Mat &input_superpixels, const unsigned int &num_superpixels, const std::vector<unsigned int> &target_ref_matches, cv::Mat &output_img);

        // static methods
        // Feature extraction
        static std::vector<cv::Scalar> computeAverageIntensities(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels);
        static std::vector<std::set<unsigned int>> findSuperPixelNeighbours(const cv::Mat &labels, const std::size_t num_superpixels);
        static std::vector<cv::Scalar> computeAverageNeighbourIntensities(const std::vector<std::set<unsigned int>> &neighbourhoods, const std::size_t num_superpixels, std::vector<cv::Scalar> avgIntensities);
        static void computePixelStdDev(const cv::Mat &input_img, cv::Mat &stddev_img);
        static std::vector<cv::Scalar> computeAverageStdDev(const cv::Mat &stddev_img, const cv::Mat &labels, const std::size_t num_superpixels);
        static std::vector<cv::Scalar> computeAverageNeighbourStdDev(const std::vector<std::set<unsigned int>> &neighbourhoods, const std::size_t num_superpixels, std::vector<cv::Scalar> avgStdDev);
        static void applyFeatureKernel(const cv::Mat &input_img, const cv::Mat &kenel, cv::Mat &output_img);
        static std::vector<cv::Scalar>
        computeAverageFeatureKernel(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels, const cv::Mat &kernel);
        static std::vector<std::vector<cv::Scalar>>
        returnGaborFeatures(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels);
        static std::vector<cv::KeyPoint> applySURF(const cv::Mat &input_img, const cv::Mat &mask, cv::Mat &descriptors);
        static std::vector<std::vector<cv::Scalar>> returnSURFFeatures(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels);

        // Feature matching
        static void matchFeatures(const cv::Mat &target_features, const cv::Mat &ref_features, std::vector<unsigned int> &ref_superpixels);

        // Color transfer
        static cv::Point2i computeCentroids(const cv::Mat &superpixels, const unsigned int &label);
        static cv::Mat sumAbsDiff(const cv::Mat &img1, const cv::Mat &img2);
        static cv::Mat getColorExact(const cv::Mat &color_img, const cv::Mat &yuv_img);
        static cv::Mat getVolColor(const cv::Mat &color_img, const cv::Mat &yuv_img, float winSize = 0.0f, int deg = 0.0f, float idy_pr = 0.0f, float idx_pr = 0.0f, int in_itr_num = 5, int out_itr_num = 1);

    public:
        Colorizer(/* args */);
        Colorizer(const cv::Mat &reference_img);
        void setReferenceImg(const cv::Mat &reference_img);
        void setSuperPixelAlgorithm(const std::string &superpixel_algorithm);
        int colorizeGreyScale(const cv::Mat &input_img, cv::Mat &output_img);
    };
}
#endif
