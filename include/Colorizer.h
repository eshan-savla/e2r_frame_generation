#ifndef IMGCOLORIZER_COLORIZER_H
#define IMGCOLORIZER_COLORIZER_H

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/lsc.hpp>
#include <opencv2/ximgproc/slic.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp> 
#include <opencv2/xfeatures2d.hpp>
/**
 * @brief The Colorizer class is responsible for colorizing grayscale images based on a reference image.
 * 
 * It uses various image processing techniques such as superpixel segmentation, feature extraction, feature matching, and color transfer.
 * The colorization process involves mapping the colors from the reference image to the grayscale image based on the similarity of their features.
 * This implementation is incomplete and only maps the colours to the center of each superpixel. Colour spreading and correction need to be implemented.
 * 
 * Usage:
 * 1. Create an instance of the Colorizer class.
 * 2. Set the reference image using the setReferenceImg() method.
 * 3. Call the colorizeGreyScale() method to colorize a grayscale image.
 * 
 * Example:
 * ```
 * cv::Mat reference_img = cv::imread("reference.jpg");
 * cv::Mat grayscale_img = cv::imread("grayscale.jpg", cv::IMREAD_GRAYSCALE);
 * 
 * colorizer::Colorizer colorizer;
 * colorizer.setReferenceImg(reference_img);
 * 
 * cv::Mat colorized_img;
 * colorizer.colorizeGreyScale(grayscale_img, colorized_img);
 * 
 * cv::imshow("Colorized Image", colorized_img);
 * cv::waitKey(0);
 * cv::destroyAllWindows();
 * 
 * cv::imwrite("colorized.jpg", colorized_img);
 * ```
 */
namespace colorizer{
    /**
     * @class Colorizer
     * @brief The Colorizer class is responsible for colorizing grayscale images based on a reference image.
     *
     * The Colorizer class provides methods for colorizing grayscale images using a reference image. It uses various
     * techniques such as superpixel segmentation, feature extraction, feature matching, and color transfer to achieve
     * the colorization process.
     */
    class Colorizer
    {
    private:
        cv::Mat reference_img, ref_img_lab, preprocessed_ref_img, ref_superpixels_labels, ref_superpixels_features;
        struct cv::Ptr<cv::ximgproc::SuperpixelLSC> superpixels_ref;
        cv::Mat blurImage(const cv::Mat &input_img);
        cv::Ptr<cv::ximgproc::SuperpixelLSC> createSuperPixels(const cv::Mat &input_img, uint region_size = 40, float ruler = 10.0f);
        static std::vector<cv::Scalar> extractFeatures(const cv::Mat &input_img, const cv::Mat &input_superpixels, const std::size_t num_superpixels);
        std::vector<int> cascadeFeatureMatching(const cv::Mat &target_features, const cv::Mat &target_superpixels, const int target_num_superpixels);
        cv::Mat applyColorTransfer(const cv::Mat &input_img, const cv::Mat &input_superpixels,
                                   const unsigned int &num_superpixels, const std::vector<int> &target_ref_matches);
        cv::Vec3d computeAverageColor(const cv::Mat &superpixel, int label);

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
        static void matchFeatures(const cv::Mat &target_features, const cv::Mat &ref_features, std::vector<int> &ref_superpixels);

        // Color transfer
        static void transferColors(const cv::Mat &bw_image, const cv::Mat &scribbled_image, cv::Mat &output_img);
        
        static cv::Point2i computeCentroids(const cv::Mat &superpixels, const int &label);
        static cv::Mat sumAbsDiff(const cv::Mat &img1, const cv::Mat &img2);
        static cv::Mat getColorExact(const cv::Mat &color_img, const cv::Mat &yuv_img);
        static cv::Mat getVolColor(const cv::Mat &color_img, const cv::Mat &yuv_img, float winSize = 0.0f, int deg = 0.0f, float idy_pr = 0.0f, float idx_pr = 0.0f, int in_itr_num = 5, int out_itr_num = 1);

    public:
        Colorizer(/* args */);
        Colorizer(const cv::Mat &reference_img);
        void setReferenceImg(const cv::Mat &reference_img);
        int colorizeGreyScale(const cv::Mat &input_img, cv::Mat &output_img);
    };
}
#endif
