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

        /**
         * Blurs the input image using a Gaussian filter.
         *
         * @param input_img The input image to be blurred.
         * @return The blurred image.
         */
        cv::Mat blurImage(const cv::Mat &input_img);

        /**
         * Creates superpixels using the SuperpixelLSC algorithm.
         *
         * @param input_img The input image on which superpixels will be created.
         * @param region_size The desired size of each superpixel region.
         * @param ruler The parameter controlling the compactness of the superpixels.
         * @return A pointer to the created SuperpixelLSC object.
         */
        cv::Ptr<cv::ximgproc::SuperpixelLSC> createSuperPixels(const cv::Mat &input_img, uint region_size = 40, float ruler = 10.0f);

        /**
         * Extracts features from the input image and input superpixels.
         *
         * @param input_img The input image.
         * @param input_superpixels Superpixels labels for each image pixel.
         * @param num_superpixels The number of superpixels.
         */
        static std::vector<cv::Scalar> extractFeatures(const cv::Mat &input_img, const cv::Mat &input_superpixels, const std::size_t num_superpixels);

        /**
         * Performs cascade feature matching on the given target features and target superpixels.
         *
         * @param target_features The target features to match.
         * @param target_superpixels Superpixels labels for each target image pixel.
         * @param target_num_superpixels The number of target superpixels.
         */
        std::vector<int> cascadeFeatureMatching(const cv::Mat &target_features, const cv::Mat &target_superpixels, const int target_num_superpixels);

        /**
         * Applies color transfer to the input image based on the input superpixels and target reference matches.
         *
         * @param input_img The input image to apply color transfer to.
         * @param input_superpixels Superpixels labels for each pixel of input b/w image.
         * @param num_superpixels The number of superpixels in the input image.
         * @param target_ref_matches The best reference superpixel matches of reference image to input superpixels.
         * @return The color transferred image.
         */
        cv::Mat applyColorTransfer(const cv::Mat &input_img, const cv::Mat &input_superpixels,
                                   const unsigned int &num_superpixels, const std::vector<int> &target_ref_matches);
        
        /**
         * Computes the average color of a superpixel in the given image.
         *
         * @param superpixel The superpixel region represented as a cv::Mat.
         * @param label The label of the superpixel.
         * @return The average color of the superpixel as a cv::Vec3d.
         */
        cv::Vec3d computeAverageColor(const cv::Mat &superpixel, int label);

        // static methods
        // Feature extraction

        /**
         * Computes the average intensities for each superpixel in the input image.
         *
         * @param input_img The input image.
         * @param labels The labels matrix indicating the superpixel labels for each pixel.
         * @param num_superpixels The number of superpixels in the image.
         * @return A vector of cv::Scalar containing the average intensities for each superpixel.
         */
        static std::vector<cv::Scalar> computeAverageIntensities(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels);

        /**
         * Finds the neighboring superpixels for each superpixel in the given image.
         *
         * @param labels The labels matrix containing superpixel label for each image.
         * @param num_superpixels The total number of superpixels in the image.
         * @return A vector of sets, where each set represents the neighbors of a superpixel.
         */
        static std::vector<std::set<unsigned int>> findSuperPixelNeighbours(const cv::Mat &labels, const std::size_t num_superpixels);

        /**
         * Computes the average intensities of the neighbors for each superpixel.
         *
         * @param neighbourhoods A vector of sets representing the neighborhoods of each superpixel.
         * @param num_superpixels The total number of superpixels.
         * @param avgIntensities A vector of cv::Scalar representing the average intensities of each superpixel.
         * @return A vector of cv::Scalar representing the computed average intensities of the neighbors for each superpixel.
         */
        static std::vector<cv::Scalar> computeAverageNeighbourIntensities(const std::vector<std::set<unsigned int>> &neighbourhoods, const std::size_t num_superpixels, std::vector<cv::Scalar> avgIntensities);

        /**
         * Computes the standard deviation of each pixel in the input image.
         *
         * @param input_img The input image for which to compute the pixel standard deviation.
         * @param stddev_img The output image where the computed pixel standard deviation will be stored.
         */
        static void computePixelStdDev(const cv::Mat &input_img, cv::Mat &stddev_img);

        /**
         * Computes the average and standard deviation of color values for each superpixel in the input image.
         *
         * @param input_img The input image.
         * @param labels The label image indicating the superpixel labels.
         * @param num_superpixels The number of superpixels in the image.
         * @return A vector of cv::Scalar objects representing the average and standard deviation of color values for each superpixel.
         */
        static std::vector<cv::Scalar> computeAverageStdDev(const cv::Mat &stddev_img, const cv::Mat &labels, const std::size_t num_superpixels);

        /**
         * Computes the average standard deviation of the neighbors for each superpixel.
         *
         * @param neighbourhoods A vector of sets representing the neighborhoods of each superpixel.
         * @param num_superpixels The total number of superpixels.
         * @param avgStdDev A vector of cv::Scalar representing the average standard deviation for each superpixel.
         * @return A vector of cv::Scalar representing the computed average standard deviation for each superpixel.
         */
        static std::vector<cv::Scalar> computeAverageNeighbourStdDev(const std::vector<std::set<unsigned int>> &neighbourhoods, const std::size_t num_superpixels, std::vector<cv::Scalar> avgStdDev);

        /**
         * Applies a feature kernel to the input image and stores the result in the output image.
         *
         * @param input_img The input image to apply the feature kernel to.
         * @param kenel The feature kernel to apply.
         * @param output_img The output image to store the result.
         */
        static void applyFeatureKernel(const cv::Mat &input_img, const cv::Mat &kenel, cv::Mat &output_img);
        
        /**
         * Computes the average feature kernel for a given input image, labels, number of superpixels, and kernel.
         *
         * @param input_img The input image.
         * @param labels The labels matrix containing superpixel label of each pixel.
         * @param num_superpixels The number of superpixels.
         * @param kernel The kernel matrix.
         * @return A vector of cv::Scalar representing the computed average feature kernel.
         */
        static std::vector<cv::Scalar>
        computeAverageFeatureKernel(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels, const cv::Mat &kernel);
        
        /**
         * Calculates and returns the Gabor features for the given input image.
         *
         * @param input_img The input image for which Gabor features need to be calculated.
         * @param labels The labels matrix containing superpixel label of each pixel.
         * @param num_superpixels The number of superpixels in the input image.
         * @return The calculated Gabor features.
         */
        static std::vector<std::vector<cv::Scalar>>
        returnGaborFeatures(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels);
        
        /**
         * Applies the SURF algorithm to detect keypoints and compute descriptors on the input image.
         *
         * @param input_img The input image on which SURF algorithm will be applied.
         * @param mask The optional mask specifying where to look for keypoints.
         * @param descriptors The computed descriptors for the detected keypoints.
         * @return A vector of keypoints detected in the input image.
         */
        static std::vector<cv::KeyPoint> applySURF(const cv::Mat &input_img, const cv::Mat &mask, cv::Mat &descriptors);

        /**
         * Returns the SURF features for the given input image, labels, and number of superpixels.
         *
         * @param input_img The input image for which SURF features are to be computed.
         * @param labels The labels corresponding to the superpixels in the input image.
         * @param num_superpixels The number of superpixels in the input image.
         * @return A vector of vectors of cv::Scalar representing the SURF features for each superpixel.
         */
        static std::vector<std::vector<cv::Scalar>> returnSURFFeatures(const cv::Mat &input_img, const cv::Mat &labels, const std::size_t num_superpixels);

        // Feature matching

        /**
         * Matches features between target and reference images.
         *
         * This function takes in target and reference feature matrices and matches the features between them by calculating euclidian distance between feautres.
         * The matched features are stored in the `ref_superpixels` vector.
         *
         * @param target_features The feature matrix of the target image.
         * @param ref_features The feature matrix of the reference image.
         * @param ref_superpixels The vector to store the matched features.
         */
        static void matchFeatures(const cv::Mat &target_features, const cv::Mat &ref_features, std::vector<int> &ref_superpixels);

        // Color transfer

        /**
         * Transfers colors from a scribbled image to a black and white image. This method is incomplete and is only here as reference for future work
         *
         * @param bw_image The black and white image.
         * @param scribbled_image The scribbled image containing color information.
         * @param output_img The output image with transferred colors.
         */
        static void transferColors(const cv::Mat &bw_image, const cv::Mat &scribbled_image, cv::Mat &output_img);
        
        /**
         * Computes the centroids of the given superpixels for a specific label.
         *
         * @param superpixels The input matrix containing the superpixels.
         * @param label The label of the superpixels for which centroids need to be computed.
         * @return The computed centroids as a cv::Point2i object.
         */
        static cv::Point2i computeCentroids(const cv::Mat &superpixels, const int &label);

        /**
         * Calculates the sum of absolute differences between two input images.
         *
         * @param img1 The first input image.
         * @param img2 The second input image.
         * @return The resulting image containing the sum of absolute differences.
         */
        static cv::Mat sumAbsDiff(const cv::Mat &img1, const cv::Mat &img2);

        /**
         * Spreads color in input image with color scribbles. This method implementation is incomplete and is only here as reference for future work.
         *
         * @param color_img Boolean matrix flagging which pixel is colored.
         * @param yuv_img The lab b/w image with color scribbles.
         * @return The color from the color image.
         */
        static cv::Mat getColorExact(const cv::Mat &color_img, const cv::Mat &yuv_img);
        
    public:
        Colorizer();
        /**
         * @brief Constructs a Colorizer object with the given reference image.
         *
         * This constructor initializes a Colorizer object with the provided reference image.
         *
         * @param reference_img The reference image used for colorization.
         */
        Colorizer(const cv::Mat &reference_img);

        /**
         * @brief Sets the reference image for colorization.
         * 
         * This function sets the reference image that will be used for colorization.
         * The reference image should be passed as a `cv::Mat` object.
         * 
         * @param reference_img The reference image to be set.
         */
        void setReferenceImg(const cv::Mat &reference_img);

        /**
         * Colorizes a grayscale image.
         *
         * This function takes a grayscale input image and colorizes it, producing an output image.
         *
         * @param input_img The input grayscale image to be colorized.
         * @param output_img The output colorized image.
         * @return An integer value indicating the success or failure of the colorization process.
         */
        int colorizeGreyScale(const cv::Mat &input_img, cv::Mat &output_img);
    };
}
#endif
