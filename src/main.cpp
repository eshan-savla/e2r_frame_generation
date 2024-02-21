#include "Colorizer.h"
using namespace colorizer;
int main(int argc, char* argv[]){

    cv::Mat reference_img = cv::imread("/home/eshan/Hochschule/FE_Projekt_Sem1/ros_ws/src/e2r_frame_generation/data/frames_0000000001.png");
    cv::cvtColor(reference_img, reference_img, cv::COLOR_BGRA2BGR);
//    cv::imshow("Reference Image", reference_img);
//    cv::waitKey(0);
    Colorizer colorizer = Colorizer(reference_img);
    cv::Mat input_img = cv::imread("/home/eshan/Hochschule/FE_Projekt_Sem1/ros_ws/src/e2r_frame_generation/data/frames_0000000020.png", cv::IMREAD_GRAYSCALE);
//    cv::imshow("Input Image", input_img);
//    cv::waitKey(0);
    cv::cvtColor(input_img, input_img, cv::COLOR_GRAY2BGR);
    cv::Mat output_img;
    colorizer.colorizeGreyScale(input_img, output_img);
    cv::imshow("Colorized Image", output_img);
    cv::waitKey(0);
}