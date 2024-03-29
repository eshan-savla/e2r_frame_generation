#include "Colorizer.h"
using namespace colorizer;
int main(int argc, char* argv[]){

    cv::Mat reference_img = cv::imread("../data/frames_00000000028.png");
    cv::cvtColor(reference_img, reference_img, cv::COLOR_BGRA2BGR);
    Colorizer colorizer = Colorizer(reference_img);
    cv::Mat input_img = cv::imread("../data/frames_0000000018.png", cv::IMREAD_GRAYSCALE);
    cv::Mat output_img;
    colorizer.colorizeGreyScale(input_img, output_img);
    cv::imwrite("../data/frames_0000000028_colorized.png", output_img);
}
