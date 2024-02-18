#include "Colorizer.h"

int main(int argc, char** argv[]){
    cv::Mat reference_img = cv::imread("<path_to_img>", cv::IMREAD_COLOR);
    Colorizer colorizer = Colorizer(reference_img);
    cv::Mat input_img = cv::imread("<path_to_img>", cv::IMREAD_GRAYSCALE);
    cv::Mat output_img;
    colorizer.colorizeGreyScale(input_img, output_img);
    cv::imwrite("<path_to_output_img>", output_img);
}