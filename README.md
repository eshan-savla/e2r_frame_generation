# E2R Frame generation

## Description

This repository aims to create a pipeline to pass in a color video with correspondting events and return a high frame rate, deblurred color video in return. The necessary impelentations for this procedure can easily be adapted to be used with ROS/ROS2 by writing respective wrappers.

## Installation
1. ### Clone the repository.
```
git clone git@github.com:eshan-savla/e2r_frame_generation.git
```

2. ### Installing dependenceis:
#### C++:
This repository requires the OpenCV library along with non-free modules, which need to be built from source. The following steps can be taken to install the library
```
sudo apt update && sudo apt install -y cmake g++ wget unzip
mkdir -p ~/OpenCV && cd ~/OpenCV
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
rm -rf opencv.zip
rm -rf opencv_contrib.zip
mkdir -p build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules -DOPENCV_ENABLE_NONFREE=On ../opencv-4.x
make
sudo make install
```

#### Python:
All python dependencies can be installed into a virtual environment using the setup_env.bash script as follows:
```
bash ./setup_env.bash
```

3. Build the project using the build system of your choice.
While the python scripts require no compiling, a CMakeLists.txt file has been provided to compile and run the colorizer library. It can be built as follows:
```
mkdir -p build && cd build
cmake ..
make
```
## Usage

Explain how to use your project here. Provide examples and instructions if necessary.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request.

## License

Specify the license under which your project is distributed.

## Contact

Provide contact information for users to reach out to you with questions or feedback.
