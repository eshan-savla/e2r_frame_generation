cmake_minimum_required(VERSION 3.8)
project(e2r_frame_generation)

# Set the C++ standard to use
set(CMAKE_CXX_STANDARD 14)

add_subdirectory(src)
# Find the required packages
# find_package(OpenCV REQUIRED)

# set($SRC
#     colorizer.cpp
# )

# set($HDRS
#     colorizer.h
# )
# # Add the executable for the colorizer program
# add_library(colorizer colorizer.cpp)
# target_include_directories(colorizer PUBLIC ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(colorizer ${OpenCV_LIBS})

# # Add the executable for the main program
# add_executable(main main.cpp)
# target_link_libraries(main colorizer ${OpenCV_LIBS})
