set(LIB_NAME ImgColorizer)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
set(SRC
./Colorizer.cpp
# ./fmg.cpp
# ./mg.cpp
# ./tensor2d.cpp
# ./tensor3d.cpp
)

set(HDRS
${PROJECT_SOURCE_DIR}/include/Colorizer.h
# ${PROJECT_SOURCE_DIR}/include/fmg.h
# ${PROJECT_SOURCE_DIR}/include/mg.h
# ${PROJECT_SOURCE_DIR}/include/tensor2d.h
# ${PROJECT_SOURCE_DIR}/include/tensor3d.h
)

add_library(${LIB_NAME} STATIC ${SRC} ${HDRS})
target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${LIB_NAME} PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(${LIB_NAME}_test ./main.cpp)
target_include_directories(${LIB_NAME}_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${LIB_NAME}_test PRIVATE ${LIB_NAME} ${OpenCV_LIBS})