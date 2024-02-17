set(LIB_NAME ImgColorizer)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SRC
./Colorizer.cpp
)

set(HDRS
${PROJECT_SOURCE_DIR}/include/Colorizer.h
)

add_library(${LIB_NAME} STATIC ${SRC} ${HDRS})
target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LIB_NAME} PRIVATE ${OpenCV_LIBS})

# add_executable(${LIB_NAME}_test ./main.cpp)
# target_link_libraries(${LIB_NAME}_test PRIVATE ${LIB_NAME})