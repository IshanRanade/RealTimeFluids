cmake_minimum_required(VERSION 3.12)
project(fluidsim CXX CUDA)

set (CMAKE_CXX_STANDARD 11)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -dc arch=sm_60)

list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -g)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

# Include all the folders that contain code
# include_directories(include)
# include_directories(src)
# include_directories(src/io)
# include_directories(src/ui)
# include_directories(src/pathtracing)
# include_directories(src/pathtracing/raytracing)
# include_directories(src/pathtracing/scene)
# include_directories(src/pathtracing/scene/geometry)

find_package(OpenGL)

set(CPP_SOURCES

)

set(CUDA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu
)

add_executable(fluidsim
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)