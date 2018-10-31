cmake_minimum_required(VERSION 3.1)

project(cis565_rasterizer)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Set up include and lib paths
include_directories(.)
include_directories("external")
include_directories("external/include")
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(EXTERNAL_LIB_PATH "external/lib/osx")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(EXTERNAL_LIB_PATH "external/lib/linux" "/usr/lib64")
elseif(WIN32)
    set(EXTERNAL_LIB_PATH "external/lib/win")
endif()
link_directories(${EXTERNAL_LIB_PATH})
list(APPEND CMAKE_LIBRARY_PATH "${EXTERNAL_LIB_PATH}")


# Find up and set up core dependency libs

set(GLFW_INCLUDE_DIR "external/include")
set(GLFW_LIBRARY_DIR "${EXTERNAL_LIB_PATH}")
find_library(GLFW_LIBRARY "glfw3" HINTS "${GLFW_LIBRARY_DIR}")

set(GLEW_INCLUDE_DIR "external/include")
set(GLEW_LIBRARY_DIR "${EXTERNAL_LIB_PATH}")
add_definitions(-DGLEW_STATIC)
find_package(GLEW)

find_package(OpenGL)

set(CORELIBS
    "${GLFW_LIBRARY}"
    "${OPENGL_LIBRARY}"
    "${GLEW_LIBRARY}"
    )

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)

# Set up different build configurations
set(CMAKE_CONFIGURATION_TYPES Debug;RelWithDebInfo;Release
    CACHE STRING "Set configuration types" FORCE)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_FLAGS_DEBUG                 "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO        "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE               "-O3    -DNDEBUG")
list(APPEND CUDA_NVCC_FLAGS_DEBUG          -O0 -g -G)
list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBINFO -O2 -g -lineinfo)
list(APPEND CUDA_NVCC_FLAGS_RELEASE        -O3    -DNDEBUG)
if (WIN32)
    set(CUDA_PROPAGATE_HOST_FLAGS ON)
    set(CMAKE_CXX_FLAGS                       "/MD /EHsc /D _CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS_DEBUG                 "/Od /Zi")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO        "/O2 /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE               "/O2 /D NDEBUG")
endif()

# OS X options
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    list(APPEND CORELIBS "-framework IOKit")
    list(APPEND CORELIBS "-framework Cocoa")
    list(APPEND CORELIBS "-framework CoreVideo")
endif()

# Linux options
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    list(APPEND CMAKE_EXE_LINKER_FLAGS "-lX11 -lXxf86vm -lXrandr -lXi")
endif()

if (WIN32)
    list(APPEND CORELIBS legacy_stdio_definitions.lib)
endif()

# CUDA linker options
find_package(Threads REQUIRED)
find_package(CUDA 8.0 REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

#add_subdirectory(stream_compaction)  # TODO: uncomment if using your own stream compaction
add_subdirectory(src)
add_subdirectory(util)

cuda_add_executable(${CMAKE_PROJECT_NAME}
    "src/main.hpp"
    "src/main.cpp"
    )

target_link_libraries(${CMAKE_PROJECT_NAME}
    src
    util
    #stream_compaction  # TODO: uncomment if using your own stream compaction
    ${CORELIBS}
    )

add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
    )
