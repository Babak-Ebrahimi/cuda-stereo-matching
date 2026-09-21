

# collect source files

cmake_minimum_required(VERSION 2.8)
project(nwcc7)

find_package(CUDA)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

set(CUDA_NVCC_FLAGS "-arch=sm_61 -allow-unsupported-compiler")


file( GLOB  hdr src/*.hpp src/*.h )
file( GLOB  cu  src/*.cu)
SET (nwcc7_files src/main.cpp)
    
CUDA_ADD_EXECUTABLE(nwcc7 ${nwcc7_files} ${hdr} ${cu})

#add_executable(nwcc7 main.cpp)

target_link_libraries(nwcc7 ${OPENCV_LIBRARIES})

