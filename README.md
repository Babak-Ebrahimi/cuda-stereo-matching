# Accurate Real-Time Stereo Matching on the GPU

CUDA implementation of a stereo matching algorithm based on **Weighted Normalized Cross-Correlation (WNCC)** for computing dense disparity/depth maps from a pair of stereo images.

The project was originally developed as a master's project in 2014 and focused on accelerating stereo matching using GPU parallelization and CUDA shared memory. The current version has been updated to build and run on a modern Linux system with a NVIDIA GTX 1080 Ti.

## Overview

Stereo matching estimates depth by finding corresponding pixels between two images captured from different viewpoints. The disparity between corresponding pixels can then be used to estimate scene depth.

The main challenge is achieving a good balance between **matching accuracy and computational efficiency**.

This implementation accelerates the computation using CUDA by:

* Parallelizing stereo matching across GPU threads
* Using CUDA shared memory to reduce global-memory access
* Computing weighted cross-correlation and weighted normalized correlation on the GPU
* Processing image regions using CUDA thread blocks
* Computing the disparity map directly on the GPU

The original project was designed with real-time 3D reconstruction applications in mind.

## Algorithm

The implementation uses **Weighted Normalized Cross-Correlation (WNCC)** as the stereo matching metric.

The CUDA implementation consists primarily of two computational stages:

1. **Kernel 1**

   * Computes the weighted average color
   * Computes the weighted auto-correlation
   * Uses shared memory to efficiently load image blocks

2. **Kernel 2**

   * Computes the weighted cross-correlation
   * Computes the weighted normalized correlation score
   * Searches across candidate disparities

The implementation uses CUDA shared memory with blocks of **32 × 32 threads** to improve memory access efficiency.

The original project documentation describes the CUDA implementation and optimization strategy in more detail.

## Requirements

* Linux
* NVIDIA GPU with CUDA support
* NVIDIA CUDA Toolkit
* GCC/G++ compatible with the installed CUDA Toolkit
* CMake
* OpenCV 4
* Git

The current project was tested with:

* NVIDIA GeForce GTX 1080 Ti
* NVIDIA driver 580.173.02
* CUDA Toolkit 12.0
* GCC/G++ 12
* CMake 3.28
* OpenCV 4.6

Other CUDA-capable NVIDIA GPUs may also work, but performance will depend on the GPU architecture.

## Build

Clone the repository:

```bash
git clone https://github.com/Babak-Ebrahimi/...
cd nwcc7
```

Create a build directory:

```bash
mkdir build
cd build
```

Configure the project:

```bash
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 cmake ..
```

Build:

```bash
make -j$(nproc)
```

The executable will be created as:

```text
build/nwcc7
```

## Run

Two stereo images are included in the `data/` directory:

```text
data/im2.png
data/im6.png
```

Run the program from the build directory:

```bash
./nwcc7 ../data/im2.png ../data/im6.png
```

A successful GPU execution prints the execution time, for example:

```text
Your code ran in: 36.967422 msecs.
```

## Performance

### Current system

The reorganized version was tested on a:

**NVIDIA GeForce GTX 1080 Ti**

The GPU implementation completed the tested stereo matching workload in approximately:

**36.97 ms**

This corresponds to roughly:

**27 frames/second**

for the measured computational workload, although end-to-end application frame rate will depend on image loading, preprocessing, output handling, and other application components.

### Original project results

The original 2014 project was evaluated on a **GeForce GTX 770**.

The original presentation reported:

| Implementation | Runtime |
| -------------- | ------: |
| CPU            |  30.1 s |
| GPU            | 48.9 ms |

The original GPU implementation therefore achieved approximately a **600× speedup** over the CPU implementation.

These historical results should not be directly compared with the current GTX 1080 Ti measurement because they were obtained on different hardware and under the original experimental setup.

## GPU Optimization

A major focus of the implementation is reducing expensive global-memory accesses.

CUDA shared memory is used to load image regions into fast on-chip memory. Threads within a CUDA block can then reuse these values while evaluating multiple candidate matching windows.

The original implementation uses **32 × 32 thread blocks** and performs the stereo matching computation through two CUDA kernels.

Profiling from the original project showed that the second CUDA kernel dominated execution time, accounting for approximately **98.8%** of the GPU computation time.

## Project Structure

```text
nwcc7/
├── CMakeLists.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── main.cpp
│   ├── student_func.cu
│   ├── timer.h
│   └── utils.h
│
├── data/
│   ├── im2.png
│   └── im6.png
│
└── docs/
    └── presentation.pdf
```

## Original Documentation

The original project presentation is included in:

```text
docs/presentation.pdf
```

It contains the original motivation, algorithm description, CUDA implementation details, optimization discussion, profiling results, and conclusions.

## Limitations and Future Work

The original evaluation noted that although the GPU implementation provided substantial computational speedup, the resulting depth-map quality could still be improved compared with state-of-the-art stereo benchmarks.

Potential future improvements include:

* Improving stereo matching accuracy
* Evaluating additional stereo datasets
* Supporting additional camera views
* Processing synchronized video streams
* Incorporating temporal information between consecutive frames
* Further CUDA kernel optimization

## References

The implementation and original project were developed as part of:

**Accurate Real-Time Stereo Matching on the GPU for 3D Reconstruction**

Master Project, Summer Semester 2014
Babak Ebrahimi
Supervisors: Dr. Norbert Schmitz and Dipl. Vladislav Golyanik

See [`docs/presentation.pdf`](docs/presentation.pdf) for the original project documentation.
