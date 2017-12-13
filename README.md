Parallel programming with OpenMP and CUDA.
Programming Assignment 2
Author: Joey Brown
Date: 10/16/2017

Description:
Write a program to generate an edge map by computing the edge magnitude at each pixel in a PNG image. Your
program should implement the Sobel edge detector in three ways: sequentially on the CPU, concurrently with
OpenMP on a multi-core CPU, and in parallel using GPGPU with CUDA on an Nvidia graphics card. Time
how long it takes to generate the edge maps. Output the timing results, along with the speedups obtained from
parallelization. Also output the three edge maps as PNG images.

Compilation:
$> nvcc -o edge -std=c++11 -fopenmp -Xcompile lodepng.cpp main.cu

Usage:
$> edge image.png