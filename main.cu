/************************************************************************
   Program:         main.cu
   Author:          Joey Brown
   Class:           CSC461 Programming Languages
   Instructor:      John M. Weiss, Ph.D.
   Date:            10 - 16 - 2017
   Description:     Demonstrate and compare the speed improvement of edge
                        detection using sequential, OpenMP, and CUDA 
                        approaches. 
   Compilation 
   instructions:    nvcc -o edge lodepng.cpp main.cu -std=c++11 -Xcompiler -fopenmp

   Usage:           edge image.png
   Notes:           Many examples were taken from vecadd.cu and grayscale.cpp
                        written by Dr. Weiss.
 ************************************************************************/

#include <iostream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include "lodepng.h"
#include <omp.h>
using namespace std;

/*********************** global type definitions ***********************/
typedef unsigned char byte;

/*********************** function prototypes ***************************/

/************************************************************************
   Function:    getGx()
   Author:      Joey Brown
   Description: returns the Gx at pixel[i] of the Sobel operator
   Parameters:  image array, index to pixel, and the width of the image.
 ************************************************************************/
int getGx(byte* image, int i, int w)
{
	return (
		(image[ i - w - 1 ] * (-1)) +
		(image[ i - w 	  ] * (-2)) +
		(image[ i - w + 1 ] * (-1)) +

		(image[ i + w - 1] * 1)		+
		(image[ i + w ] * 2 ) 		+
		(image[ i + w + 1 ] * 1)
	);
}

/************************************************************************
   Function:    getGy()
   Author:      Joey Brown
   Description: returns the Gy at pixel[i] of the Sobel operator
   Parameters:  image array, index to pixel, and the width of the image.
 ************************************************************************/
int getGy(byte* image, int i, int w)
{
	return (
		(image[ i - w - 1 ] * (-1)) +
		(image[ i - 1 	  ] * (-2)) +
		(image[ i + w - 1 ] * (-1)) +

		(image[ i - w + 1] )	+
		(image[ i + 1 ] *   2 ) 	+
		(image [ i + w  + 1 ])
	);
}

/************************************************************************
   Function:    getEdgeMagnitude()
   Author:      Joey Brown
   Description: calculates the edge magnitude given Gx and Gy
   Parameters:  Gx and Gy of the Sobel operator
 ************************************************************************/
byte getEdgeMagnitude(int Gx, int Gy) { return ( sqrt( pow( Gx,2 ) + pow( Gy,2 ) ) ); }

/************************************************************************
   Function:    getEdgeMagnitude()
   Author:      Joey Brown
   Description: another version to calculate the edge magnitude
   Parameters:  image, indexed pixel, and width of the image
 ************************************************************************/
byte getEdgeMagnitude(byte* image, int i, int w){ return ( sqrt( pow( getGx(image, i, w),2 ) + pow( getGy(image, i, w),2 ) ) ); }

/************************************************************************
   Function:    sobel_seq()
   Author:      Joey Brown
   Description: Sequential edge detection using the Sobel operator
                    (image passed in should be grayscale)
   Parameters:  image to be processed, width and height of the image.
 ************************************************************************/
byte* sobel_seq(byte* image, unsigned int w, unsigned int h)
{
	int size = w*h;
	byte* newImage = new byte[ size ];
	int index = 0;
	for (int r = 1; r < h-1; ++r)
	{
		for (int c = 1; c < w-1; ++c)
		{
			index = (r*w)+c;
			// newImage[index] = getEdgeMagnitude(getGx(image,index, w), getGy(image, index, w));
			newImage[index] = getEdgeMagnitude(image, index, w);
			// newImage[index] = sqrt(getGx)
		}
	}
	return newImage;
}

/************************************************************************
   Function:    sobel_omp()
   Author:      Joey Brown
   Description: OpenMP implementation of edge detection that delegates the
                    loops so they don't have to be processed sequentially.
   Parameters:  image to be processed, width and height of the image.
 ************************************************************************/
byte* sobel_omp(byte* image, unsigned int w, unsigned int h)
{
	int size = w*h;
	byte* newImage = new byte[ size ];
	int index = 0;
	#pragma omp parallel for collapse(2)
	for (int r = 1; r < h-1; ++r)
	{
		for (int c = 1; c < w-1; ++c)
		{
			index = (r*w)+c;
			newImage[index] = getEdgeMagnitude(image, index, w);
		}
	}
	return newImage;
}

// CUDA kernel: Sobel edge magnitude
/************************************************************************
   Function:    sobel_cuda()
   Author:      Joey Brown
   Description: CUDA kernel implementation of Sobel edge magnitude detection.
   Parameters:  target grayscale image, image width & height, reference to 
                    the resulting image with edge detection
 ************************************************************************/
__global__ void sobel_cuda(byte* image, unsigned int w, unsigned int h, byte* img_ret)
{
	int i = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	if( i < h*(w-1)+1 && i > (w)+1 )
	{
		double Gx = (image[ i - w - 1 ] * (-1)) +
		(image[ i - w 	  ] * (-2)) +
		(image[ i - w + 1 ] * (-1)) +

		(image[ i + w - 1] * 1)		+
		(image[ i + w ] * 2 ) 		+
		(image[ i + w + 1 ] * 1);

		double Gy = (image[ i - w - 1 ] * (-1)) +
		(image[ i - 1 	  ] * (-2)) +
		(image[ i + w - 1 ] * (-1)) +

		(image[ i - w + 1] )	+
		(image[ i + 1 ] *   2 ) 	+
		(image [ i + w  + 1 ]);
		

		img_ret[i] = sqrt((Gx*Gx)+(Gy*Gy));
	}
}

/****************************MAIN****************************************/
int main(int argc, char const *argv[])
{

	// check usage
    if ( argc < 2 )
    {
        printf( "Usage: %s infile.png\n", argv[0] );
        return -1;
    }

    // CUDA device properties
	cudaDeviceProp devProp;
	cudaGetDeviceProperties( &devProp, 0 );
	int cores = devProp.multiProcessorCount;
    switch ( devProp.major )
    {
        case 2: // Fermi
            if ( devProp.minor == 1 ) cores *= 48;
            else cores *= 32; break;
        case 3: // Kepler
            cores *= 192; break;
        case 5: // Maxwell
            cores *= 128; break;
        case 6: // Pascal
            if ( devProp.minor == 1 ) cores *= 128;
            else if ( devProp.minor == 0 ) cores *= 64;
            break;
    }

    // Display header message
    time_t currTime = time( 0 );
    printf( "edge map benchmarks (%s)", ctime( &currTime ) );
    printf( "CPU: %d hardware threads\n", thread::hardware_concurrency() );
    printf( "GPGPU: %s, CUDA %d.%d, %d Mbytes global memory, %d CUDA cores\n",
            devProp.name, devProp.major, devProp.minor, devProp.totalGlobalMem / 1048576, cores );


    // read input PNG file
    byte* pixels;
    unsigned int width, height;
    unsigned error = lodepng_decode_file( &pixels, &width, &height, argv[1], LCT_RGBA, 8 );
    if ( error )
    {
        printf( "decoder error while reading file %s\n", argv[1] );
        printf( "error code %u: %s\n", error, lodepng_error_text( error ) );
        return -2;
    }
    printf( "\n%s: %d rows x %d columns\n", argv[1], height, width );

    // copy 24-bit RGB data into 8-bit grayscale intensity array
    int npixels = width * height;
    byte* image = new byte [ npixels ];
    byte* img = pixels;
    for ( int i = 0; i < npixels; ++i )
    {   
        int r = *img++;
        int g = *img++;
        int b = *img++;
        int a = *img++;     // alpha channel is not used
        image[i] = 0.3 * r + 0.6 * g + 0.1 * b + 0.5;
    }
    free( pixels );     // LodePNG uses malloc, not new

    //  Reserve memory for the grayscale images
    int size = width*height;
    byte* img_CPU = new byte[size];
    byte* img_OMP = new byte[size];

    // Write grayscale PNG file
    error =  lodepng_encode_file( "gray.png", image, width, height, LCT_GREY, 8 );
    if ( error )
    {
        printf( "encoder error while writing file %s\n", "gray.png" );
        printf( "error code %u: %s\n", error, lodepng_error_text( error ) );
        return -3;
    }

    // Write sequential CPU Sobel processing PNG file
    auto c = chrono::system_clock::now();
    img_CPU = sobel_seq(image, width, height);
    chrono::duration<double> d_cpu = chrono::system_clock::now() - c;
    error = lodepng_encode_file( "image_cpu.png", img_CPU, width, height, LCT_GREY, 8 );
    if ( error )
    {
        printf( "encoder error while writing file %s\n", "gray.png" );
        printf( "error code %u: %s\n", error, lodepng_error_text( error ) );
        return -3;
    }

    // Write OpenMP Sobel processing PNG file
    c = chrono::system_clock::now();
    img_OMP = sobel_omp(image, width, height);
    chrono::duration<double> d_omp = chrono::system_clock::now() - c;
    error =  lodepng_encode_file( "image_omp.png", img_OMP, width, height, LCT_GREY, 8 );
    if ( error )
    {
        printf( "encoder error while writing file %s\n", "gray.png" );
        printf( "error code %u: %s\n", error, lodepng_error_text( error ) );
        return -3;
    }

    // Write CUDA Sobel processing PNG file
    // allocate host memory 
    int imgSize = size * sizeof(byte);
    byte* img_GPU = ( byte * )malloc( imgSize );
    
    // allocate device memory 
    byte* tmp_GPU = ( byte * )malloc( imgSize );
    byte* tmp_image = ( byte * )malloc( imgSize );
    cudaMalloc( ( void ** )&tmp_GPU, imgSize);
    cudaMalloc( ( void ** )&tmp_image, imgSize);

    // copy image arrays to device
    cudaMemcpy( tmp_image, image, imgSize, cudaMemcpyHostToDevice );
    cudaMemcpy( tmp_GPU, img_GPU, imgSize, cudaMemcpyHostToDevice );

    // launch add() kernel on GPU with M threads per block, (N+M-1)/M blocks
    int nThreads = 512;
    int nBlocks = ( size + nThreads - 1 ) / nThreads;
    c = chrono::system_clock::now();
    sobel_cuda<<< nBlocks, nThreads >>>(tmp_image, width, height, tmp_GPU);
    cudaError_t cudaerror = cudaDeviceSynchronize();            // waits for completion, returns error code
    if ( cudaerror != cudaSuccess ) 
    	fprintf( stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName( cudaerror ) );
    chrono::duration<double> d_gpu = chrono::system_clock::now() - c;

    // copy device processing back to host
    cudaMemcpy( img_GPU, tmp_GPU, imgSize, cudaMemcpyDeviceToHost );
    error =  lodepng_encode_file( "image_gpu.png", img_GPU, width, height, LCT_GREY, 8 );
    if ( error )
    {
        printf( "encoder error while writing file %s\n", "gray.png" );
        printf( "error code %u: %s\n", error, lodepng_error_text( error ) );
        return -3;
    }

    // Display benchmark results
    cout << "\nBenchmarks: \n";
    cout << "CPU: " << 1000 * d_cpu.count() << " msec\n";
    cout << "OpenMP: " << 1000 * d_omp.count() << " msec\n";
    cout << "CUDA: " << 1000 *d_gpu.count() <<" msec\n";

    // Display speedups
    cout << "\nSpeedups:\n";
    cout << "CPU -> OMP: " << (d_cpu.count()/d_omp.count()) << " X\n";
    cout << "OMP -> CUDA: " << (d_omp.count()/d_gpu.count()) << " X\n";
    cout << "CPU -> CUDA: " << (d_cpu.count()/d_gpu.count()) << " X\n";
    cout << endl;

    // Collect Garbage
    cudaFree(tmp_image);
    cudaFree(tmp_GPU);

    delete [] image;
    delete [] img_CPU;
    delete [] img_OMP;
    delete [] img_GPU;
    return 0;
}


