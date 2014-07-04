
// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************
//#include "reference_calc.cpp"
#include <algorithm>
#include <cassert>
// for uchar4 struct
#include <cuda_runtime.h>
#include "utils.h"
#include <cuda.h>
#include <stdio.h>

__global__ void Kernel_1(const unsigned char* const inputChannel_1, //inputChannel
                   unsigned char* const inputChannel_2,
                   unsigned char* const outputChannel,
                   float* const GPU_gw_average_color_1,
                   float* const GPU_gw_average_color_2,
                   float* const GPU_gw_auto_correlation_1,
                   float* const GPU_gw_auto_correlation_2,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth,int mid_shared_address)
{
  // TODO
  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
  int x= threadIdx.x+ blockIdx.x*blockDim.x;
  int y= threadIdx.y+ blockIdx.y*blockDim.y;
  int thread_1D=x+y*numCols;
  //printf("the x= %d y= %d  and  threadID %d inputvalue %d \n",x,y,thread_1D,inputChannel[thread_1D] );
  
  extern __shared__ unsigned char temp_1[];
  
  char* pointer= (char*)temp_1; 
  char* temp_2 = (char*)&pointer[mid_shared_address];
  
  //first batch transfer
  //temp_1[(threadIdx.x+filterWidth/2)+(threadIdx.y+filterWidth/2)*(blockDim.x+2*(filterWidth/2))]=inputChannel[thread_1D];
  int dest = threadIdx.y * blockDim.x+ threadIdx.x,
	 destY = dest/(blockDim.x+2*(filterWidth/2)),
	 destX = dest % (blockDim.x+2*(filterWidth/2)),
	 srcY = blockIdx.y * blockDim.y + destY - (filterWidth/2),
	 srcX = blockIdx.x * blockDim.x + destX - (filterWidth/2),
	 src = srcY * numCols + srcX;
 
  //printf("threadIdx.x= %d threadIdx.y= %d blockIdx.x= %d blockIdx.y= %d \n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y);
  //printf("dest= %d destY= %d destX= %d srcY= %d srcX= %d src %d \n",dest,destY,destX,srcY,srcX,src );
  //int help_1 = destY*(blockDim.x+2*(filterWidth/2))+destX;
  if (srcY >= 0 && srcY < numRows && srcX >= 0 && srcX < numCols){
      temp_1[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_1[src];
      temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_2[src];
      //printf("temp_1[help_1=%d]=%d \n",help_1,temp_1[help_1]);
  }
  else{
      //temp_1[destY*(blockDim.x+2*(filterWidth/2))+destX] =0; //;
      //printf("temp_1[help_1=%d]=%d \n",help_1,temp_1[help_1]);
      if (srcY < 0){
	  srcY = 0;
      }
      if(srcY >= numRows){
	  srcY =numRows-1 ;
      }
      if (srcX < 0){
	  srcX = 0;
      }
      if(srcX >= numCols){
	  srcX =numCols-1 ;
      }
      int newindex = srcY * numCols + srcX;
      temp_1[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_1[newindex];
      temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_2[newindex];
  }

//Second batch loading
  dest = threadIdx.y * blockDim.x + threadIdx.x + blockDim.x * blockDim.y;
  destY = dest / (blockDim.x+2*(filterWidth/2));
  destX = dest % (blockDim.x+2*(filterWidth/2));
  srcY = blockIdx.y * blockDim.y+ destY - (filterWidth/2);
  srcX = blockIdx.x * blockDim.x+ destX - (filterWidth/2);  
  src =  srcY * numCols + srcX;
  //printf("threadIdx.x= %d threadIdx.y= %d blockIdx.x= %d blockIdx.y= %d \n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y);
  //printf("dest= %d destY= %d destX= %d srcY= %d srcX= %d src %d \n",dest,destY,destX,srcY,srcX,src );
  if (destY < (blockDim.y+2*(filterWidth/2))) {
	//int help_2 = destY*(blockDim.x+2*(filterWidth/2))+destX;
	if (srcY >= 0 && srcY < numRows && srcX >= 0 && srcX < numCols){
	    temp_1[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_1[src];
	    temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_2[src];
	    //printf("temp_1[help_2= %d]= %d \n",help_2,temp_1[help_2]);
	}
	else{
	  if (srcY < 0){
	      srcY = 0;  
	  }
	  if(srcY >= numRows){
	      srcY =numRows-1 ;
	  }
	  if (srcX < 0){
	      srcX = 0;  
	  }
	  if(srcX >= numCols){
	      srcX =numCols-1 ;
	  }
	  int newindex2 = srcY * numCols + srcX;
	  temp_1[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_1[newindex2];
	  temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_2[newindex2];
	    //printf("temp_1[help_2= %d]= %d \n",help_2,temp_1[help_2]);    
	}
  }
  __syncthreads();
  
 // int temp1_size=(blockDim.x+(2*(filterWidth/2)))*(blockDim.y+(2*(filterWidth/2)))*sizeof(char);

  float result_1=0.0f;
  float result_2=0.0f;
  //#pragma unroll 16
  for( int filter_r=-filterWidth/2;filter_r<=filterWidth/2;filter_r++){
    //printf("hello babak %d \n",filter_r+filterWidth/2);
    for( int filter_c=-filterWidth/2;filter_c<=filterWidth/2;filter_c++){
    
      float image_value_1=static_cast<float> (temp_1[(threadIdx.y+filter_r+filterWidth/2)*(blockDim.x+2*(filterWidth/2))+threadIdx.x+filter_c+filterWidth/2]);  
      float image_value_2=static_cast<float> (temp_2[(threadIdx.y+filter_r+filterWidth/2)*(blockDim.x+2*(filterWidth/2))+threadIdx.x+filter_c+filterWidth/2]);  
      float filter_value=filter[(filter_r+filterWidth/2)*filterWidth+filter_c+filterWidth/2];
      result_1+= image_value_1*filter_value;
      result_2+= image_value_2*filter_value;
    }
  }

  //__syncthreads();
  if (y < numRows && x < numCols ){
      GPU_gw_average_color_1[thread_1D]=result_1;
      GPU_gw_average_color_2[thread_1D]=result_2;
  }
  __syncthreads();
  
 
// Computing the weighted auto correlation (alpha)   
  //result_1=0.0f;
  //result_2=0.0f;
  //#pragma unroll 16
  for( int filter_r=-filterWidth/2;filter_r<=filterWidth/2;filter_r++){
    //printf("hello babak %d \n",filter_r+filterWidth/2);
    for( int filter_c=-filterWidth/2;filter_c<=filterWidth/2;filter_c++){
    
      float image_value_1=static_cast<float> (temp_1[(threadIdx.y+filter_r+filterWidth/2)*(blockDim.x+2*(filterWidth/2))+threadIdx.x+filter_c+filterWidth/2]);
      float image_value_2=static_cast<float> (temp_2[(threadIdx.y+filter_r+filterWidth/2)*(blockDim.x+2*(filterWidth/2))+threadIdx.x+filter_c+filterWidth/2]);  
      float diff_Pow_2_1=(image_value_1-result_1)*(image_value_1-result_1);
      float diff_Pow_2_2=(image_value_2-result_2)*(image_value_2-result_2);
      float filter_value_1=filter[(filter_r+filterWidth/2)*filterWidth+filter_c+filterWidth/2];
      result_1+= filter_value_1*diff_Pow_2_1;
      result_2+= filter_value_1*diff_Pow_2_2;
    }
  }

  //__syncthreads();
  if (y < numRows && x < numCols ){
      GPU_gw_auto_correlation_1[thread_1D]=result_1;
      GPU_gw_auto_correlation_2[thread_1D]=result_2;
      //outputChannel[thread_1D]=GPU_gw_auto_correlation_1[thread_1D];
  }
  //outputChannel[thread_1D]=GPU_gw_auto_correlation_1[thread_1D];
  __syncthreads();
  
  
  
  /*
  if ( x >= numCols || y >= numRows ){
      return;
  }
  else
  {
      outputChannel[thread_1D]=result_1;
      
  }*/
  //__syncthreads();
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
}
//------------------------------

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__ void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  // TODO
  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  int absolute_image_position_x= threadIdx.x+blockIdx.x*blockDim.x;
  int absolute_image_position_y= threadIdx.y+blockIdx.y*blockDim.y;
  if ( absolute_image_position_x >= numCols ||
        absolute_image_position_y >= numRows )
  {
    return;
  }
  int i= numCols*absolute_image_position_y+absolute_image_position_x;
  redChannel[i]=inputImageRGBA[i].x;
  greenChannel[i]=inputImageRGBA[i].y;
  blueChannel[i]=inputImageRGBA[i].z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__ void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}
/**
unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  checkCudaErrors(cudaMalloc(&d_filter,sizeof(float)*(int)filterWidth*(int)filterWidth));
    

  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  cudaMemcpy(d_filter,h_filter,sizeof(float)*(int)filterWidth*(int)filterWidth,cudaMemcpyHostToDevice);

}*/

//unsigned char *d_red,*d_green,*d_blue;
//unsigned char *d_Gray_1,*d_Gray_2;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const int numRows, const int numCols,
                                const float* const h_filter, const int filterWidth)
{

  //allocate memory for the three different channels
  //original

  
  //checkCudaErrors(cudaMalloc(&d_Gray_1,   sizeof(unsigned char) * numRows * numCols));
  //checkCudaErrors(cudaMalloc(&d_Gray_2,   sizeof(unsigned char) * numRows * numCols));

  //TODO:
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  checkCudaErrors(cudaMalloc(&d_filter,sizeof(float)*filterWidth*filterWidth));
    

  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  cudaMemcpy(d_filter,h_filter,sizeof(float)*filterWidth*filterWidth,cudaMemcpyHostToDevice);

}

                
                        
void your_gaussian_blur(const unsigned char* const h_inputImageGray_1,
			unsigned char* const d_inputImageGray_1,
			unsigned char* const h_inputImageGray_2,
			unsigned char* const d_inputImageGray_2,
                        unsigned char* const d_outputImageGray,
                        float* const GPU_gw_average_color_1,
			float* const GPU_gw_auto_correlation_1,
			float* const GPU_gw_average_color_2,
			float* const GPU_gw_auto_correlation_2,
		        float* const GPU_gw_cross_correlation_3,
			float* const GPU_gw_normalized_score_4,
                        int* match_matrix,
			int* disparity_map,
			int* depth_map,
                        const int numRows, const int numCols,
                        const int filterWidth)                      
{
  //TODO: Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(32,32);
  //const dim3 blockSize(8,8);
  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  //const dim3 gridSize(15,12);
  const dim3 gridSize(ceilf(static_cast<float>(numCols) / blockSize.x),ceilf(static_cast<float>(numRows) / blockSize.y));

  //TODO: Launch a kernel for separating the RGBA image into different color channels
  //separateChannels<<<gridSize,blockSize>>>(d_inputImageRGBA,(int)numRows,(int)numCols,d_red, d_green,d_blue);

  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  int mid_shared_address=(blockSize.x+(2*(filterWidth/2)))*(blockSize.y+(2*(filterWidth/2)))*sizeof(char);
  Kernel_1<<<gridSize,blockSize,(blockSize.x+(2*(filterWidth/2)))*(blockSize.y+(2*(filterWidth/2)))*sizeof(char)*2>>>(d_inputImageGray_1,
				      d_inputImageGray_2,
				      d_outputImageGray,
				      GPU_gw_average_color_1,
				      GPU_gw_average_color_2,
				      GPU_gw_auto_correlation_1,
				      GPU_gw_auto_correlation_2,
				      (int)numRows,
				      (int)numCols,
				      d_filter,
				      filterWidth,mid_shared_address);
				      // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  //for (int r = 0; r < (int)numRows; ++r) {
  //    for (int c = 0; c < (int)numCols; ++c) {
  //	   printf("%f ",GPU_gw_auto_correlation_1[r * numCols + c]);
  //    }
  //    printf("\n");
  //}
/*  
  Kernel_1<<<gridSize,blockSize,(blockSize.x+(2*(filterWidth/2)))*(blockSize.y+(2*(filterWidth/2)))*sizeof(char)>>>(d_green,
				      d_greenBlurred,d_greenBlurred_1,
				      (int)numRows,
				      (int)numCols,
				      d_filter,
				      filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  Kernel_1<<<gridSize,blockSize,(blockSize.x+(2*(filterWidth/2)))*(blockSize.y+(2*(filterWidth/2)))*sizeof(char)>>>(d_blue,
				      d_blueBlurred,d_blueBlurred_1,
				      (int)numRows,
				      (int)numCols,
				      d_filter,
				      filterWidth);
 
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred_1,
                                             d_greenBlurred_1,
                                             d_blueBlurred_1,
                                             d_outputImageRGBA_1,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
 */

}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated

/*
void cleanup() {
  checkCudaErrors(cudaFree(d_inputImageGray_1));//d_red));
  checkCudaErrors(cudaFree(d_inputImageGray_2));//d_green));
  checkCudaErrors(cudaFree(d_outputImageGray));//d_blue));
}
*/