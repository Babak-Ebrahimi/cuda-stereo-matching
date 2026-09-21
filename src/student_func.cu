// code written by babak ebrahimi summer 2014
//#include "reference_calc.cpp"
#include <algorithm>
#include <cassert>
// for uchar4 struct
#include <cuda_runtime.h>
#include "utils.h"
#include <cuda.h>
#include <stdio.h>
#include <math.h>  
#include <float.h>

__global__ void Kernel_1(const unsigned char* const inputChannel_1, //inputChannel
                   const unsigned char* const inputChannel_2,
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
 
  unsigned char* pointer= (unsigned char*)temp_1; 
  unsigned char* temp_2 = (unsigned char*)&pointer[mid_shared_address];
  
  //first batch transfer
  //temp_1[(threadIdx.x+filterWidth/2)+(threadIdx.y+filterWidth/2)*(blockDim.x+2*(filterWidth/2))]=inputChannel[thread_1D];
  int dest = threadIdx.y * blockDim.x+ threadIdx.x,// index(in a one dimentional array) of a thread(pixel) in a block 
	 destY = dest/(blockDim.x+2*(filterWidth/2)),// y index of destination in shared memory
	 destX = dest % (blockDim.x+2*(filterWidth/2)),// x index of destination in shared memory
	 srcY = blockIdx.y * blockDim.y + destY - (filterWidth/2),// y index in source image (-filterwidth/2 because of overlaping)
	 srcX = blockIdx.x * blockDim.x + destX - (filterWidth/2),// x index in source image (-filterwidth/2 because of overlaping)
	 src = srcY * numCols + srcX;// index(in a one dimentional array) of a thread(pixel) in the source image
   
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
      //printf("destY*(blockDim.x+2*(filterWidth/2))+destX= %d \n",destY*(blockDim.x+2*(filterWidth/2))+destX);
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
	  //printf("temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX= %d]= %d , %d \n",destY*(blockDim.x+2*(filterWidth/2))+destX,(temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX]), inputChannel_2[newindex2] );    
	  
	  //printf("temp_1[destY*(blockDim.x+2*(filterWidth/2))+destX= %d]= %d , %d \n",destY*(blockDim.x+2*(filterWidth/2))+destX,(temp_1[destY*(blockDim.x+2*(filterWidth/2))+destX]), inputChannel_1[newindex2] );    
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
  result_1=result_1/(filterWidth*filterWidth);
  result_2=result_2/(filterWidth*filterWidth);
  //__syncthreads();
  if (y < numRows &&  x < numCols ){
      GPU_gw_average_color_1[thread_1D]=result_1;
      GPU_gw_average_color_2[thread_1D]=result_2;
  }
  __syncthreads();
   //printf("GPU_gw_average_color_1_2[thread_1D= %d]= %f , %f \n",thread_1D,GPU_gw_average_color_1[thread_1D],GPU_gw_average_color_2[thread_1D]); 
 
// Computing the weighted auto correlation (alpha)   
  result_1=0.0f;
  result_2=0.0f;
  //#pragma unroll 16
  
  if (y < numRows && x < numCols ){
  
    for( int filter_r=-filterWidth/2;filter_r<=filterWidth/2;filter_r++){
      //printf("hello babak %d \n",filter_r+filterWidth/2);
      for( int filter_c=-filterWidth/2;filter_c<=filterWidth/2;filter_c++){
    
	float image_value_1=static_cast<float> (temp_1[(threadIdx.y+filter_r+filterWidth/2)*(blockDim.x+2*(filterWidth/2))+threadIdx.x+filter_c+filterWidth/2]);
	float image_value_2=static_cast<float> (temp_2[(threadIdx.y+filter_r+filterWidth/2)*(blockDim.x+2*(filterWidth/2))+threadIdx.x+filter_c+filterWidth/2]);  
	//float diff_Pow_2_1=(image_value_1-result_1)*(image_value_1-result_1);
	float diff_Pow_2_1=(image_value_1-GPU_gw_average_color_1[thread_1D])*(image_value_1-GPU_gw_average_color_1[thread_1D]);
	//float diff_Pow_2_2=(image_value_2-result_2)*(image_value_2-result_2);
	float diff_Pow_2_2=(image_value_2-GPU_gw_average_color_2[thread_1D])*(image_value_2-GPU_gw_average_color_2[thread_1D]);
	float filter_value_1=filter[(filter_r+filterWidth/2)*filterWidth+filter_c+filterWidth/2];
	result_1+= filter_value_1*diff_Pow_2_1;
	result_2+= filter_value_1*diff_Pow_2_2;
      }
    }
    result_1=result_1/(filterWidth*filterWidth);
    result_2=result_2/(filterWidth*filterWidth);
  __syncthreads();
  
      
    GPU_gw_auto_correlation_1[thread_1D]=result_1;
    GPU_gw_auto_correlation_2[thread_1D]=result_2;
    //if (x==200){
	  //printf("GPU_gw_auto_correlation_2[%d][%d]=%f\n",y,x,GPU_gw_auto_correlation_2[thread_1D]);
      //}
      //outputChannel[thread_1D]=GPU_gw_auto_correlation_1[thread_1D];
  }
  //outputChannel[thread_1D]=GPU_gw_auto_correlation_1[thread_1D];
  __syncthreads();
  //printf("GPU_gw_auto_correlation_1_2[thread_1D= %d]= %f , %f \n",thread_1D,GPU_gw_auto_correlation_1[thread_1D],GPU_gw_auto_correlation_2[thread_1D]);      

  
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
//-----------------------------------
__global__ void Kernel_2(const unsigned char* const inputChannel_1, //inputChannel
                   unsigned char* const inputChannel_2,
                   unsigned char* const outputChannel,
                   float* const GPU_gw_average_color_1,
                   float* const GPU_gw_average_color_2,
                   float* const GPU_gw_auto_correlation_1,
                   float* const GPU_gw_auto_correlation_2,
                   float* const GPU_gw_cross_correlation_3,
                   float* const GPU_gw_normalized_score_4,
                   int* match_matrix,
		   int* disparity_map,
		   int* depth_map,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth,const int mid_shared_address)
{
  // TODO
  
  // NOTE: Be sure to compute any intermediate results in floating point
  // before storing the final result as unsigned char.

  // NOTE: Be careful not to try to access memory that is outside the bounds of
  // the image. You'll want code that performs the following check before accessing
  // GPU memory:
  //
  // if ( absolute_image_position_x >= numCols || absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
  int x= threadIdx.x+ blockIdx.x*blockDim.x; // x index of a tread in an image (not block)
  int y= threadIdx.y+ blockIdx.y*blockDim.y; // y index of a tread in an image (not block)
  int thread_1D=x+y*numCols;
  //printf("the x= %d y= %d  and  threadID %d inputvalue %d \n",x,y,thread_1D,inputChannel[thread_1D] );
  
  extern __shared__ unsigned char temp_1[];
 
  unsigned char* pointer= (unsigned char*)temp_1; 
  unsigned char* temp_2 = (unsigned char*)&pointer[mid_shared_address];
  
  //float result_1=0.0f;
  //float result_2=0.0f;
  const int max_disparity=200;
  //int disparity; //this value used for save the number of pixels which should be compared for each pixel
  //disparity=(x>=max_disparity)?max_disparity:x;
  //disparity=((blockIdx.x*blockDim.x)>=max_disparity)?max_disparity:x;//(blockIdx.x*blockDim.x);
  if ( x<numCols  && y< numRows){
    GPU_gw_cross_correlation_3[thread_1D]=-100000.0;//__FLT_MIN__;
    GPU_gw_normalized_score_4[thread_1D]=-100000.0;//__FLT_MIN__;
  }
  //first batch transfer
  //temp_1[(threadIdx.x+filterWidth/2)+(threadIdx.y+filterWidth/2)*(blockDim.x+2*(filterWidth/2))]=inputChannel[thread_1D];
  int dest = threadIdx.y * blockDim.x+ threadIdx.x, // index(in a one dimentional array) of a thread(pixel) in a block 
      destY = dest/(blockDim.x+2*(filterWidth/2)), // y index of destination in shared memory
      destX = dest % (blockDim.x+2*(filterWidth/2)),// x index of destination in shared memory
      srcY = blockIdx.y * blockDim.y + destY - (filterWidth/2), // y index in source image (-filterwidth/2 because of overlaping)
      srcX = blockIdx.x * blockDim.x + destX - (filterWidth/2), //// x index in source image (-filterwidth/2 because of overlaping)
      src = srcY * numCols + srcX;// index(in a one dimentional array) of a thread(pixel) in the source image
  int second_srcX= srcX-max_disparity;
  int second_src= src-max_disparity;// for transfering the second window of cross correlation to shared memory
 
  if (srcY >= 0 && srcY < numRows && srcX >= 0 && srcX < numCols){
      temp_1[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_1[src];
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
  }
  // first batch loading for second window in shared memory
  //if (blockIdx.x==14 && threadIdx.y==3 ){
    //printf("second_src=%d  ",second_src);
  //}
  
  if (srcY >= 0 && srcY < numRows && second_srcX >= 0 && second_srcX < numCols && second_src >=0 && second_src < numCols * numRows  ){
    temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_2[second_src];
    //printf("temp2[%d]=%d\t",destY*(blockDim.x+2*(filterWidth/2))+destX,temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX]);
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
      if (second_srcX < 0){
	  second_srcX = 0;
      }
      if(second_srcX >= numCols){
	  second_srcX =numCols-1 ;
      }
      int newindex = srcY * numCols + second_srcX;
      temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_2[newindex];
      //printf("temp2[%d]=%d\t",destY*(blockDim.x+2*(filterWidth/2))+destX,temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX]);
  }
//Second batch loading
  dest = threadIdx.y * blockDim.x + threadIdx.x + blockDim.x * blockDim.y;
  destY = dest / (blockDim.x+2*(filterWidth/2));
  destX = dest % (blockDim.x+2*(filterWidth/2));
  srcY = blockIdx.y * blockDim.y+ destY - (filterWidth/2);
  srcX = blockIdx.x * blockDim.x+ destX - (filterWidth/2);  
  src =  srcY * numCols + srcX;
  second_srcX= srcX-max_disparity;
  second_src= src-max_disparity;// for transfering second batch of the second window of cross correlation to shared memory
  //printf("threadIdx.x= %d threadIdx.y= %d blockIdx.x= %d blockIdx.y= %d \n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y);
  //printf("dest= %d destY= %d destX= %d srcY= %d srcX= %d src %d \n",dest,destY,destX,srcY,scX,src );
  if (destY < (blockDim.y+2*(filterWidth/2))) {
	//int help_2 = destY*(blockDim.x+2*(filterWidth/2))+destX;
	if (srcY >= 0 && srcY < numRows && srcX >= 0 && srcX < numCols ){
	    temp_1[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_1[src];
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
	}
	// second batch loading for second window in shared memory
	
	if (srcY >= 0 && srcY < numRows && second_srcX >= 0 && second_srcX < numCols  && second_src >=0 && second_src < numCols * numRows){
	    temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_2[second_src];
	    //printf("temp2[%d]=%d\t",destY*(blockDim.x+2*(filterWidth/2))+destX,temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX]);
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
	    if (second_srcX < 0){
	      second_srcX = 0;
	    }
	    if(second_srcX >= numCols){
	      second_srcX =numCols-1 ;
	    }
	    int newindex2 = srcY * numCols + second_srcX;
	    temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_2[newindex2];
	    //printf("temp2[%d]=%d\t",destY*(blockDim.x+2*(filterWidth/2))+destX,temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX]);
	}
  }
  __syncthreads();
  
  
  //until here there was just transfering the first block from main memory to shared memoryhelecoptrei zadi, 
  //after here it is computing the convultion filter and updating the shared memory 
  
  for (int j=0; j<max_disparity;j++){
  
	if (x-max_disparity+j>=0){
	  float result_1= 0.0f;
	  if (y>=0 && y < numRows && x>=0 && x < numCols && (thread_1D-max_disparity+j)>=0 ){
	    for( int filter_r=-filterWidth/2;filter_r<=filterWidth/2;filter_r++){
	      //printf("hello babak %d \n",filter_r+filterWidth/2);
	      for( int filter_c=-filterWidth/2;filter_c<=filterWidth/2;filter_c++){
		float image_value_1=static_cast<float> (temp_1[(threadIdx.y+filter_r+filterWidth/2)*(blockDim.x+2*(filterWidth/2))+threadIdx.x+filter_c+filterWidth/2]);  
		float image_value_2=static_cast<float> (temp_2[(threadIdx.y+filter_r+filterWidth/2)*(blockDim.x+2*(filterWidth/2))+threadIdx.x+filter_c+filterWidth/2]); 
		//if (x==100 && y==100 && j==50){
		  //printf(" hello babak2 x=%d y=%d j=%d  image_value_1=%f image_value_2=%f  \n", x,y,j,image_value_1,image_value_2 );
		//}
		float filter_value=filter[(filter_r+filterWidth/2)*filterWidth+filter_c+filterWidth/2];
		float image_minus_average_1= static_cast<float>(image_value_1)-static_cast<float>(GPU_gw_average_color_1[thread_1D]);
		float image_minus_average_2= static_cast<float>(image_value_2)-static_cast<float>(GPU_gw_average_color_2[thread_1D-max_disparity+j]);
		result_1+= image_minus_average_1*image_minus_average_2*filter_value;
	      }
	    }
	    //__syncthreads();
	    result_1=result_1/(filterWidth*filterWidth);
	    //printf("result_1,GPU_gw_cross_correlation_3[thread_1D=%d]=%f   %f\n",thread_1D,result_1,GPU_gw_cross_correlation_3[thread_1D]);
	    float help_3=(static_cast<float>(result_1)/(static_cast<float>(sqrt(GPU_gw_auto_correlation_1[thread_1D]*GPU_gw_auto_correlation_2[thread_1D-max_disparity+j]))));
	    //__syncthreads();
	    if (help_3>GPU_gw_normalized_score_4[thread_1D] && y < numRows && x < numCols){// && (x-max_disparity+j)>=0 ){
	      GPU_gw_cross_correlation_3[thread_1D]=result_1;
	      GPU_gw_normalized_score_4[thread_1D]=help_3;
	      match_matrix[thread_1D]=x-max_disparity+j;
	      //disparity_map[thread_1D]=abs(match_matrix[thread_1D]-x);
	      //match_matrix[thread_1D]=x-(x-max_disparity+j);
	      //if (x==100 && y==100){
		//printf("x=%d y=%d j=%d thread_1D=%d help_3=%f  match_matrix[thread_1D]=%d  \n", x,y,j,thread_1D,help_3,match_matrix[thread_1D]);
	      //}
	    }
	  }
	}
	// moving second matching window
	//first batch transfer
	//temp_1[(threadIdx.x+filterWidth/2)+(threadIdx.y+filterWidth/2)*(blockDim.x+2*(filterWidth/2))]=inputChannel[thread_1D];
	unsigned char help_1,help_2;
	int dest = threadIdx.y * blockDim.x+ threadIdx.x, // index(in a one dimentional array) of a thread(pixel) in a block 
	destY = dest/(blockDim.x+2*(filterWidth/2)), // y index of destination in shared memory
	destX = dest % (blockDim.x+2*(filterWidth/2)),// x index of destination in shared memory
	srcY = blockIdx.y * blockDim.y + destY - (filterWidth/2), // y index in source image (-filterwidth/2 because of overlaping)
	srcX = blockIdx.x * blockDim.x + destX - (filterWidth/2), //// x index in source image (-filterwidth/2 because of overlaping)
	src = srcY * numCols + srcX;// index(in a one dimentional array) of a thread(pixel) in the source image
	int second_srcX= srcX-max_disparity+(j+1);
	int second_src= src-max_disparity+(j+1);// for transfering the second window of cross correlation to shared memory
	//printf("threadIdx.x= %d threadIdx.y= %d blockIdx.x= %d blockIdx.y= %d \n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y);
	//printf("dest= %d destY= %d destX= %d srcY= %d srcX= %d src %d \n",dest,destY,destX,srcY,srcX,src );
	
	// first batch moving for second window in shared memory

	if (destX>=1 && destX< (blockDim.x+2*(filterWidth/2)) && destY < (blockDim.y+2*(filterWidth/2)) ){
	    //if(blockIdx.x==4 && blockIdx.y==0 && j==190 ){
		//printf(" hello booob2 dest=%d x=%d y=%d threadIdx.x=%d threadIdx.y=%d j=%d destX=%d destY=%d \n", dest,x,y,threadIdx.x,threadIdx.y,j,destX,destY);
	    //}
	    help_1=temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX];
	}
	__syncthreads();
	if (destX>=1 && destX< (blockDim.x+2*(filterWidth/2)) && destY < (blockDim.y+2*(filterWidth/2)) ){
	    temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX-1]=help_1;
	}
	//__syncthreads();
	if ( (destX== ((blockDim.x+2*(filterWidth/2))-1)) && destY < (blockDim.y+2*(filterWidth/2))  && second_srcX >=0  ){
	  
	  //if (blockIdx.x==5 && blockIdx.y==5){
	  //    printf(" hello booob x=%d y=%d j=%d destX=%d destY=%d \n", x,y,j,destX,destY);
	  //}
	  if (srcY >= 0 && srcY < numRows && second_srcX >= 0 && (second_srcX+1) < numCols  && second_src >=0 && second_src < numCols * numRows ){
	    temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_2[second_src];
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
	    if (second_srcX < 0){
	      second_srcX = 0;
	    }
	    if((second_srcX) >= numCols){
	      second_srcX =numCols-1 ;
	    }
	    int newindex = srcY * numCols + second_srcX;
	    temp_2[destY*(blockDim.x+2*(filterWidth/2))+destX] = inputChannel_2[newindex];
	  }
	}
	__syncthreads();
	
	//Second batch moving
	int dest_2 = threadIdx.y * blockDim.x + threadIdx.x + blockDim.x * blockDim.y;
	int destY_2 = dest_2 / (blockDim.x+2*(filterWidth/2));
	int destX_2 = dest_2 % (blockDim.x+2*(filterWidth/2));
	int srcY_2 = blockIdx.y * blockDim.y+ destY_2 - (filterWidth/2);
	int srcX_2 = blockIdx.x * blockDim.x+ destX_2 - (filterWidth/2);  
	int src_2 =  srcY_2 * numCols + srcX_2;
	int second_srcX_2= srcX_2-max_disparity+(j+1);;
	int second_src_2= src_2-max_disparity+(j+1);// for transfering second batch of the second window of cross correlation to shared memory
	//if(blockIdx.x==0 && blockIdx.y==0   && destY_2 < (blockDim.y+2*(filterWidth/2))  ){
	//    printf(" hello booob3 dest_2=%d x=%d y=%d threadIdx.x=%d threadIdx.y=%d j=%d destX_2=%d destY_2=%d \n", dest_2,x,y,threadIdx.x,threadIdx.y,j,destX_2,destY_2);
	//}
	//if(destY_2 < (blockDim.y+2*(filterWidth/2))){
	
	///*
	
	if (destX_2>=1 && destX_2< (blockDim.x+2*(filterWidth/2)) && destY_2 < (blockDim.y+2*(filterWidth/2)) ){
	    help_2=temp_2[destY_2*(blockDim.x+2*(filterWidth/2))+destX_2];
	}
	
	__syncthreads();
	if (destX_2>=1 && destX_2< (blockDim.x+2*(filterWidth/2)) && destY_2 < (blockDim.y+2*(filterWidth/2)) ){
	    temp_2[destY_2*(blockDim.x+2*(filterWidth/2))+destX_2-1]=help_2;
	}
	 //__syncthreads();
	//printf("destX=%d ((blockDim.x+2*(filterWidth/2))-1)=%d destY=%d (blockDim.y+2*(filterWidth/2))=%d \n",destX ,((blockDim.x+2*(filterWidth/2))-1),destY,(blockDim.y+2*(filterWidth/2)));
	
	//if ((destX== ((blockDim.x+2*(filterWidth/2))-1))){
	//  printf("destX=%d destY=%d \n", destX,destY);
	//}
	if ( (destX_2== ((blockDim.x+2*(filterWidth/2))-1)) && destY_2 < (blockDim.y+2*(filterWidth/2)) && second_srcX_2 >=0 ){
	  //if (x==100 && y==100){
	  //  printf(" hello booob x=%d y=%d j=%d \n", x,y,j);
	  //}
	  //printf("destX_2=%d destY_2=%d \n", destX_2,destY_2);
	  if (srcY_2 >= 0 && srcY_2 < numRows && second_srcX_2 >= 0 && (second_srcX_2+1) < numCols  && second_src_2 >=0 && second_src_2 < numCols * numRows){
	    temp_2[destY_2*(blockDim.x+2*(filterWidth/2))+destX_2] = inputChannel_2[second_src_2];
	  }
	  else{
	    //temp_1[destY*(blockDim.x+2*(filterWidth/2))+destX] =0; //;
	    //printf("temp_1[help_1=%d]=%d \n",help_1,temp_1[help_1]);
	    if (srcY_2 < 0){
	      srcY_2 = 0;
	    }
	    if(srcY_2 >= numRows){
	      srcY_2 =numRows-1 ;
	    }
	    if (second_srcX_2 < 0){
	      second_srcX_2 = 0;
	    }
	    if((second_srcX_2) >= numCols){
	      second_srcX_2 =numCols-1 ;
	    }
	    int newindex2_2 = srcY_2 * numCols + second_srcX_2;
	    temp_2[destY_2*(blockDim.x+2*(filterWidth/2))+destX_2] = inputChannel_2[newindex2_2];
	  }
	}//*/
	//__syncthreads();
/////////////////////////////////////////////////////////////////
  }
  
  __syncthreads();
  if (y < numRows && y>0 && x>0 && x < numCols ){
     disparity_map[thread_1D]=abs(match_matrix[thread_1D]-x);
  //printf("disparity_map[%d]= %d\n",thread_1D,disparity_map[thread_1D]);
  //__syncthreads();
      //outputChannel[thread_1D] =static_cast<unsigned char>(GPU_gw_average_color_2[thread_1D]);
      outputChannel[thread_1D] =static_cast<unsigned char>(disparity_map[thread_1D]);
  }
  //outputChannel[thread_1D]=GPU_gw_auto_correlation_1[thread_1D];
  __syncthreads();

  //---------------------------------------------------------------------------------------------------------
 
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
  //const dim3 gridSize(5,1);
  //const dim3 gridSize(1,1);
  const dim3 gridSize(ceilf(static_cast<float>(numCols) / blockSize.x),ceilf(static_cast<float>(numRows) / blockSize.y));

  //TODO: Launch a kernel for separating the RGBA image into different color channels
  //separateChannels<<<gridSize,blockSize>>>(d_inputImageRGBA,(int)numRows,(int)numCols,d_red, d_green,d_blue);

  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  //cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  int mid_shared_address=(blockSize.x+(2*(filterWidth/2)))*(blockSize.y+(2*(filterWidth/2)))*sizeof(unsigned char);
  Kernel_1<<<gridSize,blockSize,(blockSize.x+(2*(filterWidth/2)))*(blockSize.y+(2*(filterWidth/2)))*sizeof(unsigned char)*2>>>(d_inputImageGray_1,
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
  
  
  Kernel_2<<<gridSize,blockSize,(blockSize.x+(2*(filterWidth/2)))*(blockSize.y+(2*(filterWidth/2)))*sizeof(char)*2>>>(d_inputImageGray_1,
				      d_inputImageGray_2,
				      d_outputImageGray,
				      GPU_gw_average_color_1,
				      GPU_gw_average_color_2,
				      GPU_gw_auto_correlation_1,
				      GPU_gw_auto_correlation_2,
				      GPU_gw_cross_correlation_3,
				      GPU_gw_normalized_score_4,
				      match_matrix,
				      disparity_map,
				      depth_map,
				      (int)numRows,
				      (int)numCols,
				      d_filter,
				      filterWidth,mid_shared_address);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
 /*
 
  
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