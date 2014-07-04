//Udacity HW2 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include <algorithm>
#include <cassert>
#include <float.h>
#include <limits.h>
#include <stdlib.h> 

// for uchar4 struct
#include <cuda_runtime.h>

//#include "reference_calc.h"
//#include "compare.h"


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <cuda.h>
#include <math.h>  

using namespace cv;
using namespace std;

#define PI 3.141592653589793238463


cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;
cv::Mat imageOutputRGBA_1;
cv::Mat imageOutputRGBA_2;


cv::Mat imageInputGray_1;
cv::Mat imageInputGray_2;
cv::Mat imageOutputGray_1;
cv::Mat imageOutputGray_2;

cv::Mat G_imageInputGray_1;
cv::Mat G_imageInputGray_2;
cv::Mat G_imageOutputGray;

//uchar4 *d_inputImageRGBA_1__;
//uchar4 *d_inputImageRGBA_2__;
//uchar4 *d_outputImageRGBA_1__;
//uchar4 *d_outputImageRGBA_2__;




uchar *d_inputImageGray_1__;
uchar *d_inputImageGray_2__;
uchar *d_outputImageGray__;//_1__;
//uchar *d_outputImageGray_2__;



float *GPU_gw_average_color_1_;
float *GPU_gw_auto_correlation_1_;
float *GPU_gw_average_color_2_;
float *GPU_gw_auto_correlation_2_;
float *GPU_gw_cross_correlation_3_;
float *GPU_gw_normalized_score_4_;
int *match_matrix_;
int *disparity_map_;
int *depth_map_;

int numRows;
int numCols;

float *h_filter__;

//size_t numRows() { return imageInputRGBA.rows; }
//size_t numCols() { return imageInputRGBA.cols; }

//size_t numRows() { return imageInputGray_1.rows; }
//size_t numCols() { return imageInputGray_1.cols; }


//include the definitions of the above functions for this homework
//#include "HW2.cpp"



/*******  DEFINED IN student_func.cu *********/

void your_gaussian_blur(const uchar* const h_inputImageGray_1,
			  uchar* const d_inputImageGray_1,
			  uchar* const h_inputImageGray_2,
			  uchar* const d_inputImageGray_2,
                          uchar* const d_outputImageGray,
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
                          const int filterWidth);
  
void allocateMemoryAndCopyToGPU(const int numRows, const int numCols,
                                const float* const h_filter, const int filterWidth);
/**********************compareImages*******************************/
void compareImages(std::string reference_filename, std::string test_filename, bool useEpsCheck,
				   double perPixelError, double globalError)
{
  cv::Mat reference = cv::imread(reference_filename, -1);
  cv::Mat test = cv::imread(test_filename, -1);

  cv::Mat diff = abs(reference - test);

  cv::Mat diffSingleChannel = diff.reshape(1, 0); //convert to 1 channel, same # rows

  double minVal, maxVal;

  cv::minMaxLoc(diffSingleChannel, &minVal, &maxVal, NULL, NULL); //NULL because we don't care about location

  //now perform transform so that we bump values to the full range

  diffSingleChannel = (diffSingleChannel - minVal) * (255. / (maxVal - minVal));

  diff = diffSingleChannel.reshape(reference.channels(), 0);

  cv::imwrite("nwcc_differenceImage.png", diff);
  //OK, now we can start comparing values...
  unsigned char *referencePtr = reference.ptr<unsigned char>(0);
  unsigned char *testPtr = test.ptr<unsigned char>(0);

  if (useEpsCheck) {
    checkResultsEps(referencePtr, testPtr, reference.rows * reference.cols * reference.channels(), perPixelError, globalError);
  }
  else
  {
    checkResultsExact(referencePtr, testPtr, reference.rows * reference.cols * reference.channels());
  }

  std::cout << "PASS" << std::endl;
  return;
}

/************************channelConvolution***********************************/

void channelConvolution(const unsigned char* const channel,
                        unsigned char* const channelBlurred,
                        const int numRows, const int numCols,
                        const float *filter, const int filterWidth)
{
  //Dealing with an even width filter is trickier
  assert(filterWidth % 2 == 1);

  //For every pixel in the image
  for (int r = 0; r < (int)numRows; ++r) {
    for (int c = 0; c < (int)numCols; ++c) {
      float result = 0.f;
      //For every value in the filter around the pixel (c, r)
      for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
        for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
          //Find the global image position for this filter position
          //clamp to boundary of the image
	  int image_r = std::min(std::max(r + filter_r, 0), static_cast<int>(numRows - 1));
          int image_c = std::min(std::max(c + filter_c, 0), static_cast<int>(numCols - 1));

          float image_value = static_cast<float>(channel[image_r * numCols + image_c]);
          float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

          result += image_value * filter_value;
        }
      }

      channelBlurred[r * numCols + c] = result;
    }
  }
}

/************reference calculation***************/
void referenceCalculation(const uchar4* const rgbaImage, uchar4 *const outputImage,
                          int numRows, int numCols,
                          const float* const filter, const int filterWidth)
{
  unsigned char *red   = new unsigned char[numRows * numCols];
  unsigned char *blue  = new unsigned char[numRows * numCols];
  unsigned char *green = new unsigned char[numRows * numCols];

  unsigned char *redBlurred   = new unsigned char[numRows * numCols];
  unsigned char *blueBlurred  = new unsigned char[numRows * numCols];
  unsigned char *greenBlurred = new unsigned char[numRows * numCols];

  //First we separate the incoming RGBA image into three separate channels
  //for Red, Green and Blue
  for (int i = 0; i < numRows * numCols; ++i) {
    uchar4 rgba = rgbaImage[i];
    red[i]   = rgba.x;
    green[i] = rgba.y;
    blue[i]  = rgba.z;
  }

  //Now we can do the convolution for each of the color channels
  channelConvolution(red, redBlurred, numRows, numCols, filter, filterWidth);
  channelConvolution(green, greenBlurred, numRows, numCols, filter, filterWidth);
  channelConvolution(blue, blueBlurred, numRows, numCols, filter, filterWidth);

  //now recombine into the output image - Alpha is 255 for no transparency
  for (int i = 0; i < numRows * numCols; ++i) {
    uchar4 rgba = make_uchar4(redBlurred[i], greenBlurred[i], blueBlurred[i], 255);
    outputImage[i] = rgba;
  }

  delete[] red;
  delete[] green;
  delete[] blue;

  delete[] redBlurred;
  delete[] greenBlurred;
  delete[] blueBlurred;
}
/************reference calculation2***************/
void CPU_referenceCalculation(const uchar* const grayImage_1,const uchar* const grayImage_2, uchar *const outputImage_1,
                          uchar *const outputImage_2,int numRows, int numCols,
                          const float* const filter, const int filterWidth )
{
  //unsigned char *gray_1  = new unsigned char[numRows * numCols];
  //gw_average_color ====>  gray_weighted_average_color 
  float *gw_average_color_1   = new float[numRows * numCols]; 
  //gw_auto_correlation ====>  gray_weighted_auto_correlation
  float *gw_auto_correlation_1   = new float[numRows * numCols]; 
  //unsigned char *temp_1   = new unsigned char[numRows * numCols];  
  //unsigned char *gray_2  = new unsigned char[numRows * numCols];
  float *gw_average_color_2   = new float[numRows * numCols];
  float *gw_auto_correlation_2   = new float[numRows * numCols]; 
  //unsigned char *temp_2   = new unsigned char[numRows * numCols];  
  //gw_cross_correlation_3 ===>gray_weighted_cross_correlation_3
  float **gw_cross_correlation_3   = new float*[numRows * numCols]; 
  //gw_normalized_score_4 ===>gray_weighted_normalized_score_4 
  float **gw_normalized_score_4   = new float*[numRows * numCols];
  //unsigned char **temp_3   = new unsigned char*[numRows * numCols]; 
  
  //unsigned char *temp_4   = new unsigned char[numRows * numCols];
  //unsigned char *temp_5   = new unsigned char[numRows * numCols]; 
  
  int *match_matrix=new int[numRows*numCols]; 
  int *disparity_map=new int [numRows*numCols];
  //unsigned char *depth_map=new unsigned char [numRows*numCols];
  int *depth_map=new int [numRows*numCols];
 
  /*
  for (int r = 0; r < (int)numRows; ++r) {
    for (int c = 0; c < (int)numCols; ++c) {
      float image_value = static_cast<float>(channel[r * numCols + c]);
    }
  }*/
  const int max_disparity=80;//(int)numCols;
  int index;
  //for (int i = 0; i < numRows * numCols; ++i){
  for (int r = 0; r < (int)numRows; ++r){
    for (int c = 0; c < (int) numCols; ++c){
      index=r * numCols + c;
      if (c>=max_disparity){
	gw_cross_correlation_3[index]=new float[max_disparity];
	gw_normalized_score_4[index]=new float[max_disparity];
	//temp_3[i]=new unsigned char[max_disparity];
      } 
      else{
	gw_cross_correlation_3[index]=new float[c+1];
	gw_normalized_score_4[index]=new float[c+1];
      }
    }
  }
  
  //Now we can do the convolution for each of the color channels
  //channelConvolution(grayImage_1, gw_average_color_1, numRows, numCols, filter, filterWidth);
  //channelConvolution(grayImage_2, gw_average_color_2, numRows, numCols, filter, filterWidth);
  
  //Dealing with an even width filter is trickier
  assert(filterWidth % 2 == 1);

  //For every pixel in the image
  for (int r = 0; r < (int)numRows; ++r) {
    for (int c = 0; c < (int)numCols; ++c) {
      float result_1 = 0.f;
      float result_2 = 0.f;
      //For every value in the filter around the pixel (c, r)
      for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
        for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
          //Find the global image position for this filter position
          //clamp to boundary of the image
	  int image_r = std::min(std::max(r + filter_r, 0), static_cast<int>(numRows - 1));
          int image_c = std::min(std::max(c + filter_c, 0), static_cast<int>(numCols - 1));

          float image_value_1 = static_cast<float>(grayImage_1[image_r * numCols + image_c]);
	  float image_value_2 = static_cast<float>(grayImage_2[image_r * numCols + image_c]);
          float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

          result_1 += image_value_1 * filter_value;
	  result_2 += image_value_2 * filter_value;
        }
      }

      gw_average_color_1[r * numCols + c] = result_1/(filterWidth*filterWidth);
      gw_average_color_2[r * numCols + c] = result_2/(filterWidth*filterWidth);
    }
  }
  
  //now recombine into the output image - Alpha is 255 for no transparency
  
  //for (int i = 0; i < numRows * numCols; ++i) {
    //temp_1[i] = pow (static_cast<float> (grayImage_1[i])-static_cast<float> (gw_average_color_1[i]),2.0); 
    //temp_2[i] = pow (static_cast<float> (grayImage_2[i])-static_cast<float> (gw_average_color_2[i]),2.0);
  //}
  for (int r = 0; r < (int)numRows; ++r) {
    for (int c = 0; c < (int)numCols; ++c) {
      float result_1 = 0.f;
      float result_2 = 0.f;
      //For every value in the filter around the pixel (c, r)
      for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
        for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
          //Find the global image position for this filter position
          //clamp to boundary of the image
	  int image_r = std::min(std::max(r + filter_r, 0), static_cast<int>(numRows - 1));
          int image_c = std::min(std::max(c + filter_c, 0), static_cast<int>(numCols - 1));

          float image_value_1 = static_cast<float>(grayImage_1[image_r * numCols + image_c])-static_cast<float>(gw_average_color_1[r * numCols + c]);
          float image_value_2 = static_cast<float>(grayImage_2[image_r * numCols + image_c])-static_cast<float>(gw_average_color_2[r * numCols + c]);
	  float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

          result_1 += image_value_1* image_value_1 * filter_value;
	  result_2 += image_value_2* image_value_2 * filter_value;
        }
      }

      gw_auto_correlation_1[r * numCols + c] = result_1/(filterWidth*filterWidth);
      gw_auto_correlation_2[r * numCols + c] = result_2/(filterWidth*filterWidth);
    }
  }

  /*for (int r = 0; r < (int)numRows; ++r) {
      for (int c = 0; c < (int)numCols; ++c) {
	printf("%d ",gw_auto_correlation_1[r * numCols + c]);
      }
      printf("\n");
  }*/
  
  //channelConvolution(temp_1,gw_auto_correlation_1, numRows, numCols, filter, filterWidth);
  //channelConvolution(temp_2, gw_auto_correlation_2, numRows, numCols, filter, filterWidth);
 
    
  int disparity; //this value used for save the number of pixels which should be compared for each pixel
  for (int r = 0; r < (int)numRows; ++r) {
    for (int c = 0; c < (int)numCols; ++c) {
      if ((r * numCols + c)%10000==0)
	printf("%d\n",r);
	
      // if fixed pixel'S column index is more than max_disparity then just compare the fixed pixel convolution
      // with max_disparity numebr of pixel from left side of convolution window in right side image
      //otherwise comare it with c+1 pixel from right side image
      
      
      disparity=(c>=max_disparity)?max_disparity:c;  
      
      if (disparity==max_disparity){
	for (int j=0; j<max_disparity;j++){
	  //temp_3[r * numCols + c][j]=(static_cast<float> (grayImage_1[r * numCols + c])-static_cast<float> (gw_average_color_1[r * numCols + c]))*(static_cast<float> (grayImage_2[r * numCols + j])-static_cast<float> (gw_average_color_2[r * numCols + j]));
	  float result= 0.f;
	  //For every value in the filter around the pixel (c, r)
	  for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
	    for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
	      //Find the global image position for this filter position
	      //clamp to boundary of the image
	      int image_r = std::min(std::max(r + filter_r, 0), static_cast<int>(numRows - 1));
	      int image_c = std::min(std::max(c + filter_c, 0), static_cast<int>(numCols - 1));
	      int image_j = std::min(std::max((c-max_disparity+j) + filter_c, 0), static_cast<int>(numCols - 1));

	      float image_value_1 = static_cast<float>(grayImage_1[image_r * numCols + image_c])-static_cast<float>(gw_average_color_1[r * numCols + c]);
	      float image_value_2 = static_cast<float>(grayImage_2[image_r * numCols + image_j])-static_cast<float>(gw_average_color_2[r * numCols + (c-max_disparity+j)]);
	      float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

	      result+= image_value_1* image_value_2* filter_value;
	    
	    }
	  }
	  //if (r==200 && c== 200 )
	  //printf("%f ",result);
	  gw_cross_correlation_3[r * numCols + c][j] = result/(filterWidth*filterWidth);
	  
	}

      }
      else{
	for (int j=0; j<=c;j++){
	  //temp_3[r * numCols + c][j]=(static_cast<float> (grayImage_1[r * numCols + c])-static_cast<float> (gw_average_color_1[r * numCols + c]))*(static_cast<float> (grayImage_2[r * numCols + j])-static_cast<float> (gw_average_color_2[r * numCols + j]));
	  float result= 0.f;
	  //For every value in the filter around the pixel (c, r)
	  for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
	    for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
	      //Find the global image position for this filter position
	      //clamp to boundary of the image
	      int image_r = std::min(std::max(r + filter_r, 0), static_cast<int>(numRows - 1));
	      int image_c = std::min(std::max(c + filter_c, 0), static_cast<int>(numCols - 1));
	      int image_j = std::min(std::max(j + filter_c, 0), static_cast<int>(numCols - 1));

	      float image_value_1 = static_cast<float>(grayImage_1[image_r * numCols + image_c])-static_cast<float>(gw_average_color_1[r * numCols + c]);
	      float image_value_2 = static_cast<float>(grayImage_2[image_r * numCols + image_j])-static_cast<float>(gw_average_color_2[r * numCols + j]);
	      float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

	      result+= image_value_1* image_value_2* filter_value;
	    
	    }
	  }
	  //if (r==200 && c== 200 )
	  //printf("%f ",result);
	  gw_cross_correlation_3[r * numCols + c][j] = result/(filterWidth*filterWidth);
	  
	}
      }
      
    }
  }/*
  for (int r = 200; r <220 ; ++r) { //(int)numRows
    for (int c = 200; c <220 ; ++c) { //(int)numCols
      printf("gw_auto_correlation_2[%d]=%f\n",r * numCols + c, gw_auto_correlation_2[r * numCols + c]);
    }
  }
  for (int j=0; j<max_disparity;j++){
    printf("gw_cross_correlation_3[239][%d]=%f\n",j, gw_cross_correlation_3[239][j]);
  }
  */
    
  
  for (int r = 0; r < (int)numRows; ++r) {
    for (int c = 0; c < (int)numCols; ++c) {
      disparity=(c>=max_disparity)?max_disparity:c;  
      
      if (disparity==max_disparity){
	for (int j=0; j<max_disparity;j++){
	    /*if (r*numCols+c==200)
	    {
	      printf("gw_cross_correlation_3[%d][%d]=%f\n",r * numCols + c,j,gw_cross_correlation_3[r * numCols + c][j]);
	      printf("gw_auto_correlation_1[%d]=%d\n",r * numCols + c,gw_auto_correlation_1[r * numCols + c]);
	      printf("gw_auto_correlation_2[%d]=%d\n",r * numCols + j,gw_auto_correlation_2[r *numCols + j]);
	    }*/
	    gw_normalized_score_4[r * numCols + c][j]=static_cast<float>(gw_cross_correlation_3[r * numCols + c][j])/static_cast<float>(sqrt(static_cast<float>(gw_auto_correlation_1[r * numCols + c])*static_cast<float>(gw_auto_correlation_2[r * numCols + c-max_disparity+j])));
	}
      }
      else{
	for (int j=0; j<=c;j++){
	    /*if (r*numCols+c==200)
	    {
	      printf("gw_cross_correlation_3[%d][%d]=%f\n",r * numCols + c,j,gw_cross_correlation_3[r * numCols + c][j]);
	      printf("gw_auto_correlation_1[%d]=%d\n",r * numCols + c,gw_auto_correlation_1[r * numCols + c]);
	      printf("gw_auto_correlation_2[%d]=%d\n",r * numCols + j,gw_auto_correlation_2[r *numCols + j]);
	    }*/
	    gw_normalized_score_4[r * numCols + c][j]=static_cast<float>(gw_cross_correlation_3[r * numCols + c][j])/static_cast<float>(sqrt(static_cast<float>(gw_auto_correlation_1[r * numCols + c])*static_cast<float>(gw_auto_correlation_2[r * numCols + j])));
	    
	}
      }
    
    }
  }
  /*
   for (int j=0; j<max_disparity;j++){
    printf("gw_normalized_score_4[50306][%d]=%f\n",156+j, gw_normalized_score_4[50306][j]);
  }*/
  
    
  /*
  for (int j=0; j<max_disparity;j++) {
     printf("gw_normalized_score_4[200][%d]=%f\n",j,gw_normalized_score_4[200][j]);
  }*/
  printf("\nFLT_MIN=%f\n",FLT_MIN);
  printf("\nINT_MIN=%d\n",INT_MIN);
  float max_match_score= -1000;
  int match_index=INT_MIN;
  int search_range;
  
  for (int r = 0; r < (int)numRows; ++r) {
      for (int c = 0; c < (int)numCols; ++c) {
	max_match_score=-100000;
	match_index=INT_MIN;
	search_range=(c>=max_disparity)?max_disparity:c;
	for (int j=0; j<=search_range;j++){
	  if (gw_normalized_score_4[r * numCols + c][j]>max_match_score){
	    max_match_score=gw_normalized_score_4[r * numCols + c][j];
	    match_index=j;
	    
	  }
	  
	}
	
	if (search_range==max_disparity){
	  match_matrix[r * numCols + c]=c-max_disparity+match_index;
	}
	else{
	  match_matrix[r * numCols + c]=match_index;
	}
	
	
	/*
	if ((r==239) &&(c==0)){
	  printf("search_range = %d",search_range);
	  for (int j=0; j<=search_range;j++){
	    printf("At position (%d ,%d)[%d] the w_normalized_score_4[r * numCols + c][j]=%f is\n",r,c,j,gw_normalized_score_4[r * numCols + c][j]);
	  }
	}*/
	
      }
  }
  
  
  for (int r = 0; r < (int)numRows; ++r) {
      for (int c = 0; c < (int)numCols; ++c) {
	disparity_map[r * numCols + c]=abs(match_matrix[r * numCols + c]-c);
	//depth_map[r * numCols + c]=max_founded_disparity*(255.0*(1.0/static_cast<float>(disparity_map[r * numCols + c])));
      }
  }
  int max_founded_disparity=INT_MIN;
  int min_founded_disparity=INT_MAX;
  for (int r = 0; r < (int)numRows; ++r) {
      for (int c = 0; c < (int)numCols; ++c) {
	if (disparity_map[r * numCols + c]>max_founded_disparity){
	  max_founded_disparity=disparity_map[r * numCols + c];
	}
	if (disparity_map[r * numCols + c]<min_founded_disparity && disparity_map[r * numCols + c]!=0 ){
	  min_founded_disparity=disparity_map[r * numCols + c];
	  
	}
	/*
	if (disparity_map[r * numCols + c]==0){
	    printf("At position (%d ,%d) the disparity is zero\n",r,c);
	}
	if (disparity_map[r * numCols + c]==-2147483648){
	    printf("At position (%d ,%d) the disparity is -2147483648 and match matrix=%d \n",r,c,match_matrix[r * numCols + c]);
	}*/
	
      }
  }
  if (min_founded_disparity==0){
	  min_founded_disparity+=1;
  }
  for (int r = 0; r < (int)numRows; ++r) {
      for (int c = 0; c < (int)numCols; ++c) {
	//disparity_map[r * numCols + c]=abs(match_matrix[r * numCols + c]-c);
	if (disparity_map[r * numCols + c]==0){
	   depth_map[r * numCols + c]=255;
	  }
	  else{
	      depth_map[r * numCols + c]=(float)(max_founded_disparity)*(static_cast<float>(1.0)/(static_cast<float>(disparity_map[r * numCols + c])));
	  }
      }
  }
  
  printf("max_founded_disparity= %d\n",max_founded_disparity);
  printf("min_founded_disparity= %d\n",min_founded_disparity);
  printf("match_matrix[111,356]=%d\n",match_matrix[111 * numCols + 356]);
  
  /*
  for (int j=350;j<361;j++){
      printf("gw_normalized_score_4[111][%d]=%f\n",j,gw_normalized_score_4[111 * numCols + 356][j]);
  } 
  
  printf("\nmatch matrix:\n");
  for (int r = 180; r < 210; ++r) {
      for (int c = 180; c < 210; ++c) {
	printf("%d ",match_matrix[r * numCols + c]);
      }
      printf("\n");
  }*/
  
  printf("\ndisparity_map:\n");
  int r,c;
  for (r = 180; r < 220; ++r) {
      for (c = 180; c < 220; ++c) {
	printf("%d ",disparity_map[r * numCols + c]);
      }
      
      printf(" babak%d",r);
      printf("\n");
  }
  printf("r= %d c= %d  Goodbye lenin \n",r,c);
  printf("r= %d c= %d  Goodbye lenin \n",r,c);
  printf("%s\n","Goodbye lenin"); 
  printf("%s\n","hi karen1");
  
  /*
  printf("\ndepth map:\n");
  for (int r = 200; r < 210; ++r) {
      for (int c = 200; c < 210; ++c) {
	printf("%d ",depth_map[r * numCols + c]);
      }
      printf("\n");
  }*/
  /*
  for (int j=0; j<max_disparity;j++){
    printf("%f ",gw_normalized_score_4[200 * numCols + 200][j]);
    
  }
  printf("\n");
  for (int j=0; j<max_disparity;j++){
    printf("%f ",gw_normalized_score_4[200 * numCols + 201][j]);
    
  }
  printf("\n");
  for (int j=0; j<max_disparity;j++){
    printf("%f ",gw_normalized_score_4[200 * numCols + 202][j]);
    
  }
  
 
  for (int j=0; j<max_disparity;j++){
    printf("%f ",gw_cross_correlation_3[200 * numCols + 200][j]);
    
  }
  printf("\n");
  for (int j=0; j<max_disparity;j++){
    printf("%f ",gw_cross_correlation_3[200 * numCols + 201][j]);
    
  }
  printf("\n");
  for (int j=0; j<max_disparity;j++){
    printf("%f ",gw_cross_correlation_3[200 * numCols + 202][j]);
    
  }
  printf("\n");
   */
  printf("%s\n","hi karen2");
 int min_value=disparity_map[0], max_value=disparity_map[0];
  for (int i = 0; i < numRows * numCols; ++i) {
    //outputImage_1[i] =static_cast<int> (depth_map[i]);
    //printf("disparity_map[%d]=%d\n",i,disparity_map[i]);
    
    outputImage_1[i] =static_cast<uchar>(disparity_map[i]);
    if (disparity_map[i]==-2147483648){
      printf("disparity_map[%d][%d]=%d\n",i/450,i-((i/450)*450),disparity_map[i]);
    }
    if (disparity_map[i]<min_value)
      min_value=disparity_map[i];
    if (disparity_map[i]>max_value)
      max_value=disparity_map[i]; 
    //printf("hi babak");
    //printf("outputImage_1[%d]=%d\n",i,outputImage_1[i]);
    
    //outputImage_2[i] = disparity_map_2[i];
  }
  printf("min_value=%d\n",min_value);
  printf("max_value=%d\n",max_value);
 
 // delete[] grayfiltered_1;
 // delete[] grayfiltered_2;
 

}
/************CPU_postprocess*******/
void CPU_postProcess(const std::string& output_file, uchar* data_ptr) {
  cv::Mat output(numRows, numCols, CV_8UC1, (void*)data_ptr);
  
  Mat img_hist_equalized;
  equalizeHist(output, img_hist_equalized); //equalize the histogram

  //cv::Mat imageOutputBGR;
  //cv::cvtColor(output, imageOutputBGR, CV_RGBA2BGR);
  //cv::cvtColor(output, imageOutputBGR, CV_GRAY2BGR);
  //output the image
  printf("%s\n","hi karen3");
  //cv::imwrite(output_file.c_str(), imageOutputBGR);
  
  cv::imwrite(output_file.c_str(), img_hist_equalized);
  
  //cout << output << endl;
  
  
  
  
  
  //cv::imwrite(output_file.c_str(), output);
}


/************preprocess**********/
void  CPU_Preprocess(uchar **h_inputImageGray_1,uchar **h_inputImageGray_2,
		 uchar **h_outputImageGray_1, uchar **h_outputImageGray_2,
                 float **h_filter, int *filterWidth, const std::string &filename_1,const std::string &filename_2){
 
  cv::Mat image_1 = cv::imread(filename_1.c_str(), CV_LOAD_IMAGE_COLOR);
  cv::Mat image_2 = cv::imread(filename_2.c_str(), CV_LOAD_IMAGE_COLOR);
 
  if (image_1.empty() || image_2.empty()) {
    std::cerr << "Couldn't open file: " << filename_1<< "or"<<filename_2 << std::endl;
    exit(1);
  }
  
  //copy the transformed image from first parameter to the second one
  cv::cvtColor( image_1, imageInputGray_1, CV_BGR2GRAY);
  cv::cvtColor( image_2, imageInputGray_2, CV_BGR2GRAY);

  //allocate memory for the output
 
  imageOutputGray_1.create(image_1.rows, image_1.cols, CV_8UC1);
  imageOutputGray_2.create(image_2.rows, image_2.cols, CV_8UC1);
  
  if (!imageInputGray_1.isContinuous() || !imageOutputGray_1.isContinuous()||
    !imageInputGray_2.isContinuous() || !imageOutputGray_2.isContinuous()){
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }
  *h_inputImageGray_1  = (uchar *)imageInputGray_1.ptr<unsigned char>(0);
  *h_inputImageGray_2  = (uchar *)imageInputGray_2.ptr<unsigned char>(0);
  *h_outputImageGray_1 = (uchar *)imageOutputGray_1.ptr<unsigned char>(0);
  *h_outputImageGray_2 = (uchar *)imageOutputGray_2.ptr<unsigned char>(0);
  numRows=imageInputGray_1.rows;
  numCols=imageInputGray_1.cols;
  
  const int numPixels = numRows * numCols;
  //now create the filter that they will use
  const int blurKernelWidth = 7;
  const float blurKernelSigma = 2.;
  
  *filterWidth = blurKernelWidth;

  //create and fill the filter we will convolve with
  *h_filter = new float[blurKernelWidth * blurKernelWidth];
  h_filter__ = *h_filter;

  float filterSum = 0.f; //for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = pow (cos ( r * PI / blurKernelWidth),2.0)*pow (cos ( c * PI / blurKernelWidth),2.0);
      //float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      //filterSum += filterValue;
    }
  }  
  
  
}
//************************************



//************************************


void GPU_preProcess(uchar **h_inputImageGray_1,uchar **h_inputImageGray_2, uchar **h_outputImageGray,
                uchar **d_inputImageGray_1, uchar **d_inputImageGray_2,uchar **d_outputImageGray,
		float **GPU_gw_average_color_1,float **GPU_gw_auto_correlation_1,
		float **GPU_gw_average_color_2,float **GPU_gw_auto_correlation_2,
		float **GPU_gw_cross_correlation_3,float **GPU_gw_normalized_score_4,
                int **match_matrix,int **disparity_map,int **depth_map,
                //unsigned char **d_redBlurred,
                //unsigned char **d_redBlurred_1,
		//unsigned char **d_greenBlurred,
                //unsigned char **d_greenBlurred_1,
		//unsigned char **d_blueBlurred,
		//unsigned char **d_blueBlurred_1,
		float **h_filter, int *filterWidth, 
		const std::string &filename_1,const std::string &filename_2){
                

  //make sure the context initializes ok
  //checkCudaErrors(cudaFree(0));
  
  cv::Mat image_1 = cv::imread(filename_1.c_str(), CV_LOAD_IMAGE_COLOR);
  cv::Mat image_2 = cv::imread(filename_2.c_str(), CV_LOAD_IMAGE_COLOR);
 
  if (image_1.empty() || image_2.empty()) {
    std::cerr << "Couldn't open file: " << filename_1<< "or"<<filename_2 << std::endl;
    exit(1);
  }
  
  //copy the transformed image from first parameter to the second one
  cv::cvtColor( image_1, G_imageInputGray_1, CV_BGR2GRAY);
  cv::cvtColor( image_2, G_imageInputGray_2, CV_BGR2GRAY);

  //allocate memory for the output
 
  G_imageOutputGray.create(image_1.rows, image_1.cols, CV_8UC1);

  
  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!G_imageInputGray_1.isContinuous() || !G_imageOutputGray.isContinuous()||
    !G_imageInputGray_2.isContinuous()){
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *h_inputImageGray_1  = (unsigned char *)G_imageInputGray_1.ptr<unsigned char>(0);
  *h_inputImageGray_2  = (unsigned char *)G_imageInputGray_2.ptr<unsigned char>(0);
  *h_outputImageGray = (unsigned char *)G_imageOutputGray.ptr<unsigned char>(0);
 
  
  
  numRows=G_imageInputGray_1.rows;
  numCols=G_imageInputGray_1.cols;
  
  const int numPixels = numRows * numCols;

  //allocate memory on the device for both input and output
  checkCudaErrors(   cudaMalloc((void**)d_inputImageGray_1, sizeof(unsigned char) * numPixels)   );
  
  cudaError_t t1 = cudaGetLastError();
  
  cout << cudaGetErrorString(t1) << endl;

  char c;
  cin >> c;  
  
  checkCudaErrors(cudaMalloc((void**)d_inputImageGray_2, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc((void**)d_outputImageGray, sizeof(unsigned char) * numPixels));
  
  checkCudaErrors(cudaMemset(*d_outputImageGray, 0,  sizeof(unsigned char) * numPixels)); //make sure no memory is left laying around
  
  checkCudaErrors(cudaMalloc((void**)GPU_gw_average_color_1,    sizeof(float) * numPixels));
  checkCudaErrors(cudaMalloc((void**)GPU_gw_auto_correlation_1,  sizeof(float) * numPixels));
  checkCudaErrors(cudaMalloc((void**)GPU_gw_average_color_2,   sizeof(float) * numPixels));
  checkCudaErrors(cudaMalloc((void**)GPU_gw_auto_correlation_2,  sizeof(float) * numPixels));
  checkCudaErrors(cudaMemset(*GPU_gw_average_color_1,   0, sizeof(float) * numPixels));
  checkCudaErrors(cudaMemset(*GPU_gw_auto_correlation_1, 0, sizeof(float) * numPixels));
  checkCudaErrors(cudaMemset(*GPU_gw_average_color_2,  0, sizeof(float) * numPixels));
  checkCudaErrors(cudaMemset(*GPU_gw_auto_correlation_2, 0, sizeof(float) * numPixels));
  
  checkCudaErrors(cudaMalloc((void**)GPU_gw_cross_correlation_3,   sizeof(float) * numPixels));
  checkCudaErrors(cudaMalloc((void**)GPU_gw_normalized_score_4,  sizeof(float) * numPixels));
  checkCudaErrors(cudaMemset(*GPU_gw_cross_correlation_3, 0, sizeof(float) * numPixels));
  checkCudaErrors(cudaMemset(*GPU_gw_normalized_score_4, 0, sizeof(float) * numPixels));
  
  checkCudaErrors(cudaMalloc((void**)match_matrix,  sizeof(int) * numPixels));
  checkCudaErrors(cudaMalloc((void**)disparity_map,  sizeof(int) * numPixels));
  checkCudaErrors(cudaMalloc((void**)depth_map,  sizeof(int) * numPixels));
  checkCudaErrors(cudaMemset((void*) *match_matrix, 0,  sizeof(int) * numPixels));
  checkCudaErrors(cudaMemset((void*)*disparity_map, 0,  sizeof(int) * numPixels));
  checkCudaErrors(cudaMemset((void*)*depth_map, 0,  sizeof(int) * numPixels));
  
  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_inputImageGray_1, *h_inputImageGray_1, sizeof(uchar) * numPixels, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*d_inputImageGray_2, *h_inputImageGray_2, sizeof(uchar) * numPixels, cudaMemcpyHostToDevice));

  d_inputImageGray_1__=*d_inputImageGray_1;
  d_inputImageGray_2__=*d_inputImageGray_2;
  d_outputImageGray__ =*d_outputImageGray;

  GPU_gw_average_color_1_=*GPU_gw_average_color_1;
  GPU_gw_auto_correlation_1_=*GPU_gw_auto_correlation_1;
  GPU_gw_average_color_2_=*GPU_gw_average_color_2;
  GPU_gw_auto_correlation_2_=*GPU_gw_auto_correlation_2;
  GPU_gw_cross_correlation_3_=*GPU_gw_cross_correlation_3;
  GPU_gw_normalized_score_4_=*GPU_gw_normalized_score_4;
  match_matrix_=*match_matrix;
  disparity_map_=*disparity_map;
  depth_map_=*depth_map;
  
   

  //now create the filter that they will use
  const int blurKernelWidth = 7;
  const float blurKernelSigma = 2.;
  
  *filterWidth = blurKernelWidth;
  
  //create and fill the filter we will convolve with
  *h_filter = new float[blurKernelWidth * blurKernelWidth];
  h_filter__ = *h_filter;

  float filterSum = 0.f; //for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = pow (cos ( r * PI / blurKernelWidth),2.0)*pow (cos ( c * PI / blurKernelWidth),2.0);
      //float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      //filterSum += filterValue;
    }
  }
  /*
  float normalizationFactor = 1.f / filterSum;

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    }
  }


  //blurred
  checkCudaErrors(cudaMalloc(d_redBlurred,    sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_greenBlurred,  sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_blueBlurred,   sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_redBlurred,   0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_blueBlurred,  0, sizeof(unsigned char) * numPixels));
  
 */
}

/************postprocess*******/
void GPU_postProcess(const std::string& output_file, uchar* data_ptr) {
  cv::Mat output(numRows, numCols, CV_8UC1, (void*)data_ptr);

  cv::Mat imageOutputBGR;
  cv::cvtColor(output, imageOutputBGR, CV_GRAY2BGR);//CV_RGBA2BGR);
  //output the image
  cv::imwrite(output_file.c_str(), imageOutputBGR);
}
/**********cleanUp**********/
void CPU_cleanUp(uchar **h_inputImageGray_1,uchar **h_inputImageGray_2, uchar **h_outputImageGray_1, uchar **h_outputImageGray_2)
{

  
  delete[] h_inputImageGray_1;
  delete[] h_inputImageGray_2;
  delete[] h_outputImageGray_1;
  delete[] h_outputImageGray_2;
  delete[] h_filter__;

}

/**********cleanUp**********/
void GPU_cleanUp(void)
{

  checkCudaErrors(cudaFree(d_inputImageGray_1__));
  checkCudaErrors(cudaFree(d_inputImageGray_2__));
  checkCudaErrors(cudaFree(d_outputImageGray__));
  checkCudaErrors(cudaFree(GPU_gw_average_color_1_));
  checkCudaErrors(cudaFree(GPU_gw_auto_correlation_1_));
  checkCudaErrors(cudaFree(GPU_gw_average_color_2_));
  checkCudaErrors(cudaFree(GPU_gw_auto_correlation_2_));
  checkCudaErrors(cudaFree(GPU_gw_cross_correlation_3_));
  checkCudaErrors(cudaFree(GPU_gw_normalized_score_4_));
  checkCudaErrors(cudaFree(match_matrix_));
  checkCudaErrors(cudaFree(disparity_map_));
  checkCudaErrors(cudaFree(depth_map_));
  
  
  delete[] h_filter__;

}
/************generate reference image***************/
// An unused bit of code showing how to accomplish this assignment using OpenCV.  It is much faster 
//    than the naive implementation in reference_calc.cpp.
void generateReferenceImage(std::string input_file, std::string reference_file, int kernel_size)
{
	cv::Mat input = cv::imread(input_file);
	// Create an identical image for the output as a placeholder
	cv::Mat reference = cv::imread(input_file);
	cv::GaussianBlur(input, reference, cv::Size2i(kernel_size, kernel_size),0);
	cv::imwrite(reference_file, reference);
}

/*******  GPU_Part *************/

//***********************
void GPU_Part(int argc, char **argv)
{
  
  uchar *h_inputImageGray_1,*h_inputImageGray_2,*d_inputImageGray_1,*d_inputImageGray_2;
  uchar *h_outputImageGray,*d_outputImageGray;
  //unsigned char *d_GrayBlurred,*d_GrayBlurred_1;//,*d_redBlurred,*d_redBlurred_1, *d_greenBlurred,*d_greenBlurred_1, *d_blueBlurred,*d_blueBlurred_1;
  
  float *GPU_gw_average_color_1; //  = new float[numRows * numCols]; 
  float *GPU_gw_auto_correlation_1; //  = new float[numRows * numCols]; 
  float *GPU_gw_average_color_2; //  = new float[numRows * numCols];
  float *GPU_gw_auto_correlation_2; //  = new float[numRows * numCols]; 
  float *GPU_gw_cross_correlation_3; //  = new float*[numRows * numCols]; 
  float *GPU_gw_normalized_score_4; //  = new float*[numRows * numCols];
 
  int *match_matrix;//=new int[numRows*numCols]; 
  int *disparity_map;//=new int [numRows*numCols];
  int *depth_map;//=new int [numRows*numCols];
 
  /*
  for (int r = 0; r < (int)numRows; ++r) {
    for (int c = 0; c < (int)numCols; ++c) {
      float image_value = static_cast<float>(channel[r * numCols + c]);
    }
  }*/
  const int max_disparity=80;
  
  

  float *h_filter;
  int    filterWidth;
  std::string input_file_1;
  std::string input_file_2;
  std::string output_file;

  std::string reference_file;
  double perPixelError = 0.0;
  double globalError   = 0.0;
  bool useEpsCheck = false;
  switch (argc)
  {
	case 2:
	  input_file_1 = std::string(argv[1]);
	  output_file = "nwcc_output_1.png";
	  //output_file_2 = "nwcc_output_2.png";
	  reference_file = "nwcc_reference.png";
	  break;
	case 3:
	  input_file_1  = std::string(argv[1]);
	  input_file_2 = std::string(argv[2]);
	  output_file = "nwcc_output_1.png";
	  //output_file_2 = "nwcc_output_2.png";
	  reference_file = "nwcc_reference.png";
	  break;
	case 4:
	  input_file_1  = std::string(argv[1]);
          output_file = std::string(argv[2]);
	  reference_file = std::string(argv[3]);
	  break;
	case 6:
	  useEpsCheck=true;
	  input_file_1  = std::string(argv[1]);
	  output_file = std::string(argv[2]);
	  reference_file = std::string(argv[3]);
	  perPixelError = atof(argv[4]);
      globalError   = atof(argv[5]);
	  break;
	default:
      std::cerr << "Usage: ./nwcc input_file_1  input_file_2  [output_filename] [reference_filename] [perPixelError] [globalError]" << std::endl;
      exit(1);
  }
  GPU_preProcess(&h_inputImageGray_1,&h_inputImageGray_2,&h_outputImageGray,
		 &d_inputImageGray_1,&d_inputImageGray_2,&d_outputImageGray,
		 &GPU_gw_average_color_1,&GPU_gw_auto_correlation_1,
		 &GPU_gw_average_color_2,&GPU_gw_auto_correlation_2,
		 &GPU_gw_cross_correlation_3,&GPU_gw_normalized_score_4,
                 &match_matrix,&disparity_map,&depth_map,
		 &h_filter, &filterWidth, input_file_1,input_file_2);
  
  
  
  
  //load the image and give us our input and output pointers
  //preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &h_outputImageRGBA_1, &d_inputImageRGBA, &d_outputImageRGBA, &d_outputImageRGBA_1,
  //           &d_redBlurred,&d_redBlurred_1, &d_greenBlurred,&d_greenBlurred_1, &d_blueBlurred,&d_blueBlurred_1,
  //           &h_filter, &filterWidth, input_file);

  
  allocateMemoryAndCopyToGPU(numRows, numCols, h_filter, filterWidth);
  GpuTimer timer;
  timer.Start();
 
  //your_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA,d_outputImageRGBA_1, numRows, numCols,
  //                   d_redBlurred,d_redBlurred_1, d_greenBlurred,d_greenBlurred_1, d_blueBlurred,d_blueBlurred_1, filterWidth);
  
  your_gaussian_blur(h_inputImageGray_1, d_inputImageGray_1,
		     h_inputImageGray_2, d_inputImageGray_2, d_outputImageGray,
		     GPU_gw_average_color_1,GPU_gw_auto_correlation_1,
		     GPU_gw_average_color_2,GPU_gw_auto_correlation_2,
		     GPU_gw_cross_correlation_3,GPU_gw_normalized_score_4,
                     match_matrix,disparity_map,depth_map,
		     numRows, numCols, filterWidth);
  
   timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }
  
  //check results and output the blurred image
  int numPixels = numRows*numCols;
  //
  //copy the output back to the host
  checkCudaErrors(cudaMemcpy(h_outputImageGray, d_outputImageGray__, sizeof(uchar) * numPixels, cudaMemcpyDeviceToHost));
  //checkCudaErrors(cudaMemcpy(h_outputImageRGBA_1, d_outputImageRGBA_1__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));
  
  
  //GPU_postProcess(output_file, h_outputImageGray);
  //postProcess(output_file_1, h_outputImageRGBA_1);
  
  //GPU_referenceCalculation(h_inputImageGray_1,h_inputImageGray_2, h_outputImageGray_1,h_outputImageGray_2,numRows, numCols,h_filter, filterWidth);
  
  //GPU_postProcess(output_file_1, h_outputImageGray_1);
  //CPU_postProcess(output_file_2, h_outputImageGray_2);
  
  
  //CPU_postProcess(reference_file, h_outputImageRGBA);
    //  Cheater easy way with OpenCV
    //generateReferenceImage(input_file, reference_file, filterWidth);
  //compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);
  
  //checkCudaErrors(cudaFree(d_inputImageGray_1));
  //checkCudaErrors(cudaFree(d_inputImageGray_2));
  //checkCudaErrors(cudaFree(d_outputImageGray));
  //checkCudaErrors(cudaFree(d_outputImageGray__));
  
  GPU_cleanUp();
}


//***********************
void CPU_Part(int argc, char **argv)
{
  
  
  uchar *h_inputImageGray_1,*h_inputImageGray_2;
  uchar *h_outputImageGray_1,*h_outputImageGray_2;
  //unsigned char *d_GrayBlurred,*d_GrayBlurred_1,*d_redBlurred,*d_redBlurred_1, *d_greenBlurred,*d_greenBlurred_1, *d_blueBlurred,*d_blueBlurred_1;

  float *h_filter;
  int    filterWidth;

  std::string input_file_1;
  std::string input_file_2;
  std::string output_file_1;
  std::string output_file_2;
  std::string reference_file;
  double perPixelError = 0.0;
  double globalError   = 0.0;
  bool useEpsCheck = false;
  switch (argc)
  {
	case 2:
	  input_file_1 = std::string(argv[1]);
	  output_file_1 = "nwcc_output_1.png";
	  output_file_2 = "nwcc_output_2.png";
	  reference_file = "nwcc_reference.png";
	  break;
	case 3:
	  input_file_1  = std::string(argv[1]);
	  input_file_2 = std::string(argv[2]);
	  output_file_1 = "nwcc_output_1.png";
	  output_file_2 = "nwcc_output_2.png";
	  reference_file = "nwcc_reference.png";
	  break;
	case 4:
	  input_file_1  = std::string(argv[1]);
          output_file_1 = std::string(argv[2]);
	  reference_file = std::string(argv[3]);
	  break;
	case 6:
	  useEpsCheck=true;
	  input_file_1  = std::string(argv[1]);
	  output_file_1 = std::string(argv[2]);
	  reference_file = std::string(argv[3]);
	  perPixelError = atof(argv[4]);
      globalError   = atof(argv[5]);
	  break;
	default:
      std::cerr << "Usage: ./nwcc input_file_1  input_file_2  [output_filename] [reference_filename] [perPixelError] [globalError]" << std::endl;
      exit(1);
  }
  CPU_Preprocess(&h_inputImageGray_1,&h_inputImageGray_2,
		 &h_outputImageGray_1, &h_outputImageGray_2,
                 &h_filter, &filterWidth, input_file_1,input_file_2);
  
  
  CPU_referenceCalculation(h_inputImageGray_1,h_inputImageGray_2, h_outputImageGray_1,h_outputImageGray_2,
                       numRows, numCols,
                       h_filter, filterWidth);
  
  CPU_postProcess(output_file_1, h_outputImageGray_1);
  //CPU_postProcess(output_file_2, h_outputImageGray_2);
  
  
  //CPU_postProcess(reference_file, h_outputImageRGBA);
    //  Cheater easy way with OpenCV
    //generateReferenceImage(input_file, reference_file, filterWidth);
  //compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);
  
  CPU_cleanUp(&h_inputImageGray_1,&h_inputImageGray_2, &h_outputImageGray_1,&h_outputImageGray_2);
}
/*******  Begin main *********/

int main(int argc, char **argv) {
  
  GPU_Part(argc,argv);
  //CPU_Part(argc,argv);
  

  return 0;
}
