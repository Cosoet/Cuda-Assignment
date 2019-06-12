#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

extern cv::Mat imageInputRGBA;
extern cv::Mat imageOutputRGBA;

extern uchar4 *d_inputImageRGBA__;
extern uchar4 *d_outputImageRGBA__;

extern float *h_filter__;

size_t numRows();
size_t numCols();

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
	uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
	unsigned char **d_redBlurred,
	unsigned char **d_greenBlurred,
	unsigned char **d_blueBlurred,
	float **h_filter, int *filterWidth,
	const std::string &filename);

void postProcess(const std::string& output_file, uchar4* data_ptr);

void cleanUp(void);


// An unused bit of code showing how to accomplish this assignment using OpenCV.  It is much faster 
//    than the naive implementation in reference_calc.cpp.
void generateReferenceImage(std::string input_file, std::string reference_file, int kernel_size);