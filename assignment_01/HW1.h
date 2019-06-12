#ifndef HW1_H
#define HW1_H

#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

extern cv::Mat imageRGBA;
extern cv::Mat imageGrey;

extern uchar4        *d_rgbaImage__;
extern unsigned char *d_greyImage__;

size_t numRows();
size_t numCols();

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, unsigned char **greyImage,
	uchar4 **d_rgbaImage, unsigned char **d_greyImage,
	const std::string &filename);

void postProcess(const std::string& output_file, unsigned char* data_ptr);

void cleanup();

void generateReferenceImage(std::string input_filename, std::string output_filename);
#endif
