#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "adaboost.h"
#include "haar.h"
#include "filelib.h"

const int Category[2] = {1, -1};

int main(int argc, char *argv[])
{
    if (argc != 4) {
	std::cerr << "Usage: " << argv[0] << " <in:haar-param> <in:test-face-list> <in:test-non-face-list>" << std::endl;
	return -1;
    }
    
    AdaBoost model;
    model.loadfile(argv[1]);
    
    std::vector<std::vector<std::string> > fileList;
    fileList.push_back(file::loadfile(argv[2]));
    fileList.push_back(file::loadfile(argv[3]));
    
    int sampleNum = fileList[0].size() + fileList[1].size();
    int positiveSampleList = fileList[0].size();
    int negativeSampleList = fileList[1].size();
    
    std::vector<cv::Mat> image(sampleNum);
    int *label = new int[sampleNum];
    for (int i = 0; i < fileList.size(); ++i) {
	for (int j = 0; j < fileList[i].size(); ++j)
	{
	    image[i * fileList[0].size() + j] = cv::imread(fileList[i][j], 0);
	    label[i * fileList[0].size() + j] = Category[i];
	}
    }
    
    int width = image[0].cols;
    int height = image[0].rows;
    
    double **src = new double*[height];
    for (int i = 0; i < height; ++i) {
	src[i] = new double[width];
    }
    double **dst = new double*[height + 1];
    for (int i = 0; i < height + 1; ++i) {
	dst[i] = new double[width + 1];
    }
    
    int count = 0;
    for (int i = 0; i < sampleNum; ++i) {
	for (int y = 0; y < height; ++y) {
	    for (int x = 0; x < width; ++x) {
		src[y][x] = static_cast<double>(image[i].at<uchar>(y, x));
	    }
	}
	createIntegralImage(src, dst, width, height);
	
	if (label[i] == model.classify(dst))
	{
	    count++;
	}
    }
    
    double accuracy = static_cast<double>(count) / static_cast<double>(sampleNum);
    
    std::cout << "accuracy: " << accuracy * 100 << "%" << std::endl;
    
    for (int i = 0; i < height; ++i) {
	delete[] src[i];
    }
    for (int i = 0; i < height + 1; ++i) {
	delete[] dst[i];
    }
    delete[] src;
    delete[] dst;
    
    delete[] label;
    
    return 0;
}
