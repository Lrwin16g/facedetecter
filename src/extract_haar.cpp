#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "haar.h"
#include "filelib.h"

const int categoryNum = 2;
const int category[2] = {1, -1};

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
	std::cerr << "Usage: " << argv[0] << " <face-list> <nonface-list> <haar-param> <haar-dat>" << std::endl;
	return -1;
    }
    
    std::vector<std::vector<std::string> > fileList(categoryNum);
    for (int i = 0; i < categoryNum; ++i)
    {
	fileList[i] = file::loadfile(argv[i + 1]);
    }
    
    size_t sampleNum = fileList[0].size() + fileList[1].size();
    
    std::vector<cv::Mat> imageList;
    int *label = new int[sampleNum];
    for (int i = 0; i < categoryNum; ++i)
    {
	for (size_t j = 0; j < fileList[i].size(); ++j)
	{
	    imageList.push_back(cv::imread(fileList[i][j], 0));
	    label[i * fileList[0].size() + j] = category[i];
	}
    }
    
    int width = imageList[0].cols;
    int height = imageList[0].rows;
    
    int scanStep = 2;
    int sizeStep = 1;
    
    std::vector<Haar> haar = createHaarFeatures(width, height, scanStep, sizeStep);
    
    std::cout << "haarFeature: " << haar.size() << std::endl;
    saveHaarFeatures(argv[3], haar);
    
    double **src = new double*[height];
    double **dst = new double*[height];
    for (int i = 0; i < height; ++i) {
	src[i] = new double[width];
	dst[i] = new double[width];
    }
    
    size_t classifierNum = haar.size();
    
    double **sample = new double*[classifierNum];
    for (size_t i = 0; i < classifierNum; ++i) {
	sample[i] = new double[sampleNum];
    }
    
    for (size_t i = 0; i < sampleNum; ++i)
    {
	for (int y = 0; y < height; ++y)
	{
	    for (int x = 0; x < width; ++x)
	    {
		src[y][x] = static_cast<double>(imageList[i].at<uchar>(y, x));
	    }
	}
	createIntegralImage(src, dst, width, height);
	
	for (size_t j = 0; j < classifierNum; ++j)
	{
	    sample[j][i] = haar[j].extract(dst);
	}
	
	if (i % 100 == 0)
	{
	    std::cout << "extracting: " << i << "/" << sampleNum << std::endl;
	}
    }
    
    file::savefile(argv[4], sample, classifierNum, sampleNum, true, " ", true);
    
    std::cout << "sample: " << sampleNum << std::endl;
    std::cout << "classifier: " << classifierNum << std::endl;
    
    std::string name = argv[4];
    std::string ext = ".label";
    name.replace(name.rfind(".dat"), ext.length(), ext);
    
    file::savefile(name.c_str(), label, sampleNum, false);
    
    for (size_t i = 0; i < classifierNum; ++i) {
	delete[] sample[i];
    }
    delete[] sample;
    delete[] label;
    
    for (int i = 0; i < height; ++i) {
	delete[] src[i];
	delete[] dst[i];
    }
    delete[] src;
    delete[] dst;
    
    return 0;
}
