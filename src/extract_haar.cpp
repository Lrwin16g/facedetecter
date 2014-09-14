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
    int sizeStep = 2;
    
    std::vector<Haar*> haarFeatures = createHaarFeatures(width, height, scanStep, sizeStep);
    
    std::cout << "haarFeature: " << haarFeatures.size() << std::endl;
    saveHaarFeatures(argv[3], haarFeatures);
    
    double **src = new double*[height];
    double **dst = new double*[height];
    for (int i = 0; i < height; ++i) {
	src[i] = new double[width];
	dst[i] = new double[width];
    }
    
    size_t classifierNum = haarFeatures.size();
    
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
	    sample[j][i] = haarFeatures[j]->extract(dst);
	}
	
	if (i % 100 == 0)
	{
	    std::cout << "extracting: " << i << "/" << sampleNum << std::endl;
	}
    }
    
    std::ofstream fout(argv[4]);
    for (size_t i = 0; i < classifierNum; ++i)
    {
	for (size_t j = 0; j < sampleNum; ++j)
	{
	    fout << sample[i][j] << " ";
	}
	fout << std::endl;
	if (i % 100 == 0)
	{
	    std::cout << "saving: " << i << "/" << classifierNum << std::endl;
	}
    }
    for (size_t i = 0; i < sampleNum; ++i)
    {
	fout << label[i] << " ";
    }
    fout.close();
    
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
    
    for (size_t i = 0; i < haarFeatures.size(); ++i) {
	delete haarFeatures[i];
    }
    
    return 0;
}
