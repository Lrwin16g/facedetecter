#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "haar.h"
#include "filelib.h"

const int categoryNum = 2;
const int category[2] = {1, -1};

int classify(const std::vector<Haar*> &haarFeatures, double **image)
{
    double value = 0.0;
    for (int i = 0; i < haarFeatures.size(); ++i)
    {
	if (haarFeatures[i]->classify(image))
	{
	    value += haarFeatures[i]->weight();
	}
	else
	{
	    value -= haarFeatures[i]->weight();
	}
    }
    
    if (value >= 0.0) {
	return category[0];
    } else {
	return category[1];
    }
}

int main(int argc, char *argv[])
{
    if (argc != 4) {
	std::cerr << "Usage: " << argv[0] << " <param> <facelist> <nonfacelist>" << std::endl;
	return -1;
    }
    
    std::vector<Haar*> haarFeatures = loadHaarFeatures(argv[1]);
    
    int classifierNum = haarFeatures.size();
    
    std::vector<std::vector<std::string> > fileList;
    fileList.push_back(file::loadfile(argv[2]));
    fileList.push_back(file::loadfile(argv[3]));
    
    int sampleNum = fileList[0].size() + fileList[1].size();
    int faceList = fileList[0].size();
    int nonfaceList = fileList[1].size();
    
    std::vector<cv::Mat> image(sampleNum);
    int *label = new int[sampleNum];
    for (int i = 0; i < fileList.size(); ++i) {
	for (int j = 0; j < fileList[i].size(); ++j)
	{
	    image[i * fileList[0].size() + j] = cv::imread(fileList[i][j], 0);
	    label[i * fileList[0].size() + j] = category[i];
	}
    }
    
    int width = image[0].cols;
    int height = image[0].rows;
    
    double **src = new double*[height];
    double **dst = new double*[height];
    for (int i = 0; i < height; ++i) {
	src[i] = new double[width];
	dst[i] = new double[width];
    }
    
    int count = 0;
    for (int i = 0; i < sampleNum; ++i) {
	for (int y = 0; y < height; ++y) {
	    for (int x = 0; x < width; ++x) {
		src[y][x] = static_cast<double>(image[i].at<uchar>(y, x));
	    }
	}
	createIntegralImage(src, dst, width, height);
	
	if (label[i] == classify(haarFeatures, dst))
	{
	    count++;
	}
    }
    
    double accuracy = static_cast<double>(count) / static_cast<double>(sampleNum);
    
    std::cout << accuracy << std::endl;
    
    for (int i = 0; i < height; ++i) {
	delete[] src[i];
	delete[] dst[i];
    }
    delete[] src;
    delete[] dst;
    
    delete[] label;
    
    for (int i = 0; i < classifierNum; ++i) {
	delete haarFeatures[i];
    }
    
    return 0;
}
