#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "haar.h"
#include "filelib.h"

// 顔:1、非顔:-1
const int categoryNum = 2;
const int category[2] = {1, -1};

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
	std::cerr << "Usage: " << argv[0] << " <[in]file-list> <[in]haar-param> <[out]dat-file>" << std::endl;
	return -1;
    }
    
    // 入力画像リストの読込み
    std::vector<std::string> fileList = file::loadfile(argv[1]);
    size_t sampleNum = fileList.size();
    
    // 画像の読込み、配列の作成
    std::vector<cv::Mat> imageList;
    for (size_t i = 0; i < sampleNum; ++i)
    {
	cv::Mat image = cv::imread(fileList[i], 0);
	cv::equalizeHist(image, image);
	imageList.push_back(image);
    }
    
    int width = imageList[0].cols;
    int height = imageList[0].rows;
    std::cout << "width: " << width << " height: " << height << std::endl;
    
    // Haar-like特徴量の読込み
    std::vector<Haar> haar = loadHaarFeatures(argv[2]);
    size_t classifierNum = haar.size();
    std::cout << "HaarNum: " << classifierNum << std::endl;
    
    // 特徴量抽出の開始
    double **src = new double*[height];
    double **dst = new double*[height + 1];
    for (int i = 0; i < height; ++i) {
	src[i] = new double[width];
    }
    for (int i = 0; i < height + 1; ++i) {
	dst[i] = new double[width + 1];
    }
    
    std::vector<std::vector<double> > sample(classifierNum);
    for (size_t i = 0; i < classifierNum; ++i) {
	sample[i].resize(sampleNum);
    }
    
    // Haar-like特徴量の抽出
    for (size_t i = 0; i < sampleNum; ++i)
    {
	for (int y = 0; y < height; ++y)
	{
	    for (int x = 0; x < width; ++x)
	    {
		src[y][x] = static_cast<double>(imageList[i].at<uchar>(y, x));
	    }
	}
	// 積分画像の作成
	createIntegralImage(src, dst, width, height);
	
	for (size_t j = 0; j < classifierNum; ++j)
	{
	    sample[j][i] = haar[j].extract(dst);
	}
	
	if (i % 100 == 0)
	{
	    std::cout << "extracting: " << i << "/" << sampleNum << "\r" << std::flush;
	}
    }
    std::cout << std::endl;
    
    // 特徴量抽出結果の保存
    file::savemat(argv[3], sample, true);
    
    std::cout << "sample: " << sampleNum << std::endl;
    std::cout << "classifier: " << classifierNum << std::endl;
    
    for (int i = 0; i < height; ++i) {
	delete[] src[i];
    }
    for (int i = 0; i < height + 1; ++i) {
	delete[] dst[i];
    }
    delete[] src; src = NULL;
    delete[] dst; dst = NULL;
    
    return 0;
}
