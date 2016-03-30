#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "cascadeclassifier.h"
#include "haar.h"

const int Category[2] = {1, -1};

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
	std::cerr << "Usage: " << argv[0] << " <[in]image-data> <[in]classifier-param> <[in]width> <[in]height>" << std::endl;
	return -1;
    }
    
    // 入力画像の読込み
    cv::Mat image = cv::imread(argv[1], 0);
    
    CascadeClassifier classifier;
    classifier.loadfile(argv[2]);
    
    int windowWidth = atoi(argv[3]);
    int windowHeight = atoi(argv[4]);
    
    int scanStep = 2;
    
    int imageWidth = image.cols;
    int imageHeight = image.rows;
    
    //std::cout << imageWidth << " " << imageHeight << std::endl;
    
    double **srcImage = new double*[windowHeight];
    for (int i = 0; i < windowHeight; ++i) {
	srcImage[i] = new double[windowWidth];
    }
    double **dstImage = new double*[windowHeight + 1];
    for (int i = 0; i < windowHeight + 1; ++i) {
	dstImage[i] = new double[windowWidth + 1];
    }
    
    /*double **windowImage = new double*[windowHeight + 1];
    for (int i = 0; i < windowHeight + 1; ++i) {
	windowImage[i] = new double[windowWidth + 1];
    }*/
    
    for (double scale = 1.0; scale < 10; scale += 2.0)
    {
	
	int width = windowWidth * scale;
	int height = windowHeight * scale;
	
	std::cout << height << " " << width << std::endl;
	
	for (int y = 0; (y + height) < imageHeight; y += scanStep)
	{
	    for (int x = 0; (x + width) < imageWidth; x += scanStep)
	    {
		for (int v = 0; v < windowHeight; ++v)
		{
		    for (int u = 0; u < windowWidth; ++u)
		    {
			srcImage[v][u] = static_cast<double>(image.at<uchar>(y + static_cast<int>(v * scale), x + static_cast<int>(u * scale)));
		    }
		}
		
		createIntegralImage(srcImage, dstImage, windowWidth, windowHeight);
		
		if (classifier.classify(dstImage) == Category[0])
		{
		    std::cout << "y: " << y << "\tx: " << x << std::endl;
		    cv::rectangle(image, cv::Point(x, y), cv::Point(x + width, y + height), cv::Scalar(0,0,200), 1, 4);
		}
	    }
	}
    }
    
    cv::namedWindow("drawing", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
    cv::imshow("drawing", image);
    cv::waitKey(0);
    
    /*for (int i = 0; i < windowHeight + 1; ++i) {
	delete[] windowImage[i];
    }
    delete[] windowImage;  windowImage = NULL;*/
    
    for (int i = 0; i < windowHeight; ++i) {
	delete[] srcImage[i];
    }
    for (int i = 0; i < windowHeight + 1; ++i) {
	delete[] dstImage[i];
    }
    delete[] srcImage;	srcImage = NULL;
    delete[] dstImage;	dstImage = NULL;
    
    return 0;
}
