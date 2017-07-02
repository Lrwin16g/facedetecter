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
    cv::Mat src = cv::imread(argv[1]);
    cv::Mat image;
    cv::cvtColor(src, image, CV_RGB2GRAY);
    cv::equalizeHist(image, image);
    
    //CascadeClassifier classifier;
    AdaBoost classifier;
    classifier.loadfile(argv[2]);
    classifier.setThreshold(5.0);
    
    int windowWidth = atoi(argv[3]);
    int windowHeight = atoi(argv[4]);
    
    int scanStep = 1;
    
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
    
    for (double scale = 3.0; scale <= 10.0; scale *= 1.1)
    {
	int width = imageWidth / scale;
	int height = imageHeight / scale;
	
	std::cout << height << " " << width << std::endl;
	
	cv::Mat resizeImage(height, width, image.type());
	cv::resize(image, resizeImage, resizeImage.size(), cv::INTER_CUBIC);
	
	for (int y = 0; (y + windowHeight) < height; y += scanStep)
	{
	    for (int x = 0; (x + windowWidth) < width; x += scanStep)
	    {
		for (int v = 0; v < windowHeight; ++v)
		{
		    for (int u = 0; u < windowWidth; ++u)
		    {
			srcImage[v][u] = static_cast<double>(resizeImage.at<uchar>(y + v, x + u));
		    }
		}
		
		createIntegralImage(srcImage, dstImage, windowWidth, windowHeight);
		
		if (classifier.classify(dstImage) == Category[0])
		{
		    std::cout << "y: " << y << "\tx: " << x << std::endl;
		    cv::rectangle(src, cv::Point(x * scale, y * scale), cv::Point((x + windowWidth) * scale, (y + windowHeight) * scale), cv::Scalar(0,255,0), 1, 4);
		}
	    }
	}
    }
    
    cv::namedWindow("drawing", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
    cv::imshow("drawing", src);
    cv::imwrite("result.bmp", src);
    cv::waitKey(0);
    
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
