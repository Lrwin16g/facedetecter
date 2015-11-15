#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "haar.h"
#include "cascadeclassifier.h"
#include "filelib.h"

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
	std::cerr << "Usage: " << argv[0] << " <in:haar-param> <in:train-face-dat> <in:train-nonface-dat> <in:validate-face-dat>" << std::endl;
	return -1;
    }
    
    // Haar-like特徴量の読込み
    std::cout << std::endl << "Loading Haar-like Feature Parameter..." << std::endl;
    std::vector<Haar> candidateSet = loadHaarFeatures(argv[1]);
    size_t candidateNum = candidateSet.size();
    std::cout << "HaarNum: " << candidateNum << std::endl << std::endl;
    
    // 学習用顔画像特徴量の読込み
    std::cout << "Loading Training Positive Sample Set..." << std::endl;
    std::vector<std::vector<double> > trainPositiveSampleSet = file::loadmat<double>(argv[2]);
    size_t trainPositiveSampleNum = trainPositiveSampleSet[0].size();
    std::cout << "classifierNum: " << trainPositiveSampleSet.size() << std::endl;
    std::cout << "trainPositiveSampleNum: " << trainPositiveSampleNum << std::endl << std::endl;
    
    // 学習用非顔画像特徴量の読込み
    std::cout << "Loading Training Negative Sample Set..." << std::endl;
    std::vector<std::vector<double> > trainNegativeSampleSet = file::loadmat<double>(argv[3]);
    size_t trainNegativeSampleNum = trainNegativeSampleSet[0].size();
    std::cout << "classifierNum: " << trainNegativeSampleSet.size() << std::endl;
    std::cout << "trainNegativeSampleNum: " << trainNegativeSampleNum << std::endl << std::endl;
    
    // 検証用顔画像特徴量の読込み
    std::cout << "Loading Validation Positive Sample Set..." << std::endl;
    std::vector<std::vector<double> > validatePositiveSampleSet = file::loadmat<double>(argv[4]);
    std::cout << "classifierNum: " << validatePositiveSampleSet.size() << std::endl;
    std::cout << "validatePositiveSampleNum: " << validatePositiveSampleSet[0].size() << std::endl << std::endl;
    
    // 検証用非顔画像特徴量の読込み
    /*std::cout << "Loading Validation Negative Sample Set..." << std::endl;
    std::vector<std::vector<double> > validateNegativeSampleSet = file::loadmat<double>(argv[5]);
    std::cout << "classifierNum: " << validateNegativeSampleSet.size() << std::endl;
    std::cout << "validatePositiveSampleNum: " << validateNegativeSampleSet[0].size() << std::endl << std::endl;*/
    
    // 学習用パラメータの設定
    double minDetectionRate = 0.99;
    double maxFalsePositiveRate = 0.5;
    int maxCascadeNum = 10;
    double maxTotalFalsePositiveRate = std::pow(maxFalsePositiveRate, static_cast<double>(maxCascadeNum));
    
    // カスケード識別器の学習
    CascadeClassifier model;
    model.train(trainPositiveSampleSet, trainNegativeSampleSet, candidateSet,
		validatePositiveSampleSet, //validateNegativeSampleSet,
		minDetectionRate, maxFalsePositiveRate, maxTotalFalsePositiveRate,
		maxCascadeNum);
    
    return 0;
}
