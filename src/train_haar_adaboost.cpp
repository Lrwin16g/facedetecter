#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

#include "haar.h"
#include "adaboost.h"
#include "filelib.h"

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
	std::cerr << "Usage: " << argv[0] << " <in:haar-param> <in:train-face-dat> <in:train-nonface-dat>" << std::endl;
	return -1;
    }
    
    // Haar-like特徴量の読込み
    std::cout << std::endl << "Loading Haar-like Feature Parameter..." << std::endl;
    std::vector<Haar> candidateSet = loadHaarFeatures(argv[1]);
    size_t classifierNum = candidateSet.size();
    std::cout << "classifierNum: " << classifierNum << std::endl << std::endl;
    
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
    
    // 学習用サンプルセットの作成
    std::vector<std::vector<double> > trainSampleSet(trainPositiveSampleSet);
    for (size_t i = 0; i < classifierNum; ++i) {
	std::copy(trainNegativeSampleSet[i].begin(), trainNegativeSampleSet[i].end(), std::back_inserter(trainSampleSet[i]));
    }
    size_t trainSampleNum = trainSampleSet[0].size();
    
    // 学習用ラベルセットの作成
    std::vector<int> labelSet(trainSampleNum);
    for (size_t i = 0; i < trainPositiveSampleNum; ++i) {
	labelSet[i] = 1;
    }
    for (size_t i = trainPositiveSampleNum; i < trainSampleNum; ++i) {
	labelSet[i] = -1;
    }
    
    // AdaBoostによる学習
    AdaBoost model;
    model.initialize(classifierNum, trainSampleNum, trainSampleSet, labelSet);
    //model.train(300, trainPositiveSampleSet, trainNegativeSampleSet, candidateSet);
    model.train(trainSampleSet, labelSet, candidateSet, 300);
    
    return 0;
}
