#include "cascadeclassifier.h"

#include <iostream>

CascadeClassifier::CascadeClassifier()
{
}

CascadeClassifier::~CascadeClassifier()
{
}

void CascadeClassifier::train(const std::vector<std::vector<double> > &positiveSampleSet,
			      const std::vector<std::vector<double> > &negativeSampleSet,
			      const std::vector<Haar> &candidateSet,
			      const std::vector<std::vector<double> > &validatePositiveSampleSet,
			      //const std::vector<std::vector<double> > &validateNegativeSampleSet,
			      double minDetectionRate,
			      double maxFalsePositiveRate,
			      double maxTotalFalsePositiveRate,
			      int maxCascadeNum)
{
    size_t classifierNum = candidateSet.size();
    size_t positiveSampleNum = positiveSampleSet[0].size();
    size_t negativeSampleNum = negativeSampleSet[0].size();
    //size_t validatePositiveSampleNum = validatePositiveSampleSet[0].size();
    //size_t validateNegativeSampleNum = validateNegativeSampleSet[0].size();
    
    // 検証用サンプルセットの作成
    /*std::vector<std::vector<double> > validateSampleSet(validatePositiveSampleSet);
    for (size_t i = 0; i < classifierNum; ++i) {
	//std::copy(validateNegativeSampleSet[i].begin(), validateNegativeSampleSet[i].end(), std::back_inserter(validateSampleSet[i]));
	std::copy(validatePositiveSampleSet[i].begin(), validatePositiveSampleSet[i].end(), std::back_inserter(validateSampleSet[i]));
    }
    size_t validateSampleNum = validateSampleSet[0].size();*/
    
    // 検証用ラベルセットの作成
    /*std::vector<int> validateLabelSet(validateSampleNum);
    for (size_t i = 0; i < validatePositiveSampleNum; ++i) {
	validateLabelSet[i] = 1;
    }
    for (size_t i = validatePositiveSampleNum; i < validateSampleNum; ++i) {
	validateLabelSet[i] = -1;
    }*/
    
    cascade_.clear();
    
    double prevDetectionRate = 1.0;
    double prevFalsePositiveRate = 1.0;
    
    std::vector<std::vector<double> > trainNegativeSampleSet(negativeSampleSet);
    size_t trainNegativeSampleNum = trainNegativeSampleSet[0].size();
    /*std::cout << "classifierNum: " << trainNegativeSampleSet.size() << std::endl;
    std::cout << "sampleNum: " << trainNegativeSampleSet[0].size() << std::endl;
    for (size_t i = 0; i < trainNegativeSampleSet[0].size(); ++i) {
	std::cout << trainNegativeSampleSet[0][i] << " ";
    }
    std::cout << std::endl;*/
    
    // 学習の開始
    for (int cascadeNum = 0; cascadeNum < maxCascadeNum; ++cascadeNum)
    {
	double currentFalsePositiveRate = prevFalsePositiveRate;
	
	// 学習用サンプルセットの作成
	std::vector<std::vector<double> > trainSampleSet(positiveSampleSet);
	for (size_t i = 0; i < classifierNum; ++i) {
	    std::copy(trainNegativeSampleSet[i].begin(), trainNegativeSampleSet[i].end(), std::back_inserter(trainSampleSet[i]));
	}
	size_t trainSampleNum = trainSampleSet[0].size();
	/*std::cout << "classifierNum: " << sampleSet.size() << std::endl;
	std::cout << "sampleNum: " << sampleSet[0].size() << std::endl;
	for (size_t i = 0; i < sampleSet[0].size(); ++i) {
	    std::cout << sampleSet[0][i] << " ";
	}
	std::cout << std::endl;*/
	
	// 学習用ラベルセットの作成
	std::vector<int> labelSet(trainSampleNum);
	for (size_t i = 0; i < positiveSampleNum; ++i) {
	    labelSet[i] = 1;
	}
	for (size_t i = positiveSampleNum; i < trainSampleNum; ++i) {
	    labelSet[i] = -1;
	}
	/*std::cout << "sampleNum: " << labelSet.size() << std::endl;
	for (size_t i = 0; i < trainSampleNum; ++i) {
	    std::cout << labelSet[i] << " ";
	}
	std::cout << std::endl;*/
	
	// 層における学習
	AdaBoost model;
	model.initialize(classifierNum, trainSampleNum, trainSampleSet, labelSet);
	
	for (int featureNum = 0; featureNum < classifierNum; ++featureNum)
	//for (int featureNum = 0; featureNum < 6; ++featureNum)
	{
	    //model.train(featureNum, trainSampleSet, labelSet, candidateSet);
	    model.trainOnce(trainSampleSet, labelSet, candidateSet);
	    
	    double targetDetectionRate = minDetectionRate * prevDetectionRate;
	    std::cout << "TargetDetectionRate: " << targetDetectionRate
		      << "\tTagetFalsePositiveRate: " << maxFalsePositiveRate * prevFalsePositiveRate << std::endl;
	    
	    //std::pair<double, double> result = model.adjustThreshold(validateSampleSet, validateLabelSet, targetDetectionRate);
	    std::pair<double, double> result = model.adjustThreshold(validatePositiveSampleSet, trainSampleSet, labelSet, targetDetectionRate);
	    double currentDetectionRate = result.first;
	    currentFalsePositiveRate = result.second;
	    
	    if (currentFalsePositiveRate <= maxFalsePositiveRate * prevFalsePositiveRate)
	    {
		prevDetectionRate = currentDetectionRate;
		prevFalsePositiveRate = currentFalsePositiveRate;
		break;
	    }
	}
	
	// 学習した識別器を追加
	cascade_.push_back(model);
	
	std::cout << std::endl << "currentFalsePositiveRate: " << currentFalsePositiveRate << std::endl << std::endl;
	
	if (currentFalsePositiveRate > maxTotalFalsePositiveRate)
	{
	    std::vector<std::vector<double> > nextNegativeSampleSet(classifierNum);
	    
	    for (size_t i = 0; i < trainNegativeSampleNum; ++i)
	    {
		if (model.classify(i, trainNegativeSampleSet) == 1)
		{
		    for (size_t j = 0; j < classifierNum; ++j)
		    {
			nextNegativeSampleSet[j].push_back(trainNegativeSampleSet[j][i]);
		    }
		}
	    }
	    
	    /*for (size_t i = 0; i < classifierNum; ++i) {
		for (size_t j = 0; j < nextNegativeSampleSet[i].size(); ++j) {
		    std::cout << nextNegativeSampleSet[i][j] << " ";
		}
		std::cout << std::endl;
	    }*/
	    
	    trainNegativeSampleNum = nextNegativeSampleSet[0].size();
	    for (size_t i = 0; i < classifierNum; ++i) {
		trainNegativeSampleSet[i].clear();
		trainNegativeSampleSet[i].reserve(trainNegativeSampleNum);
		std::copy(nextNegativeSampleSet[i].begin(), nextNegativeSampleSet[i].end(), std::back_inserter(trainNegativeSampleSet[i]));
	    }
	    std::cout << "trainNegativeSampleNum: " << trainNegativeSampleNum << std::endl;
	    
	    if (trainNegativeSampleNum == 0) {
		break;
	    }
	    
	} else {
	    break;
	}
    }
}
