#include "cascadeclassifier.h"

#include <iostream>

CascadeClassifier::CascadeClassifier()
{
}

CascadeClassifier::~CascadeClassifier()
{
}

void CascadeClassifier::train(const std::vector<Haar> &candidateSet,
			      const std::vector<std::vector<double> > &positiveSampleSet,
			      const std::vector<std::vector<double> > &negativeSampleSet,
			      const std::vector<std::vector<double> > &validatePositiveSampleSet,
			      double minDetectionRate,
			      double maxFalsePositiveRate,
			      double maxTotalFalsePositiveRate,
			      int maxCascadeNum)
{
    size_t classifierNum = candidateSet.size();
    size_t positiveSampleNum = positiveSampleSet[0].size();
    size_t negativeSampleNum = negativeSampleSet[0].size();
    size_t validatePositiveSampleNum = validatePositiveSampleSet[0].size();
    
    cascade_.clear();
    
    double prevDetectionRate = 1.0;
    double prevFalsePositiveRate = 1.0;
    
    std::vector<std::vector<double> > trainNegativeSampleSet(negativeSampleSet);
    size_t trainNegativeSampleNum = trainNegativeSampleSet[0].size();
    
    std::cout << "---------- Training Start ----------" << std::endl << std::endl;
    
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
	
	// 学習用ラベルセットの作成
	std::vector<int> labelSet(trainSampleNum);
	for (size_t i = 0; i < positiveSampleNum; ++i) {
	    labelSet[i] = 1;
	}
	for (size_t i = positiveSampleNum; i < trainSampleNum; ++i) {
	    labelSet[i] = -1;
	}
	
	// 層における学習
	AdaBoost model;
	model.initialize(classifierNum, trainSampleNum, trainSampleSet, labelSet);
	
	for (int featureNum = 0; featureNum < classifierNum; ++featureNum)
	{
	    model.trainOnce(trainSampleSet, labelSet, candidateSet);
	    
	    double targetDetectionRate = minDetectionRate * prevDetectionRate;
	    std::cout << "TargetDetectionRate: " << targetDetectionRate
		      << "\tTagetFalsePositiveRate: " << maxFalsePositiveRate * prevFalsePositiveRate << std::endl;
	    
	    std::pair<double, double> result = model.adjustThreshold(validatePositiveSampleSet, trainSampleSet, labelSet, targetDetectionRate);
	    double currentDetectionRate = result.first;
	    currentFalsePositiveRate = result.second;
	    
	    savefile("tmp_cascade.param");
	    
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
	    
	    trainNegativeSampleNum = nextNegativeSampleSet[0].size();
	    for (size_t i = 0; i < classifierNum; ++i) {
		trainNegativeSampleSet[i].clear();
		trainNegativeSampleSet[i].reserve(trainNegativeSampleNum);
		std::copy(nextNegativeSampleSet[i].begin(), nextNegativeSampleSet[i].end(), std::back_inserter(trainNegativeSampleSet[i]));
	    }
	    std::cout << "trainNegativeSampleNum: " << trainNegativeSampleNum << std::endl;
	    
	    if (trainNegativeSampleNum <= 1) {
		break;
	    }
	    
	} else {
	    break;
	}
    }
    
    savefile("final_cascade.param");
}

void CascadeClassifier::loadfile(const char *filename)
{
    std::ifstream fin(filename);
    
    cascade_.clear();
    int cascadeNum = 0;
    fin >> cascadeNum;
    fin.ignore(256, '\n');
    cascade_.resize(cascadeNum);
    
    for (int i = 0; i < cascadeNum; ++i) {
	cascade_[i].load(fin);
    }
    
    fin.close();
}

void CascadeClassifier::savefile(const char *filename)
{
    std::ofstream fout(filename);
    
    fout << cascade_.size() << std::endl;
    
    for (size_t i = 0; i < cascade_.size(); ++i) {
	cascade_[i].save(fout);
    }
    
    fout.close();
}
