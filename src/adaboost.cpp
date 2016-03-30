#include "adaboost.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <limits>
#include <cmath>
#include <algorithm>

#include "haar.h"
#include "filelib.h"

const int Category[2] = {1, -1};

AdaBoost::AdaBoost()
    : classifierNum_(0), sampleNum_(0), threshold_(0.0)
{
}

AdaBoost::~AdaBoost()
{

}

void AdaBoost::initialize(int classifierNum, int sampleNum,
			  const std::vector<std::vector<double> > &sampleSet,
			  const std::vector<int> &labelSet)
{
    classifierNum_ = classifierNum;
    sampleNum_ = sampleNum;
    
    classifier_.clear();
    classifier_.reserve(classifierNum_);
    alpha_.clear();
    alpha_.reserve(classifierNum_);
    weight_.resize(sampleNum_);
    
    // 重みの初期化
    initializeWeight(labelSet);
    
    // 整列済みサンプルの作成
    sortedSampleSet_.resize(sampleSet.size());
    for (size_t i = 0; i < sampleSet.size(); ++i)
    {
	sortedSampleSet_[i].resize(sampleNum_);
	for (int j = 0; j < sampleNum_; ++j)
	{
	    sortedSampleSet_[i][j] = Sample(sampleSet[i][j], labelSet[j], j);
	}
	std::sort(sortedSampleSet_[i].begin(), sortedSampleSet_[i].end());
    }
}

void AdaBoost::train(const std::vector<std::vector<double> > &sampleSet,
		     const std::vector<int> &labelSet,
		     const std::vector<Haar> &candidateSet,
		     int maxClassifierNum)
{
    std::cout << "---------- Training Start ----------" << std::endl << std::endl;
    
    for (int i = 0; i < maxClassifierNum; ++i)
    {
	trainOnce(sampleSet, labelSet, candidateSet);
    }
}

void AdaBoost::trainOnce(const std::vector<std::vector<double> > &sampleSet,
			 const std::vector<int> &labelSet,
			 const std::vector<Haar> &candidateSet)
{
    // 重みの正規化
    normalizeWeight();
    
    double epsilon = std::numeric_limits<double>::max();
    int idx = -1;
    double parity;
    double threshold;
    for (size_t j = 0; j < candidateSet.size(); ++j)
    {
	double p = 1.0;
	double theta = 0.0;
	double error = evaluateParameter(sampleSet[j], labelSet, sortedSampleSet_[j], p, theta);
	if (error < epsilon)
	{
	    epsilon = error;
	    idx = j;
	    parity = p;
	    threshold = theta;
	}
	
	if (j % (candidateSet.size() / 100) == 0) {
	    int progress = j * 100 / candidateSet.size();
	    std::cout << "training: " << progress << "%\r" << std::flush;
	}
    }
    
    Haar elem(candidateSet[idx]);
    elem.setParity(parity);
    elem.setThreshold(threshold);
    elem.setIndex(idx);
    classifier_.push_back(elem);
    
    double alpha = 1.0 / 2.0 * log((1.0 - epsilon) / epsilon);
    alpha_.push_back(alpha);
    
    for (int j = 0; j < sampleNum_; ++j)
    {
	weight_[j] *= exp(-alpha * static_cast<double>(labelSet[j] * classify(sampleSet[idx][j], parity, threshold)));
    }
    
    std::cout << classifier_.size() << ": classifier: " << idx << " alpha: " << alpha << " error: " << epsilon << std::endl;
    
    savefile("tmp_adaboost.param");
}

std::pair<double, double> AdaBoost::adjustThreshold(const std::vector<std::vector<double> > &validateSampleSet,
						    const std::vector<std::vector<double> > &trainSampleSet,
						    const std::vector<int> &labelSet,
						    double targetDetectionRate)
{
    // 顔サンプルと非顔サンプルの数の確認
    size_t positiveSampleNum = 0, negativeSampleNum = 0;
    for (size_t i = 0; i < labelSet.size(); ++i) {
	if (labelSet[i] == Category[0]) {
	    positiveSampleNum++;
	} else {
	    negativeSampleNum++;
	}
    }
    
    size_t validateSampleNum = validateSampleSet[0].size();
    
    size_t truePositiveNum = 0, falsePositiveNum = 0,
           trueNegativeNum = 0, falseNegativeNum = 0;
    
    std::vector<double> sortedValue;
    sortedValue.reserve(validateSampleNum);
    
    for (size_t i = 0; i < validateSampleNum; ++i)
    {
	double value = 0.0;
	
	for (size_t j = 0; j < classifier_.size(); ++j)
	{
	    int idx = classifier_[j].index();
	    if (validateSampleSet[idx][i] * classifier_[j].parity() > classifier_[j].threshold())
	    {
		value += alpha_[j];
	    }
	    else
	    {
		value -= alpha_[j];
	    }
	}
	
	sortedValue.push_back(value);
	
	if (value >= 0.0) {
	    truePositiveNum++;
	} else {
	    falseNegativeNum++;
	}
    }
    
    double detectionRate = static_cast<double>(truePositiveNum) / static_cast<double>(validateSampleNum);
    std::cout << "detectionRate: " << detectionRate << std::endl;
    
    std::sort(sortedValue.begin(), sortedValue.end(), std::greater<double>());
    size_t targetPositiveNum = static_cast<size_t>(std::ceil(targetDetectionRate * validateSampleNum));
    std::cout << "targetPositiveNum: " << targetPositiveNum << std::endl;
    
    threshold_ = sortedValue[targetPositiveNum - 1];
    std::cout << "threshold: " << threshold_ << std::endl;
    
    truePositiveNum = 0; falsePositiveNum = 0;
    trueNegativeNum = 0; falseNegativeNum = 0;
    
    for (size_t i = 0; i < labelSet.size(); ++i)
    {
	double value = 0.0;
	
	for (size_t j = 0; j < classifier_.size(); ++j)
	{
	    int idx = classifier_[j].index();
	    if (trainSampleSet[idx][i] * classifier_[j].parity() > classifier_[j].threshold())
	    {
		value += alpha_[j];
	    }
	    else
	    {
		value -= alpha_[j];
	    }
	}
	
	if (labelSet[i] == Category[0]) {
	    if (value >= threshold_) {
		truePositiveNum++;
	    } else {
		falseNegativeNum++;
	    }
	} else {
	    if (value >= threshold_) {
		falsePositiveNum++;
	    } else {
		trueNegativeNum++;
	    }
	}
    }
    
    detectionRate = static_cast<double>(truePositiveNum) / static_cast<double>(positiveSampleNum);
    double falsePositiveRate = static_cast<double>(falsePositiveNum) / static_cast<double>(negativeSampleNum);
    std::cout << "      DetectionRate: " << detectionRate << "\t";
    std::cout << "FalsePositiveRate: " << falsePositiveRate << std::endl;
    std::cout << std::endl;
    
    return std::pair<double, double>(detectionRate, falsePositiveRate);
}

int AdaBoost::classify(int index, const std::vector<std::vector<double> > &sampleSet)
{
    double value = 0.0;
    
    for (size_t i = 0; i < classifier_.size(); ++i)
    {
	int idx = classifier_[i].index();
	if (sampleSet[idx][index] * classifier_[i].parity() > classifier_[i].threshold())
	{
	    value += alpha_[i];
	}
	else
	{
	    value -= alpha_[i];
	}
    }
    
    if (value >= threshold_) {
	return 1;
    } else {
	return -1;
    }
}

int AdaBoost::classify(double const * const * image)
{
    double value = 0.0;
    for (int i = 0; i < classifier_.size(); ++i)
    {
	if (classifier_[i].classify(image))
	{
	    value += alpha_[i];
	}
	else
	{
	    value -= alpha_[i];
	}
    }
    
    if (value >= threshold_) {
	return Category[0];
    } else {
	return Category[1];
    }
}

void AdaBoost::loadfile(const char *filename)
{
    std::ifstream fin(filename);
    
    load(fin);
    
    fin.close();
}

void AdaBoost::savefile(const char *filename)
{
    std::ofstream fout(filename);
    
    save(fout);
    
    fout.close();
}

void AdaBoost::load(std::ifstream &fin)
{
    classifier_.clear();
    alpha_.clear();
    
    int count = 0;
    fin >> count;
    fin.ignore(256, '\n');
    classifier_.resize(count);
    alpha_.resize(count);
    
    fin >> threshold_;
    fin.ignore(256, '\n');
    
    for (int i = 0; i < count; ++i) {
	fin >> alpha_[i];
    }
    fin.ignore(256, '\n');
    
    for (int i = 0; i < count; ++i) {
	classifier_[i].load(fin);
    }
    fin.ignore(256, '\n');
}

void AdaBoost::save(std::ofstream &fout)
{
    fout << classifier_.size() << std::endl;
    fout << threshold_ << std::endl;
    
    for (size_t i = 0; i < classifier_.size(); ++i) {
	fout << alpha_[i] << " ";
    }
    fout << std::endl;
    
    for (size_t i = 0; i < classifier_.size(); ++i) {
	classifier_[i].save(fout);
    }
    fout << std::endl;
}

void AdaBoost::initializeWeight(const std::vector<int> &labelSet)
{
    /*int sampleNumPerCategory[categoryNum_] = {0, 0};
    for (int i = 0; i < sampleNum_; ++i)
    {
	if (labelSet[i] == category_[0]) {
	    sampleNumPerCategory[0]++;
	} else {
	    sampleNumPerCategory[1]++;
	}
    }
    
    for (int i = 0; i < sampleNum_; ++i)
    {
	if (labelSet[i] == 1) {
	    weight_[i] = 1.0 / static_cast<double>(2 * sampleNumPerCategory[0]);
	} else {
	    weight_[i] = 1.0 / static_cast<double>(2 * sampleNumPerCategory[1]);
	}
    }*/
    
    for (int i = 0; i < sampleNum_; ++i)
    {
	weight_[i] = 1.0 / static_cast<double>(sampleNum_);
    }
}

void AdaBoost::normalizeWeight()
{
    double sum = 0.0;
    for (int i = 0; i < sampleNum_; ++i) {
	sum += weight_[i];
    }
    for (int i = 0; i < sampleNum_; ++i) {
	weight_[i] /= sum;
    }
}

int AdaBoost::classify(double value, double parity, double threshold)
{
    if (value * parity > parity * threshold) {
	return Category[0];
    } else {
	return Category[1];
    }
}

double AdaBoost::evaluateParameter(const std::vector<double> &sample,
				   const std::vector<int> &labelSet,
				   const std::vector<Sample> &sortedSampleSet,
				   double &parity, double &threshold)
{
    double epsilon = std::numeric_limits<double>::max();
    for (int i = 0; i < sampleNum_ - 1; ++i)
    {
	if (sortedSampleSet[i].category == sortedSampleSet[i + 1].category) {
	    continue;
	}
	if (sortedSampleSet[i].value == sortedSampleSet[i + 1].value) {
	    continue;
	}
	
	double p = 1.0;
	double theta = (sortedSampleSet[i].value + sortedSampleSet[i + 1].value) / 2.0;
	
	double error = 0.0;
	for (int j = 0; j < sampleNum_; ++j)
	{
	    if (labelSet[j] != classify(sample[j], p, theta))
	    {
		error += weight_[j];
	    }
	}
	
	if (error > 0.5) {
	    error = 1.0 - error;
	    p = -1.0;
	}
	
	if (error < epsilon) {
	    epsilon = error;
	    parity = p;
	    threshold = theta;
	}
    }
    
    return epsilon;
}
