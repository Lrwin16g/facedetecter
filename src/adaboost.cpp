#include "adaboost.h"
#include <iostream>
#include <limits>
#include <cmath>

#include "haar.h"

const int categoryNum_ = 2;

AdaBoost::AdaBoost(int classifierNum, int sampleNum)
    : classifierNum_(classifierNum), sampleNum_(sampleNum),
      weight_(NULL), alpha_(NULL), data_(NULL), index_(NULL)
{
    category_[0] = 1;	category_[1] = -1;
    alpha_ = new double[classifierNum_];
    weight_ = new double[sampleNum_];
    data_ = new double[sampleNum_];
    index_ = new int[sampleNum_];
}

AdaBoost::~AdaBoost()
{
    if (alpha_ != NULL) {
	delete[] alpha_;
    }
    if (weight_ != NULL) {
	delete[] weight_;
    }
    if (data_ != NULL) {
	delete[] data_;
    }
    if (index_ != NULL) {
	delete[] index_;
    }
}

void AdaBoost::initializeWeight(const int *label)
{
    int sampleNumPerCategory[categoryNum_] = {0, 0};
    for (int i = 0; i < sampleNum_; ++i)
    {
	if (label[i] == category_[0]) {
	    sampleNumPerCategory[0]++;
	} else {
	    sampleNumPerCategory[1]++;
	}
    }
    
    for (int i = 0; i < sampleNum_; ++i)
    {
	if (label[i] == 1) {
	    weight_[i] = 1.0 / static_cast<double>(2 * sampleNumPerCategory[0]);
	} else {
	    weight_[i] = 1.0 / static_cast<double>(2 * sampleNumPerCategory[1]);
	}
    }
}

void AdaBoost::train(const double * const *sample, const int *label, const std::vector<Haar> &candidate)
{
    for (int i = 0; i < classifierNum_; ++i)
    {
	normalizeWeight();
	
	double epsilon = std::numeric_limits<double>::max();
	int idx = -1;
	double parity;
	double threshold;
	for (size_t j = 0; j < candidate.size(); ++j)
	{
	    double p = 1.0;
	    double theta = 0.0;
	    double error = evaluateParameter(sample[j], label, p, theta);
	    if (error < epsilon)
	    {
		epsilon = error;
		idx = j;
		parity = p;
		threshold = theta;
	    }
	    
	    if (j % 100 == 0) {
		std::cout << j << "/" << candidate.size() << std::endl;
	    }
	}
	
	classifier_.push_back(candidate[idx]);
	classifier_[i].setParity(parity);
	classifier_[i].setThreshold(threshold);
	
	alpha_[i] = 1.0 / 2.0 * log((1.0 - epsilon) / epsilon);
	
	for (int j = 0; j < sampleNum_; ++j)
	{
	    weight_[j] *= exp(-alpha_[i] * static_cast<double>(label[j] * classify(sample[idx][j], parity, threshold)));
	}
	
	std::cout << i << ": classifier: " << idx << " alpha: " << alpha_[i] << " error: " << epsilon << std::endl;
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

void sort_twin(double *array_1, int *array_2, int len)
{
    for (int i = 0; i < len - 1; ++i)
    {
	for (int j = len - 1; j > i; --j)
	{
	    if (array_1[j - 1] > array_1[j])
	    {
		double tmp_1 = array_1[j];
		array_1[j] = array_1[j - 1];
		array_1[j - 1] = tmp_1;
		int tmp_2 = array_2[j];
		array_2[j] = array_2[j - 1];
		array_2[j - 1] = tmp_2;
	    }
	}
    }
}

int AdaBoost::classify(double sample, double parity, double threshold)
{
    if (sample * parity > parity * threshold) {
	return category_[0];
    } else {
	return category_[1];
    }
}

double AdaBoost::evaluateParameter(const double *sample, const int *label, double &parity, double &threshold)
{
    for (int i = 0; i < sampleNum_; ++i) {
	data_[i] = sample[i];
	index_[i] = i;
    }
    
    sort_twin(data_, index_, sampleNum_);
    
    double epsilon = std::numeric_limits<double>::max();
    for (int i = 0; i < sampleNum_ - 1; ++i)
    {
	if (label[index_[i]] == label[index_[i + 1]]) {
	    continue;
	}
	if (data_[i] == data_[i + 1]) {
	    continue;
	}
	
	double p = 1.0;
	double theta = (data_[i] + data_[i + 1]) / 2.0;
	
	double error = 0.0;
	for (int j = 0; j < sampleNum_; ++j)
	{
	    if (label[j] != classify(sample[j], p, theta))
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
