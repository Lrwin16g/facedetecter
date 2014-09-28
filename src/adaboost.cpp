#include "adaboost.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <limits>
#include <cmath>

#include "haar.h"
#include "filelib.h"

const int categoryNum_ = 2;

AdaBoost::AdaBoost(int classifierNum, int sampleNum)
    : classifierNum_(classifierNum), sampleNum_(sampleNum),
      weight_(NULL), alpha_(NULL)
{
    category_[0] = 1;	category_[1] = -1;
    alpha_ = new double[classifierNum_];
    weight_ = new double[sampleNum_];
}

AdaBoost::~AdaBoost()
{
    if (alpha_ != NULL) {
	delete[] alpha_;
    }
    if (weight_ != NULL) {
	delete[] weight_;
    }
}

void AdaBoost::train(const double * const *sample, const int *label, const std::vector<Haar> &candidate)
{
    initializeWeight(label);
    
    std::vector<std::vector<Sample> > sortedSample(candidate.size());
    for (size_t i = 0; i < candidate.size(); ++i)
    {
	sortedSample[i].resize(sampleNum_);
	for (int j = 0; j < sampleNum_; ++j)
	{
	    sortedSample[i][j] = Sample(sample[i][j], label[j], j);
	}
	std::sort(sortedSample[i].begin(), sortedSample[i].end());
    }
    
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
	    double error = evaluateParameter(sample[j], label, sortedSample[j], p, theta);
	    if (error < epsilon)
	    {
		epsilon = error;
		idx = j;
		parity = p;
		threshold = theta;
	    }
	    
	    if (j % 100 == 0) {
		std::cout << "training: " << j << "/" << candidate.size() << std::endl;
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
	
	std::stringstream ss;
	ss << std::setw(3) << std::setfill('0') << classifier_.size();
	std::string filename_1 = "mit_cbcl_" + ss.str() + ".param";
	std::string filename_2 = "mit_cbcl_" + ss.str() + ".alpha";
	saveHaarFeatures(filename_1.c_str(), classifier_);
	file::savefile(filename_2.c_str(), alpha_, classifier_.size(), false);
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

int AdaBoost::classify(double sample, double parity, double threshold)
{
    if (sample * parity > parity * threshold) {
	return category_[0];
    } else {
	return category_[1];
    }
}

double AdaBoost::evaluateParameter(const double *sample, const int *label,
				   const std::vector<Sample> &sortedSample,
				   double &parity, double &threshold)
{
    double epsilon = std::numeric_limits<double>::max();
    for (int i = 0; i < sampleNum_ - 1; ++i)
    {
	if (sortedSample[i].category == sortedSample[i + 1].category) {
	    continue;
	}
	if (sortedSample[i].value == sortedSample[i + 1].value) {
	    continue;
	}
	
	double p = 1.0;
	double theta = (sortedSample[i].value + sortedSample[i + 1].value) / 2.0;
	
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
