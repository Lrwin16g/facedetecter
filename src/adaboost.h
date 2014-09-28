#ifndef ADABOOST_H
#define ADABOOST_H

#include <vector>
#include "haar.h"

class AdaBoost
{
public:
    AdaBoost(int classifierNum, int sampleNum);
    ~AdaBoost();
    
    void train(const double * const *sample, const int *label, const std::vector<Haar> &candidate);
    
private:
    void initializeWeight(const int *label);
    void normalizeWeight();
    int classify(double sample, double parity, double threshold);
    
    struct Sample
    {
	double value;
	int category;
	int index;
	
	Sample() {}
	Sample(double a, int b, int c) {
	    value = a;
	    category = b;
	    index = c;
	}
	
	bool operator<(const Sample &sample) const {
	    return value < sample.value;
	}
    };
    
    double evaluateParameter(const double *sample, const int *label, const std::vector<Sample> &sortedSample, double &parity, double &threshold);
    
    int classifierNum_;
    int sampleNum_;
    int category_[2];
    
    std::vector<Haar> classifier_;
    double *alpha_;
    double *weight_;
};

#endif
