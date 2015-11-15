#ifndef CASCADECLASSIFIER_H
#define CASCADECLASSIFIER_H

#include <vector>
#include "adaboost.h"

class CascadeClassifier
{
public:
    CascadeClassifier();
    ~CascadeClassifier();
    
    void train(const std::vector<std::vector<double> > &positiveSampleSet,
	       const std::vector<std::vector<double> > &negativeSampleSet,
	       const std::vector<Haar> &candidateSet,
	       const std::vector<std::vector<double> > &validatePositiveSampleSet,
	       //const std::vector<std::vector<double> > &validateNegativeSampleSet,
	       double minDetectionRate,
	       double maxFalsePositiveRate,
	       double maxTotalFalsePositiveRate,
	       int maxCascadeNum);
    
private:
    std::vector<AdaBoost> cascade_;
};

#endif
