#ifndef CASCADECLASSIFIER_H
#define CASCADECLASSIFIER_H

#include <vector>
#include "adaboost.h"

class CascadeClassifier
{
public:
    CascadeClassifier();
    ~CascadeClassifier();
    
    void train(const std::vector<Haar> &candidateSet,
	       const std::vector<std::vector<double> > &positiveSampleSet,
	       const std::vector<std::vector<double> > &negativeSampleSet,
	       const std::vector<std::vector<double> > &validatePositiveSampleSet,
	       double minDetectionRate,
	       double maxFalsePositiveRate,
	       double maxTotalFalsePositiveRate,
	       int maxCascadeNum);
    
    int classify(double const * const * image);
    
    void loadfile(const char *filename);
    void savefile(const char *filename);
    
private:
    std::vector<AdaBoost> cascade_;
};

#endif
