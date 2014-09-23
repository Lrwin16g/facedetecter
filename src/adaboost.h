#ifndef ADABOOST_H
#define ADABOOST_H

#include <vector>
#include "haar.h"

class AdaBoost
{
public:
    AdaBoost(int classifierNum, int sampleNum);
    ~AdaBoost();
    
    void initializeWeight(const int *label);
    void train(const double * const *sample, const int *label, const std::vector<Haar> &candidate);
    
private:
    void normalizeWeight();
    int classify(double sample, double parity, double threshold);
    double evaluateParameter(const double *sample, const int *label, double &parity, double &threshold);
    
    int classifierNum_;
    int sampleNum_;
    int category_[2];
    
    std::vector<Haar> classifier_;
    double *alpha_;
    double *weight_;
    double *data_;
    int *index_;
};

#endif
