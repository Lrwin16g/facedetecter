#ifndef ADABOOST_H
#define ADABOOST_H

#include <vector>

class Haar;

class AdaBoost
{
public:
    AdaBoost();
    ~AdaBoost();
    void train(double **sample, int *label, std::vector<Haar*> &candidate, int sampleNum, int classifierNum);
    
private:
    int classifierNum_;
    int sampleNum_;
    const int categoryNum_ = 2;
    const int category_[2] = {1, -1};
    
    std::vector<Haar*> classifier_;
    double *weight_;
    double *alpha_;
};

#endif
