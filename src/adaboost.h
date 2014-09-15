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
    const int categoryNum = 2;
    const int category[2] = {1, -1};
    
    std::vector<Haar*> classifier;
    double *weight;
    double *alpha;
};

#endif
