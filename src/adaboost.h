#ifndef ADABOOST_H
#define ADABOOST_H

#include <vector>
#include <map>
#include "haar.h"

class AdaBoost
{
public:
    AdaBoost();
    ~AdaBoost();
    
    void initialize(int classifierNum, int sampleNum,
		    const std::vector<std::vector<double> > &sampleSet,
		    const std::vector<int> &labelSet);
    void train(int classifierNum,
	       const std::vector<std::vector<double> > &sampleSet,
	       const std::vector<int> &labelSet,
	       const std::vector<Haar> &candidateSet);
    void trainOnce(const std::vector<std::vector<double> > &sampleSet,
		   const std::vector<int> &labelSet,
		   const std::vector<Haar> &candidateSet);
    /*std::pair<double, double> adjustThreshold(const std::vector<std::vector<double> > &sampleSet,
					      const std::vector<int> &labelSet,
					      double targetDetectionRate);*/
    std::pair<double, double> adjustThreshold(const std::vector<std::vector<double> > &validateSampleSet,
					      const std::vector<std::vector<double> > &trainSampleSet,
					      const std::vector<int> &labelSet,
					      double targetDetectionRate);
    int classify(int index, const std::vector<std::vector<double> > &sampleSet);
    
private:
    void initializeWeight(const std::vector<int> &label);
    void normalizeWeight();
    int classify(double value, double parity, double threshold);
    
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
    
    double evaluateParameter(const std::vector<double> &sample,
			     const std::vector<int> &labelSet,
			     const std::vector<Sample> &sortedSampleSet,
			     double &parity, double &threshold);
    
    int classifierNum_;
    int sampleNum_;
    int category_[2];
    double threshold_;
    
    std::vector<Haar> classifier_;
    std::vector<double> alpha_;
    std::vector<double> weight_;
    std::vector<std::vector<Sample> > sortedSampleSet_;
};

#endif
