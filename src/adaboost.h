#ifndef ADABOOST_H
#define ADABOOST_H

#include <vector>
#include <map>
#include <fstream>
#include "haar.h"

class AdaBoost
{
public:
    AdaBoost();
    ~AdaBoost();
    
    void initialize(int classifierNum, int sampleNum,
		    const std::vector<std::vector<double> > &sampleSet,
		    const std::vector<int> &labelSet);
    
    void train(const std::vector<std::vector<double> > &sampleSet,
	       const std::vector<int> &labelSet,
	       const std::vector<Haar> &candidateSet,
	       int maxClassifierNum);
    void trainOnce(const std::vector<std::vector<double> > &sampleSet,
		   const std::vector<int> &labelSet,
		   const std::vector<Haar> &candidateSet);
    std::pair<double, double> adjustThreshold(const std::vector<std::vector<double> > &validateSampleSet,
					      const std::vector<std::vector<double> > &trainSampleSet,
					      const std::vector<int> &labelSet,
					      double targetDetectionRate);
    
    int classify(int index, const std::vector<std::vector<double> > &sampleSet);
    int classify(double const * const * image);
    
    void loadfile(const char *filename);
    void savefile(const char *filename);
    
    void load(std::ifstream &fin);
    void save(std::ofstream &fout);
    
private:
    void initializeWeight(const std::vector<int> &label);
    void initializeWeight();
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
    double threshold_;
    
    std::vector<Haar> classifier_;
    std::vector<double> alpha_;
    std::vector<double> weight_;
    std::vector<std::vector<Sample> > sortedSampleSet_;
};

#endif
