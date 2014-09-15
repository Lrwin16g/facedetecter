#include "adaboost.h"
#include <iostream>
#include <limits>
#include <cmath>

#include "haar.h"

void AdaBoost::train(double **sample, int *label, std::vector<Haar*> &candidate, int sampleNum, int classifierNum)
{
    cleanUp();
    
    initialize(classifierNum, candidate.size(), sampleNum);
    
    int sampleNumPerCategory[categoryNum_] = {0, 0};
    for (int i = 0; i < sampleNum; ++i)
    {
	if (label[i] == category_[0]) {
	    sampleNumPerCategory[0]++;
	} else {
	    sampleNumPerCategory[1]++;
	}
    }
    
    for (int i = 0; i < sampleNum; ++i)
    {
	if (label[i] == 1) {
	    weight_[i] = 1.0 / static_cast<double>(2 * sampleNumPerCategory[0]);
	} else {
	    weight_[i] = 1.0 / static_cast<double>(2 * sampleNumPerCategory[1]);
	}
    }
    
    for (int i = 0; i < classifierNum_; ++i)
    {
	double sum = 0.0;
	for (int i = 0; i < sampleNum_; ++i) {
	    sum += weight_[i];
	}
	for (int i = 0; i < sampleNum_; ++i) {
	    weight_[i] /= sum;
	}
	
	double epsilon = std::numeric_limits<double>::max();
	int index = -1;
	for (size_t j = 0; j < candidate.size(); ++j)
	{
	    double error = evalClassifier(sample[j], label, candidate);
	    if (error < epsilon)
	    {
		epsilon = error;
		index = j;
	    }
	    
	    if (j % 100 == 0) {
		std::cout << j << "/" << candidate.size() << std::endl;
	    }
	}
	
	classifier.push_back(candidate[index]);
	
	alpha_[i] = 1.0 /2.0 * log((1.0 - epsilon) / epsilon);
	
	for (int j = 0; j < sampleNum_; ++j)
	{
	    weight_[j] *= exp(-alpha_[i] * static_cast<double>(label[j] * candidate[index]->classify(sample[index][j])));
	}
	
	std::cout << i << ": classifier: " << index << " alpha: " << alpha_[i] << " error: " << epsilon << std::endl;
    }
}
