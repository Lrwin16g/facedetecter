#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

#include "haar.h"
#include "filelib.h"

int classify(double sample, int parity, double threshold)
{
    if (sample * static_cast<double>(parity) > static_cast<double>(parity) * threshold)
    {
	return 1;
    }
    else
    {
	return -1;
    }
}

double evaluateParameter(const double *sample, const int *label, const double *weight, int sampleNum, int &parity, double &threshold)
{
    double *data = new double[sampleNum];
    int *index = new int[sampleNum];
    for (int i = 0; i < sampleNum; ++i)
    {
	data[i] = sample[i];
	index[i] = i;
    }
    
    for (int i = 0; i < sampleNum - 1; ++i)
    {
	for (int j = sampleNum - 1; j > i; --j)
	{
	    if (data[j - 1] > data[j])
	    {
		double tmp = data[j];
		data[j] = data[j - 1];
		data[j - 1] = tmp;
		int tmp_1 = index[j];
		index[j] = index[j - 1];
		index[j - 1] = tmp_1;
	    }
	}
    }
    
    double epsilon = std::numeric_limits<double>::max();
    for (int i = 0; i < sampleNum - 1; ++i)
    {
	if (label[index[i]] == label[index[i + 1]])
	{
	    continue;
	}
	if (data[i] == data[i + 1])
	{
	    continue;
	}
	
	int p = 1;
	double theta = (data[i] + data[i + 1]) / 2.0;
	
	double error = 0.0;
	for (int j = 0; j < sampleNum; ++j)
	{
	    if (label[j] != classify(sample[j], p, theta))
	    {
		error += weight[j];
	    }
	}
	
	if (error > 0.5)
	{
	    error = 1.0 - error;
	    p = -1;
	}
	
	if (error < epsilon)
	{
	    epsilon = error;
	    parity = p;
	    threshold = theta;
	}
    }
    
    delete[] data;
    delete[] index;
    
    return epsilon;
}

int main(int argc, char *argv[])
{
    std::vector<Haar*> haarFeatures = loadHaarFeatures("/Users/ynakamura/workspace/github/FaceRecognition/src/mit_cbcl.haar");
    
    std::vector<std::vector<double> > data = file::loadfile<double>("/Users/ynakamura/workspace/github/FaceRecognition/src/mit_cbcl.dat", ' ');
    
    int classifierNum = data.size() - 1;
    int sampleNum = data[0].size();
    
    double **sample = new double*[classifierNum];
    for (int i = 0; i < classifierNum; ++i)
    {
	sample[i] = new double[sampleNum];
	for (int j = 0; j < sampleNum; ++j)
	{
	    sample[i][j] = data[i][j];
	}
    }
    
    const int category[2] = {1, -1};
    int sampleNumPerCategory[2] = {0, 0};
    
    int *label = new int[sampleNum];
    for (int i = 0; i < sampleNum; ++i)
    {
	label[i] = static_cast<int>(data[data.size() - 1][i]);
	if (label[i] == 1)
	{
	    sampleNumPerCategory[0]++;
	}
	else
	{
	    sampleNumPerCategory[1]++;
	}
    }
    
#ifdef _DEBUG
    for (int i = 0; i < sampleNum; ++i)
    {
	std::cout << label[i] << " ";
    }
    std::cout << std::endl;
#endif
    
    double *weight = new double[sampleNum];
    for (int i = 0; i < sampleNum; ++i)
    {
	if (label[i] == 1)
	{
	    weight[i] = 1.0 / static_cast<double>(2 * sampleNumPerCategory[0]);
	}
	else
	{
	    weight[i] = 1.0 / static_cast<double>(2 * sampleNumPerCategory[1]);
	}
    }
    
    std::vector<Haar*> strongClassifier;
    
    int weakClassifierNum = 110;
    int *weakClassifier = new int[weakClassifierNum];
    int *parity = new int[weakClassifierNum];
    double *threshold = new double[weakClassifierNum];
    double *alpha = new double[weakClassifierNum];
    
    for (int i = 0; i < weakClassifierNum; ++i)
    {
	double sum = 0.0;
	for (int j = 0; j < sampleNum; ++j)
	{
	    sum += weight[j];
	}
	for (int j = 0; j < sampleNum; ++j)
	{
	    weight[j] /= sum;
	}
	
	double epsilon = std::numeric_limits<double>::max();
	for (int j = 0; j < classifierNum; ++j)
	{
	    int p = 1;
	    double theta = 0.0;
	    double error = evaluateParameter(sample[j], label, weight, sampleNum, p, theta);
	    if (error < epsilon)
	    {
		epsilon = error;
		weakClassifier[i] = j;
		parity[i] = p;
		threshold[i] = theta;
	    }
	    if (j % 100 == 0)
	    {
		std::cout << j << "/" << classifierNum << ": error: " << error << " parity: " << p << " threshold: " << theta << std::endl;
	    }
	}
	
	alpha[i] = 1.0 / 2.0 * log((1.0 - epsilon) / epsilon);
	
	for (int j = 0; j < sampleNum; ++j)
	{
	    weight[j] *= exp(-alpha[i] * static_cast<double>(label[j] * classify(sample[weakClassifier[i]][j], parity[i], threshold[i])));
	}
	
	std::cout << i << ": classifier: " << weakClassifier[i] << " alpha: " << alpha[i] << " error: " << epsilon << std::endl;
	
	strongClassifier.push_back(haarFeatures[weakClassifier[i]]);
	std::string filename = "mit_cbcl_";
	std::stringstream ss;
	ss << strongClassifier.size();
	filename += ss.str();
	filename += ".param";
	std::ofstream fout(filename);
	for (int j = 0; j < strongClassifier.size(); ++j)
	{
	    fout << strongClassifier[j]->name() << " " << strongClassifier[j]->x() << " "
	     << strongClassifier[j]->y() << " " << strongClassifier[j]->width() << " "
	     << strongClassifier[j]->height() << " " << alpha[j] << " " << parity[j] << " " << threshold[j] << std::endl;
	}
	fout.close();
    }
    
    delete[] weakClassifier;
    delete[] parity;
    delete[] threshold;
    delete[] alpha;
    
    for (int i = 0; i < classifierNum; ++i)
    {
	delete[] sample[i];
    }
    delete[] sample;
    delete[] label;
    delete[] weight;
    
    /*for (size_t i = 0; i < haarFeatures.size(); ++i)
    {
	delete haarFeatures[i];
    }*/
    
    return 0;
}
