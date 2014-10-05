#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

#include "haar.h"
#include "adaboost.h"
#include "filelib.h"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
	std::cerr << "Usage: " << argv[0] << " <haar-param> <haar-dat>" << std::endl;
	return -1;
    }
    
    std::vector<Haar> haar = loadHaarFeatures(argv[1]);
    
    std::cout << "haarNum: " << haar.size() << std::endl;
    
    std::string datname = argv[2];
    std::string labelname = datname;
    std::string ext = ".label";
    labelname.replace(labelname.rfind(".dat"), ext.length(), ext);
    
    std::vector<int> labelvec = file::loadfile<int>(labelname.c_str(), false);
    size_t sampleNum = labelvec.size();
    
    std::vector<std::vector<double> > data = file::loadfile<double>(datname.c_str(), sampleNum, true);
    size_t classifierNum = data.size();
    
    std::cout << "classifierNum: " << classifierNum << std::endl;
    std::cout << "sampleNum: " << sampleNum << std::endl;
    
    double **sample = new double*[classifierNum];
    for (int i = 0; i < classifierNum; ++i) {
	sample[i] = new double[sampleNum];
	for (int j = 0; j < sampleNum; ++j) {
	    sample[i][j] = data[i][j];
	}
    }
    
    int *label = new int[sampleNum];
    for (int i = 0; i < sampleNum; ++i) {
	label[i] = labelvec[i];
    }
    
    AdaBoost model(300, sampleNum);
    model.train(sample, label, haar);
    
    for (int i = 0; i < classifierNum; ++i) {
	delete[] sample[i];
    }
    delete[] sample;
    delete[] label;
    
    return 0;
}
