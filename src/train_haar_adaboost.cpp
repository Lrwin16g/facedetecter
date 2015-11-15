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
    
    // Haar-like特徴量の読込み
    std::vector<Haar> candidateSet = loadHaarFeatures(argv[1]);
    std::cout << "haarNum: " << candidateSet.size() << std::endl;
    
    std::string datname = argv[2];
    std::string labelname = file::splitext(datname)[0] + ".label";
    
    // ラベルの読込み
    std::vector<int> labelSet = file::loadfile<int>(labelname.c_str(), false);
    
    // 特徴量抽出結果の読込み
    std::vector<std::vector<double> > sampleSet = file::loadfile<double>(datname.c_str(), labelSet.size(), true);
    std::cout << "classifierNum: " << sampleSet.size() << std::endl;
    std::cout << "sampleNum: " << labelSet.size() << std::endl;
    
    AdaBoost model;
    model.train(300, sampleSet, labelSet, candidateSet);
    
    return 0;
}
