#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "haar.h"
#include "filelib.h"

// 顔:1、非顔:-1
const int categoryNum = 2;
const int category[2] = {1, -1};

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
	std::cerr << "Usage: " << argv[0] << " <[in]image-width> <[in]image-height> <[in]scan-step> <[in]size-step> <[out]haar-param>" << std::endl;
	return -1;
    }
    
    int width = atoi(argv[1]);	// 画像の横幅
    int height = atoi(argv[2]);	// 画像の縦幅
    
    int scanStep = atoi(argv[3]);	// 特徴量を走査する間隔の設定
    int sizeStep = atoi(argv[4]);	// 特徴量のサイズを変える間隔の設定
    
    // Haar-like特徴量の配列の作成
    std::vector<Haar> haar = createHaarFeatures(width, height, scanStep, sizeStep);
    
    // 特徴量のリストの保存
    std::cout << "haarFeature: " << haar.size() << std::endl;
    std::ostringstream oss;
    oss << haar.size();
    std::string name = std::string(argv[5]) + "_" + oss.str() + ".haar";
    saveHaarFeatures(name.c_str(), haar);
    
    return 0;
}
