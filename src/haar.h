#ifndef HAAR_H
#define HAAR_H

#include <vector>

// Haar-like特徴量クラス
class Haar
{
public:
    Haar(int type, int x, int y, int width, int height, double parity, double threshold);
    Haar(const Haar &obj);
    ~Haar();
    
    double extract(double const * const * image);
    bool classify(double const * const * image);
    static bool isValidRange(int type, int x, int y, int width, int height, int rectWidth, int rectHeight);
    
    Haar& operator=(const Haar &obj);
    
    inline int    type()      const { return type_; }
    inline int    x()         const { return x_; }
    inline int    y()         const { return y_; }
    inline int    width()     const { return width_; }
    inline int    height()    const { return height_; }
    inline double parity()    const { return parity_; }
    inline double threshold() const { return threshold_; }
    
    inline void setParity(double parity)	{ parity_ = parity; }
    inline void setThreshold(double threshold)	{ threshold_ = threshold; }
    
private:
    int type_;
    int x_, y_;
    int width_, height_;
    double parity_, threshold_;
};

// ROIのサイズからHaar-like特徴量の配列を作成
std::vector<Haar> createHaarFeatures(int width, int height, int scanStep, int sizeStep);

// Haar-like特徴量を読込み
std::vector<Haar> loadHaarFeatures(const char *filename);

// Haar-like特徴量を保存
void saveHaarFeatures(const char *filename, const std::vector<Haar> &haar);

// 積分画像
void createIntegralImage(double const * const * src, double **dst, int width, int height);
double calcLuminance(double const * const * image, int x, int y, int width, int height);

#endif
