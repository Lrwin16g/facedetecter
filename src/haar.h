#ifndef HAAR_H
#define HAAR_H

#include <vector>

// Haar-like特徴量の基底クラス
class Haar
{
public:
    Haar(int x, int y, int width, int height);
    Haar(int x, int y, int width, int height, double weight, double parity, double threshold);
    virtual ~Haar() {}
    
    virtual double extract(double const * const * image) = 0;
    bool classify(double const * const * image);
    int classify(double sample);
    virtual const char* name() = 0;
    
    inline int x() const {return x_;}
    inline int y() const {return y_;}
    inline int width() const {return width_;}
    inline int height() const {return height_;}
    inline double value() const {return value_;}
    inline double weight() const {return weight_;}
    inline double parity() const {return parity_;}
    inline double threshold() const {return threshold_;}
    
protected:
    int x_, y_;
    int width_, height_;
    double value_;
    double weight_, parity_, threshold_;
};

// Edge特徴(横)
class HaarHEdge : public Haar
{
public:
    HaarHEdge(int x, int y, int width, int height);
    HaarHEdge(int x, int y, int width, int height, double weight, double parity, double threshold);
    double extract(double const * const * image);
    const char* name();
    static bool isValidRange(int x, int y, int width, int height, int rectWidth, int rectHeight);
};

// Edge特徴(縦)
class HaarVEdge : public Haar
{
public:
    HaarVEdge(int x, int y, int width, int height);
    HaarVEdge(int x, int y, int width, int height, double weight, double parity, double threshold);
    double extract(double const * const * image);
    const char* name();
    static bool isValidRange(int x, int y, int width, int height, int rectWidth, int rectHeight);
};

// Line特徴(横)
class HaarHLine : public Haar
{
public:
    HaarHLine(int x, int y, int width, int height);
    HaarHLine(int x, int y, int width, int height, double weight, double parity, double threshold);
    double extract(double const * const * image);
    const char* name();
    static bool isValidRange(int x, int y, int width, int height, int rectWidth, int rectHeight);
};

// Line特徴(縦)
class HaarVLine : public Haar
{
public:
    HaarVLine(int x, int y, int width, int height);
    HaarVLine(int x, int y, int width, int height, double weight, double parity, double threshold);
    double extract(double const * const * image);
    const char* name();
    static bool isValidRange(int x, int y, int width, int height, int rectWidth, int rectHeight);
};

// ROIのサイズからHaar-like特徴量の配列を作成
std::vector<Haar*> createHaarFeatures(int width, int height, int scanStep, int sizeStep);

// Haar-like特徴量を読込み
std::vector<Haar*> loadHaarFeatures(const char *filename);

// Haar-like特徴量を保存
void saveHaarFeatures(const char *filename, std::vector<Haar*> &haarFeatures);

// 積分画像
void createIntegralImage(double const * const * src, double **dst, int width, int height);
double calcLuminance(double const * const * image, int x, int y, int width, int height);

#endif
