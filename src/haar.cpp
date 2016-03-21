#include "haar.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "filelib.h"

const int TypeNum = 6;

Haar::Haar()
    : type_(0), x_(0), y_(0), width_(0), height_(0),
      parity_(0.0), threshold_(0.0), index_(0)
{
}

Haar::Haar(int type, int x, int y, int width, int height, double parity, double threshold)
    : type_(type), x_(x), y_(y), width_(width), height_(height),
      parity_(parity), threshold_(threshold), index_(0)
{
}

Haar::Haar(const Haar &obj)
{
    type_ = obj.type();
    x_ = obj.x();
    y_ = obj.y();
    width_ = obj.width();
    height_ = obj.height();
    parity_ = obj.parity();
    threshold_ = obj.threshold();
    index_ = obj.index();
}

Haar::~Haar()
{
}

double Haar::extract(double const * const * image)
{
    double value;
    double white_1, white_2, white_3, white_4, white_5, white_6, white_7, white_8;
    double black_1, black_2;
    
    switch (type_)
    {
    case 0:	// HaarHEdge
	white_1 = calcLuminance(image, x_, y_, width_, height_);
	black_1 = calcLuminance(image, x_ + width_ + 1, y_, width_, height_);
	value = (white_1 - black_1) / static_cast<double>((width_ + 1) * (height_ + 1));
	break;
    
    case 1:	// HaarVEdge
	white_1 = calcLuminance(image, x_, y_, width_, height_);
	black_1 = calcLuminance(image, x_, y_ + height_ + 1, width_, height_);
	value = (white_1 - black_1) / static_cast<double>((width_ + 1) * (height_ + 1));
	break;
    
    case 2:	// HaarHLine
	white_1 = calcLuminance(image, x_, y_, width_ * 3 + 2, height_);
	black_1 = calcLuminance(image, x_ + width_ + 1, y_, width_, height_);
	value = (white_1 - 3.0 * black_1) / static_cast<double>(2 * (width_ + 1) * (height_ + 1));
	break;
    
    case 3:	// HaarVLine
	white_1 = calcLuminance(image, x_, y_, width_, height_ * 3 + 2);
	black_1 = calcLuminance(image, x_, y_ + height_ + 1, width_, height_);
	value = (white_1 - 3.0 * black_1) / static_cast<double>(2 * (width_ + 1) * (height_ + 1));
	break;
    
    case 4:	// HaarChecker
	white_1 = calcLuminance(image, x_, y_, width_ * 2 + 1, height_ * 2 + 1);
	black_1 = calcLuminance(image, x_ + width_ + 1, y_, width_, height_);
	black_2 = calcLuminance(image, x_, y_ + height_ + 1, width_, height_);
	value = (white_1 - 2.0 * (black_1 + black_2)) / static_cast<double>(2 * (width_ + 1) * (height_ + 1));
	break;
    
    case 5:	// HaarCenterSurround
	white_1 = calcLuminance(image, x_, y_, width_ * 3 + 2, height_ * 3 + 2);
	black_1 = calcLuminance(image, x_ + width_ + 1, y_ + height_ + 1, width_, height_);
	value = (white_1 - 9.0 * black_1) / static_cast<double>(8 * (width_ + 1) * (height_ + 1));
	break;
    }
    
    return value;
}

bool Haar::classify(double const * const * image)
{
    double value = extract(image);
    if (value * parity_ > parity_ * threshold_) {
	return true;
    } else {
	return false;
    }
}

bool Haar::isValidRange(int type, int x, int y, int width, int height, int rectWidth, int rectHeight)
{
    switch (type)
    {
    case 0:	// HaarHEdge
	if ((x + width * 2 + 1 < rectWidth) && (y + height < rectHeight)) {
	    return true;
	} else {
	    return false;
	}
	break;
    
    case 1:	// HaarVEdge
	if ((x + width < rectWidth) && (y + height * 2 + 1 < rectHeight)) {
	    return true;
	} else {
	    return false;
	}
	break;
    
    case 2:	// HaarHLine
	if ((x + width * 3 + 2 < rectWidth) && (y + height < rectHeight)) {
	    return true;
	} else {
	    return false;
	}
	break;
    
    case 3:	// HaarVLine
	if ((x + width < rectWidth) && (y + height * 3 + 2 < rectHeight)) {
	    return true;
	} else {
	    return false;
	}
	break;
    
    case 4:	// HaarChecker
	if ((x + width * 2 + 1 < rectWidth) && (y + height * 2 + 1 < rectHeight)) {
	    return true;
	} else {
	    return false;
	}
	break;
    
    case 5:	// HaarCenterSurround
	if ((x + width * 3 + 2 < rectWidth) && (y + height * 3 + 2 < rectHeight)) {
	    return true;
	} else {
	    return false;
	}
	break;
    }
    
    return false;
}

void Haar::load(std::ifstream &fin)
{
    fin >> threshold_ >> parity_ >> type_;
    fin >> x_ >> y_ >> width_ >> height_;
    fin.ignore(256, '\n');
}

void Haar::save(std::ofstream &fout) const
{
    fout << threshold_ << " ";
    fout << parity_ << " ";
    fout << type_ << " ";
    fout << x_ << " ";
    fout << y_ << " ";
    fout << width_ << " ";
    fout << height_ << " ";
    fout << std::endl;
}

Haar& Haar::operator=(const Haar &obj)
{
    type_ = obj.type();
    x_ = obj.x();
    y_ = obj.y();
    width_ = obj.width();
    height_ = obj.height();
    parity_ = obj.parity();
    threshold_ = obj.threshold();
    
    return *this;
}

std::vector<Haar> createHaarFeatures(int width, int height, int scanStep, int sizeStep)
{
    std::vector<Haar> haar;
    for (int y = 0; y < height; y += scanStep)
    {
	for (int x = 0; x < width; x += scanStep)
	{
	    for (int h = 0; h < height; h += sizeStep)
	    {
		for (int w = 0; w < width; w += sizeStep)
		{
		    for (int t = 0; t < TypeNum; ++t)
		    {
			if (Haar::isValidRange(t, x, y, w, h, width, height))
			{
			    haar.push_back(Haar(t, x, y, w, h, 0.0, 0.0));
			}
		    }
		}
	    }
	}
    }
    
    return haar;
}

std::vector<Haar> loadHaarFeatures(const char *filename)
{
    std::ifstream fin(filename);
    int count = 0;
    fin >> count;
    fin.ignore(256, '\n');
    
    std::vector<Haar> haar(count);
    for (int i = 0; i < count; ++i)
    {
	haar[i].load(fin);
    }
    fin.close();
    
    return haar;
}

void saveHaarFeatures(const char *filename, const std::vector<Haar> &haar)
{
    std::ofstream fout(filename);
    fout << haar.size() << std::endl;
    
    for (size_t i = 0; i < haar.size(); ++i)
    {
	haar[i].save(fout);
    }
    fout.close();
}

void createIntegralImage(double const * const * src, double **dst, int width, int height)
{
    for (int y = 0; y < height + 1; ++y) {
	dst[y][0] = 0.0;
    }
    for (int x = 0; x < width + 1; ++x) {
	dst[0][x] = 0.0;
    }
    
    for (int x = 1; x < width + 1; ++x)
    {
	dst[1][x] = src[0][x - 1];
    }
    
    for (int y = 2; y < height + 1; ++y)
    {
	for (int x = 1; x < width + 1; ++x)
	{
	    dst[y][x] = src[y - 1][x - 1] + dst[y - 1][x];
	}
    }
    
    for (int y = 1; y < height + 1; ++y)
    {
	for (int x = 2; x < width + 1; ++x)
	{
	    dst[y][x] += dst[y][x - 1];
	}
    }
}

double calcLuminance(double const * const * image, int x, int y, int width, int height)
{
    double term_1 = image[y + height + 1][x + width + 1];
    double term_2 = image[y][x];
    double term_3 = image[y + height + 1][x];
    double term_4 = image[y][x + width + 1];
    
    return (term_1 + term_2) - (term_3 + term_4);
}
