#include "haar.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

const int TypeNum = 5;

std::vector<std::string> split(const std::string &str, const char delim)
{
    std::string::size_type i = 0;
    std::string::size_type j = str.find(delim);
    
    std::vector<std::string> result;
    
    while (j != std::string::npos) {
	result.push_back(str.substr(i, j - i));
        i = ++j;
        j = str.find(delim, j);
        
        if (j == std::string::npos) {
	    result.push_back(str.substr(i, str.length()));
	}
    }
    
    return result;
}

std::vector<std::vector<std::string> > loadfile(const char *filename, const char delim)
{
    std::ifstream ifs(filename);
    std::vector<std::vector<std::string> > data;
    std::string line;
    while (std::getline(ifs, line))
    {
	std::vector<std::string> tokens = split(line.substr(0, line.length()), delim);
	data.push_back(tokens);
    }
    ifs.close();
    
    return data;
}

Haar::Haar(int type, int x, int y, int width, int height, double parity, double threshold)
    : type_(type), x_(x), y_(y), width_(width), height_(height),
      parity_(parity), threshold_(threshold)
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
}

Haar::~Haar()
{
}

double Haar::extract(double const * const * image)
{
    double value;
    double white_1, white_2;
    double black_1, black_2;
    
    switch (type_)
    {
    case 0:	// HaarHEdge
	white_1 = calcLuminance(image, x_, y_, width_, height_);
	black_1 = calcLuminance(image, x_ + width_, y_, width_, height_);
	value = (white_1 - black_1) / static_cast<double>(width_ * height_);
	break;
    
    case 1:	// HaarVEdge
	white_1 = calcLuminance(image, x_, y_, width_, height_);
	black_1 = calcLuminance(image, x_, y_ + height_, width_, height_);
	value = (white_1 - black_1) / static_cast<double>(width_ * height_);
	break;
    
    case 2:	// HaarHLine
	white_1 = calcLuminance(image, x_, y_, width_ * 3, height_);
	black_1 = calcLuminance(image, x_ + width_, y_, width_, height_);
	value = (white_1 - 2.0 * black_1) / static_cast<double>(width_ * height_);
	break;
    
    case 3:	// HaarVLine
	white_1 = calcLuminance(image, x_, y_, width_, height_ * 3);
	black_1 = calcLuminance(image, x_, y_ + height_, width_, height_);
	value = (white_1 - 2.0 * black_1) / static_cast<double>(width_ * height_);
	break;
    
    case 4:	// HaarChecker
	white_1 = calcLuminance(image, x_, y_, width_ * 2, height_ * 2);
	black_1 = calcLuminance(image, x_ + width_, y_, width_, height_);
	black_2 = calcLuminance(image, x_, y_ + height_, width_, height_);
	value = (white_1 - 2.0 * (black_1 + black_2)) / static_cast<double>(width_ * height_);
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
	if ((x + width * 2 < rectWidth) && (y + height < rectHeight)) {
	    return true;
	} else {
	    return false;
	}
	break;
    
    case 1:	// HaarVEdge
	if ((x + width < rectWidth) && (y + height * 2 < rectHeight)) {
	    return true;
	} else {
	    return false;
	}
	break;
    
    case 2:	// HaarHLine
	if ((x + width * 3 < rectWidth) && (y + height < rectHeight)) {
	    return true;
	} else {
	    return false;
	}
	break;
    
    case 3:	// HaarVLine
	if ((x + width < rectWidth) && (y + height * 3 < rectHeight)) {
	    return true;
	} else {
	    return false;
	}
	break;
    
    case 4:	// HaarChecker
	if ((x + width * 2 < rectWidth) && (y + height * 2 < rectHeight)) {
	    return true;
	} else {
	    return false;
	}
	break;
    }
    
    return false;
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
    for (int y = 1; y < height; y += scanStep)
    {
	for (int x = 1; x < width; x += scanStep)
	{
	    for (int h = 1; h < height; h += sizeStep)
	    {
		for (int w = 1; w < width; w += sizeStep)
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
    std::vector<std::vector<std::string> > str = loadfile(filename, ' ');
    std::vector<Haar> haar;
    
    for (size_t i = 0; i < str.size(); ++i)
    {
	int    type      = atoi(str[i][0].c_str());
	int    x         = atoi(str[i][1].c_str());
	int    y         = atoi(str[i][2].c_str());
	int    width     = atoi(str[i][3].c_str());
	int    height    = atoi(str[i][4].c_str());
	double parity    = atof(str[i][5].c_str());
	double threshold = atof(str[i][6].c_str());
	
	haar.push_back(Haar(type, x, y, width, height, parity, threshold));
    }
    
    return haar;
}

void saveHaarFeatures(const char *filename, const std::vector<Haar> &haar)
{
    std::ofstream fout(filename);
    for (size_t i = 0; i < haar.size(); ++i)
    {
	fout << haar[i].type() << " "
	     << haar[i].x() << " "
	     << haar[i].y() << " "
	     << haar[i].width() << " "
	     << haar[i].height() << " "
	     << haar[i].parity() << " "
	     << haar[i].threshold() << " "
	     << std::endl;
    }
    fout.close();
}

void createIntegralImage(double const * const * src, double **dst, int width, int height)
{
    for (int x = 0; x < width; ++x)
    {
	dst[0][x] = src[0][x];
    }
    
    for (int y = 1; y < height; ++y)
    {
	for (int x = 0; x < width; ++x)
	{
	    dst[y][x] = src[y][x] + dst[y - 1][x];
	}
    }
    
    for (int y = 0; y < height; ++y)
    {
	for (int x = 1; x < width; ++x)
	{
	    dst[y][x] += dst[y][x - 1];
	}
    }
}

double calcLuminance(double const * const * image, int x, int y, int width, int height)
{
    double term_1 = image[y - 1 + height][x - 1 + width];
    double term_2 = image[y - 1][x - 1];
    double term_3 = image[y - 1 + height][x - 1];
    double term_4 = image[y - 1][x - 1 + width];
    
    return (term_1 + term_2) - (term_3 + term_4);
}
