#include "haar.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>

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

Haar::Haar(int x, int y, int width, int height)
    : x_(x), y_(y), width_(width), height_(height),
      value_(0.0), weight_(0.0), parity_(0.0), threshold_(0.0)
{
}

Haar::Haar(int x, int y, int width, int height, double weight, double parity, double threshold)
    : x_(x), y_(y), width_(width), height_(height),
      value_(0.0), weight_(weight), parity_(parity), threshold_(threshold)
{
}

bool Haar::classify(double const * const * image)
{
    extract(image);
    if (value_ * parity_ > parity_ * threshold_)
    {
	return true;
    }
    else
    {
	return false;
    }
}

HaarHEdge::HaarHEdge(int x, int y, int width, int height)
    : Haar(x, y, width, height)
{
}

HaarHEdge::HaarHEdge(int x, int y, int width, int height, double weight, double parity, double threshold)
    : Haar(x, y, width, height, weight, parity, threshold)
{
}

double HaarHEdge::extract(double const * const * image)
{
    double white = calcLuminance(image, x_, y_, width_, height_) / static_cast<double>(width_ * height_);
    double black = calcLuminance(image, x_ + width_, y_, width_, height_) / static_cast<double>(width_ * height_);
    
    value_ = white - black;
    
    return value_;
}

const char* HaarHEdge::name()
{
    return "HaarHEdge";
}

bool HaarHEdge::isValidRange(int x, int y, int width, int height, int rectWidth, int rectHeight)
{
    if ((x + width * 2 < rectWidth) && (y + height < rectHeight))
    {
	return true;
    } else
    {
	return false;
    }
}

HaarVEdge::HaarVEdge(int x, int y, int width, int height)
     : Haar(x, y, width, height)
{
}

HaarVEdge::HaarVEdge(int x, int y, int width, int height, double weight, double parity, double threshold)
    : Haar(x, y, width, height, weight, parity, threshold)
{
}

double HaarVEdge::extract(double const * const * image)
{
    double white = calcLuminance(image, x_, y_, width_, height_) / static_cast<double>(width_ * height_);
    double black = calcLuminance(image, x_, y_ + height_, width_, height_) / static_cast<double>(width_ * height_);
    
    value_ = white - black;
    
    return value_;
}

const char* HaarVEdge::name()
{
    return "HaarVEdge";
}

bool HaarVEdge::isValidRange(int x, int y, int width, int height, int rectWidth, int rectHeight)
{
    if ((x + width < rectWidth) && (y + height * 2 < rectHeight))
    {
	return true;
    } else
    {
	return false;
    }
}

HaarHLine::HaarHLine(int x, int y, int width, int height)
     : Haar(x, y, width, height)
{
}

HaarHLine::HaarHLine(int x, int y, int width, int height, double weight, double parity, double threshold)
    : Haar(x, y, width, height, weight, parity, threshold)
{
}

double HaarHLine::extract(double const * const * image)
{
    double white_1 = calcLuminance(image, x_, y_, width_, height_) / static_cast<double>(width_ * height_);
    double black = calcLuminance(image, x_ + width_, y_, width_, height_) / static_cast<double>(width_ * height_);
    double white_2 = calcLuminance(image, x_ + width_ * 2, y_, width_, height_) / static_cast<double>(width_ * height_);
    
    value_ = white_1 + white_2 - black;
    
    return value_;
}

const char* HaarHLine::name()
{
    return "HaarHLine";
}

bool HaarHLine::isValidRange(int x, int y, int width, int height, int rectWidth, int rectHeight)
{
    if ((x + width * 3 < rectWidth) && (y + height < rectHeight))
    {
	return true;
    } else
    {
	return false;
    }
}

HaarVLine::HaarVLine(int x, int y, int width, int height)
     : Haar(x, y, width, height)
{
}

HaarVLine::HaarVLine(int x, int y, int width, int height, double weight, double parity, double threshold)
    : Haar(x, y, width, height, weight, parity, threshold)
{
}

double HaarVLine::extract(double const * const * image)
{
    double white_1 = calcLuminance(image, x_, y_, width_, height_) / static_cast<double>(width_ * height_);
    double black = calcLuminance(image, x_, y_ + height_, width_, height_) / static_cast<double>(width_ * height_);
    double white_2 = calcLuminance(image, x_, y_ + height_ * 2, width_, height_) / static_cast<double>(width_ * height_);
    
    value_ = white_1 + white_2 - black;
    
    return value_;
}

const char* HaarVLine::name()
{
    return "HaarVLine";
}

bool HaarVLine::isValidRange(int x, int y, int width, int height, int rectWidth, int rectHeight)
{
    if ((x + width < rectWidth) && (y + height * 3 < rectHeight))
    {
	return true;
    } else
    {
	return false;
    }
}

std::vector<Haar*> createHaarFeatures(int width, int height, int scanStep, int sizeStep)
{
    std::vector<Haar*> haarFeatures;
    for (int y = 1; y < height; y += scanStep)
    {
	for (int x = 1; x < width; x += scanStep)
	{
	    for (int h = 1; h < height; h += sizeStep)
	    {
		for (int w = 1; w < width; w += sizeStep)
		{
		    if (HaarHEdge::isValidRange(x, y, w, h, width, height))
		    {
			haarFeatures.push_back(new HaarHEdge(x, y, w, h));
		    }
		    if (HaarVEdge::isValidRange(x, y, w, h, width, height))
		    {
			haarFeatures.push_back(new HaarVEdge(x, y, w, h));
		    }
		    if (HaarHLine::isValidRange(x, y, w, h, width, height))
		    {
			haarFeatures.push_back(new HaarHLine(x, y, w, h));
		    }
		    if (HaarVLine::isValidRange(x, y, w, h, width, height))
		    {
			haarFeatures.push_back(new HaarVLine(x, y, w, h));
		    }
		}
	    }
	}
    }
    return haarFeatures;
}

std::vector<Haar*> loadHaarFeatures(const char *filename)
{
    std::vector<std::vector<std::string> > str = loadfile(filename, ' ');
    std::vector<Haar*> haarFeatures;
    for (size_t i = 0; i < str.size(); ++i)
    {
	std::string name = str[i][0];
	if (name == "HaarHEdge")
	{
	    if (str[i].size() > 5) {
		haarFeatures.push_back(new HaarHEdge(atoi(str[i][1].c_str()), atoi(str[i][2].c_str()), atoi(str[i][3].c_str()), atoi(str[i][4].c_str()), atof(str[i][5].c_str()), atof(str[i][6].c_str()), atof(str[i][7].c_str())));
	    } else {
		haarFeatures.push_back(new HaarHEdge(atoi(str[i][1].c_str()), atoi(str[i][2].c_str()), atoi(str[i][3].c_str()), atoi(str[i][4].c_str())));
	    }
	}
	else if (name == "HaarVEdge")
	{
	    if (str[i].size() > 5) {
		haarFeatures.push_back(new HaarVEdge(atoi(str[i][1].c_str()), atoi(str[i][2].c_str()), atoi(str[i][3].c_str()), atoi(str[i][4].c_str()), atof(str[i][5].c_str()), atof(str[i][6].c_str()), atof(str[i][7].c_str())));
	    } else {
		haarFeatures.push_back(new HaarVEdge(atoi(str[i][1].c_str()), atoi(str[i][2].c_str()), atoi(str[i][3].c_str()), atoi(str[i][4].c_str())));
	    }
	}
	else if (name == "HaarHLine")
	{
	    if (str[i].size() > 5) {
		haarFeatures.push_back(new HaarHLine(atoi(str[i][1].c_str()), atoi(str[i][2].c_str()), atoi(str[i][3].c_str()), atoi(str[i][4].c_str()), atof(str[i][5].c_str()), atof(str[i][6].c_str()), atof(str[i][7].c_str())));
	    } else {
		haarFeatures.push_back(new HaarHLine(atoi(str[i][1].c_str()), atoi(str[i][2].c_str()), atoi(str[i][3].c_str()), atoi(str[i][4].c_str())));
	    }
	}
	else if (name == "HaarVLine")
	{
	    if (str[i].size() > 5) {
		haarFeatures.push_back(new HaarVLine(atoi(str[i][1].c_str()), atoi(str[i][2].c_str()), atoi(str[i][3].c_str()), atoi(str[i][4].c_str()), atof(str[i][5].c_str()), atof(str[i][6].c_str()), atof(str[i][7].c_str())));
	    } else {
		haarFeatures.push_back(new HaarVLine(atoi(str[i][1].c_str()), atoi(str[i][2].c_str()), atoi(str[i][3].c_str()), atoi(str[i][4].c_str())));
	    }
	}
	else
	{
	    std::cerr << "Error: cannot load Haar-like Feature" << std::endl;
	}
    }
    return haarFeatures;
}

void saveHaarFeatures(const char *filename, std::vector<Haar*> &haarFeatures)
{
    std::ofstream fout(filename);
    for (size_t i = 0; i < haarFeatures.size(); ++i)
    {
	fout << haarFeatures[i]->name() << " " << haarFeatures[i]->x() << " "
	     << haarFeatures[i]->y() << " " << haarFeatures[i]->width() << " "
	     << haarFeatures[i]->height() << std::endl;
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
