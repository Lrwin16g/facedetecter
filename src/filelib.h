#ifndef FILELIB_H
#define FILELIB_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace file
{
    std::vector<std::string> split(const std::string &str, const char *delim);

    std::vector<std::string> loadfile(const char *filename);

    std::vector<std::vector<std::string> > loadfile(const char *filename, const char *delim);
    
    template<typename T>
    std::vector<T> loadfile(const char *filename, bool isBin)
    {
	std::ifstream ifs;
	std::vector<T> data;
	if (isBin)
	{
	    ifs.open(filename, std::ios::binary);
	    T elem;
	    while (true)
	    {
		ifs.read(reinterpret_cast<char*>(&elem), sizeof(T));
		if (ifs.eof()) {
		    break;
		}
		data.push_back(elem);
	    }
	} else {
	    ifs.open(filename);
	    std::string line;
	    while (std::getline(ifs, line))
	    {
		std::istringstream iss(line);
		T elem;
		iss >> elem;
		data.push_back(elem);
	    }
	}
	ifs.close();
	
	return data;
    };
    
    template<typename T>
    std::vector<std::vector<T> > loadfile(const char *filename, const char *delim, bool verbose)
    {
	std::ifstream ifs(filename);
	std::vector<std::vector<T> > data;
	std::string str;
	while (std::getline(ifs, str))
	{
	    std::vector<std::string> token = split(str, delim);
	    std::vector<T> elem(token.size());
	    for (size_t i = 0; i < token.size(); ++i)
	    {
		std::istringstream iss(token[i]);
		T value;
		iss >> value;
		elem[i] = value;
	    }
	    data.push_back(elem);
	    
	    if (verbose) {
		if (data.size() % 100 == 0) {
		    std::cout << "loading: " << filename << " " << data.size() << "/***" << std::endl;
		}
	    }
	}
	
	return data;
    };
    
    template<typename T>
    std::vector<std::vector<T> > loadfile(const char *filename, int col, bool verbose)
    {
	std::ifstream ifs(filename, std::ios::binary);
	std::vector<std::vector<T> > data;
	while (true)
	{
	    std::vector<T> line(col);
	    for (int i = 0; i < col; ++i)
	    {
		T elem;
		ifs.read(reinterpret_cast<char*>(&elem), sizeof(T));
		line[i] = elem;
	    }
	    if (ifs.eof()) {
		break;
	    }
	    data.push_back(line);
	    
	    if (verbose) {
		if (data.size() % 100 == 0) {
		    std::cout << "loading: " << filename << " " << data.size() << "/***" << std::endl;
		}
	    }
	}
	ifs.close();
	
	return data;
    };
    
    template<typename T>
    void savefile(const char *filename, const std::vector<T> &data, bool isBin)
    {
	std::ofstream ofs;
	if (isBin)
	{
	    ofs.open(filename, std::ios::binary);
	    for (size_t i = 0; i < data.size(); ++i)
	    {
		T elem = data[i];
		ofs.write(reinterpret_cast<char*>(&elem), sizeof(T));
	    }
	} else {
	    ofs.open(filename);
	    for (size_t i = 0; i < data.size(); ++i)
	    {
		ofs << data[i] << std::endl;
	    }
	}
	ofs.close();
    };
    
    template<typename T>
    void savefile(const char *filename, const std::vector<std::vector<T> > &data, bool isBin, const char *delim = " ", bool verbose = false)
    {
	std::ofstream ofs;
	if (isBin)
	{
	    ofs.open(filename, std::ios::binary);
	    for (size_t i = 0; i < data.size(); ++i)
	    {
		for (size_t j = 0; j < data[i].size(); ++j)
		{
		    T elem = data[i][j];
		    ofs.write(reinterpret_cast<char*>(&elem), sizeof(T));
		}
		
		if (verbose) {
		    if (i % 100 == 0) {
			std::cout << "saving: " << filename << " " << i << "/" << data.size() << std::endl;
		    }
		}
	    }
	} else {
	    ofs.open(filename);
	    for (size_t i = 0; i < data.size(); ++i)
	    {
		for (size_t j = 0; j < data[i].size() - 1; ++j)
		{
		    ofs << data[i][j] << delim;
		}
		ofs << data[i][data[i].size() - 1] << std::endl;
		
		if (verbose) {
		    if (i % 100 == 0) {
			std::cout << "saving: " << filename << " " << i << "/" << data.size() << std::endl;
		    }
		}
	    }
	}
	ofs.close();
    };
    
    template<typename T>
    void savefile(const char *filename, const T *data, int row, bool isBin)
    {
	std::ofstream ofs;
	if (isBin)
	{
	    ofs.open(filename, std::ios::binary);
	    for (int i = 0; i < row; ++i)
	    {
		T elem = data[i];
		ofs.write(reinterpret_cast<char*>(&elem), sizeof(T));
	    }
	} else {
	    ofs.open(filename);
	    for (int i = 0; i < row; ++i)
	    {
		ofs << data[i] << std::endl;
	    }
	}
	ofs.close();
    };
    
    template<typename T>
    void savefile(const char *filename, T const * const * data, int row, int col, bool isBin, const char *delim = " ", bool verbose = false)
    {
	std::ofstream ofs;
	if (isBin)
	{
	    ofs.open(filename, std::ios::binary);
	    for (int i = 0; i < row; ++i)
	    {
		for (int j = 0; j < col; ++j)
		{
		    T elem = data[i][j];
		    ofs.write(reinterpret_cast<char*>(&elem), sizeof(T));
		}
		
		if (verbose) {
		    if (i % 100 == 0) {
			std::cout << "saving: " << filename << " " << i << "/" << row << std::endl;
		    }
		}
	    }
	} else {
	    ofs.open(filename);
	    for (int i = 0; i < row; ++i)
	    {
		for (int j = 0; j < col - 1; ++j)
		{
		    ofs << data[i][j] << delim;
		}
		ofs << data[i][col - 1] << std::endl;
		
		if (verbose) {
		    if (i % 100 == 0) {
			std::cout << "saving: " << filename << " " << i << "/" << row << std::endl;
		    }
		}
	    }
	}
	ofs.close();
    };
};

#endif
