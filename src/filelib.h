#ifndef FILELIB_H
#define FILELIB_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace file
{
    inline std::vector<std::string> split(const std::string &str, const char *delim)
    {
	std::string::size_type i = 0;
	std::string::size_type j = str.find(delim);
    
	std::vector<std::string> result;
    
	while (j != std::string::npos)
	{
	    result.push_back(str.substr(i, j - i));
	    i = ++j;
	    j = str.find(delim, j);
        
	    if (j == std::string::npos)
	    {
	        result.push_back(str.substr(i, str.length()));
	    }
	}
    
	return result;
    }
    
    inline std::vector<std::string> splitext(const std::string &str)
    {
	std::string::size_type index = str.rfind(".");
    
	std::vector<std::string> result;
	result.push_back(str.substr(0, index));
	result.push_back(str.substr(index, str.length()));
    
	return result;
    }

    inline std::vector<std::string> loadfile(const char *filename)
    {
       std::vector<std::string> data;
       std::ifstream ifs(filename);
       std::string line;
       while (std::getline(ifs, line))
       {
	    data.push_back(line);
       }
       return data;
    }

    inline std::vector<std::vector<std::string> > loadfile(const char *filename, const char *delim)
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
    }
    
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
		    std::cout << "loading: " << filename << " " << data.size() << "/***" << "\r" << std::flush;
		}
	    }
	}
	if (verbose) {
	    std::cout << std::endl;
	}
	
	return data;
    }
    
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
		    std::cout << "loading: " << filename << " " << data.size() << "/***" << "\r" << std::flush;
		}
	    }
	}
	if (verbose) {
	    std::cout << std::endl;
	}
	ifs.close();
	
	return data;
    }
    
    template<typename T>
    std::vector<std::vector<T> > loadmat(const char *filename, bool verbose = false)
    {
	std::ifstream ifs(filename, std::ios::binary);
	
	size_t row, col;
	ifs.read(reinterpret_cast<char*>(&row), sizeof(size_t));
	ifs.read(reinterpret_cast<char*>(&col), sizeof(size_t));
	
	std::vector<std::vector<T> > data(row);
	
	for (size_t i = 0; i < row; ++i)
	{
	    data[i].resize(col);
	    for (size_t j = 0; j < col; ++j)
	    {
		T elem;
		ifs.read(reinterpret_cast<char*>(&elem), sizeof(T));
		data[i][j] = elem;
	    }
	    
	    if (verbose) {
		if (i % 100 == 0) {
		    std::cout << "loading: " << filename << " " << i << "/" << row << "\r" << std::flush;
		}
	    }
	}
	if (verbose) {
	    std::cout << std::endl;
	}
	ifs.close();
	
	return data;
    }
    
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
    }
    
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
			std::cout << "saving: " << filename << " " << i << "/" << data.size() << "\r" << std::flush;
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
			std::cout << "saving: " << filename << " " << i << "/" << data.size() << "\r" << std::flush;
		    }
		}
	    }
	}
	if (verbose) {
	    std::cout << std::endl;
	}
	ofs.close();
    }
    
    template<typename T>
    void savemat(const char *filename, const std::vector<std::vector<T> > &data, bool verbose = false)
    {
	std::ofstream ofs(filename, std::ios::binary);
	
	size_t row = data.size();
	size_t col = data[0].size();
	ofs.write(reinterpret_cast<char*>(&row), sizeof(size_t));
	ofs.write(reinterpret_cast<char*>(&col), sizeof(size_t));
	
	for (size_t i = 0; i < row; ++i)
	{
	    for (size_t j = 0; j < col; ++j)
	    {
		T elem = data[i][j];
		ofs.write(reinterpret_cast<char*>(&elem), sizeof(T));
	    }
	    
	    if (verbose) {
		if (i % 100 == 0) {
		    std::cout << "saving: " << filename << " " << i << "/" << data.size() << "\r" << std::flush;
		}
	    }
	}
	if (verbose) {
	    std::cout << std::endl;
	}
	ofs.close();
    }
    
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
    }
    
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
			std::cout << "saving: " << filename << " " << i << "/" << row << "\r" << std::flush;
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
			std::cout << "saving: " << filename << " " << i << "/" << row << "\r" << std::flush;
		    }
		}
	    }
	}
	if (verbose) {
	    std::cout << std::endl;
	}
	ofs.close();
    }
}

#endif
