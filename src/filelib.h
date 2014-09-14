#ifndef FILELIB_H
#define FILELIB_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace file
{
    std::vector<std::string> split(const std::string &str, const char delim);
    
    template<typename T>
    std::vector<std::vector<T> > loadfile(const char *filename, const char delim = ' ', bool verbose = false)
    {
	std::ifstream ifs(filename);
	std::vector<std::vector<T> > data;
	std::string str;
	int count = 0;
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
	    count++;
	    if (verbose)
	    {
		if (count % 100 == 0)
		{
		    std::cout << count << std::endl;
		}
	    }
	}
	
	return data;
    };
    
    std::vector<std::string> loadfile(const char *filename);
};

#endif
