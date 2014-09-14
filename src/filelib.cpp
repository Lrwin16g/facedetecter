#include "filelib.h"

std::vector<std::string> file::split(const std::string &str, const char delim)
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

std::vector<std::string> file::loadfile(const char *filename)
{
    std::vector<std::string> list;
    std::ifstream ifs(filename);
    std::string line;
    while (std::getline(ifs, line))
    {
	list.push_back(line);
    }
    
    return list;
}
