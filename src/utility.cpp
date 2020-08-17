#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#include "utility.h"

#ifdef WIN64
#include <io.h>
#endif

// for test function => AREA_LIMIT_MATCHING
bool sortLengthFunc(std::vector<int> a, std::vector<int> b) {
	return a.size() > b.size();
}

/******************************
folder        : [ Input] 目標資料夾
Output        : [Output] 資料夾底下所有的檔名

目的          : 取得目標資料夾底下所有的檔名
******************************/
std::vector<std::string> get_all_files_names_within_folder(std::string folder)
{
	std::vector<std::string> files;

#ifdef WIN64

	_finddata_t file;
	intptr_t lf;
	std::string name = folder + "\\" + "*";

	if ((lf = _findfirst(name.c_str(), &file)) == -1) {
		std::cout << name << " not found!!!" << std::endl;
	}
	else {
		do {
			if (strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0)
				files.push_back(file.name);
			//cout << file.name << endl;
			//system("PAUSE");
		} while (_findnext(lf, &file) == 0);
	}
	_findclose(lf);

#endif
	//std::sort(files.begin(), files.end(), sortAlgo);
	return files;
}


/******************************
*p            : [ Input] 目標字串
sp            : [ Input] 要分割的目標字元
Output        : [Output] 分割後的字串

目的          : 分割字串
******************************/
std::vector<std::string> split(const char * p, char sp) {
	std::stringstream ss(p);
	std::string line;

	std::vector<std::string> outStr;
	outStr.clear();

	while (std::getline(ss, line, sp)) {
		outStr.emplace_back(line);
	}
	return outStr;
}

/******************************
path          : [ Input] 目標txt檔
Output        : [Output] txt檔的內容字串

目的          : 讀入txt檔內容字串
******************************/
std::vector<std::string> getList(std::string path) {
	std::ifstream fpStream(path);
	std::vector<std::string> allContent;
	for (std::string lineStr; std::getline(fpStream, lineStr); ) {
		//char tempStr[1024] = { 0 };
		allContent.emplace_back(lineStr);
	}
	fpStream.close();
	return allContent;
}


/******************************
csvPath         : [ Input] 目標csv檔
removeFirstLine : [ Input] 是否要移除第一行
Output          : [Output] csv檔的字串內容

目的            : 讀入csv檔的字串內容
******************************/
std::vector<std::vector<std::string>> csvLoader(std::string csvPath, bool removeFirstLine) {

	std::ifstream csvStream(csvPath);
	std::vector<std::vector<std::string>> allContent;

	if (removeFirstLine == true) {
		std::string lineStr;
		std::getline(csvStream, lineStr);
	}
	for (std::string lineStr; std::getline(csvStream, lineStr); ) {
		std::vector<std::string> splitLineStr = split(lineStr.c_str(), ',');
		allContent.emplace_back(splitLineStr);
	}
	return allContent;
}


/******************************
inputVec        : [ Input] 目標檔名字串
subVec          : [ Input] 欲保留的副檔名字串
Output          : [Output] 過濾後的檔名字串

目的            : 讀入csv檔的字串內容
******************************/
std::vector<std::string> checkSubName(std::vector<std::string> inputVec, std::vector<std::string> subVec) {
	std::vector<std::string> outputVec;
	for (int inputIndex = 0; inputIndex < inputVec.size(); inputIndex++) {
		for (int subIndex = 0; subIndex < subVec.size(); subIndex++) {

			if (inputVec[inputIndex].find(subVec[subIndex], 0) != std::string::npos) {
				outputVec.emplace_back(inputVec[inputIndex]);
			}
		}
	}
	return outputVec;
}


/******************************
t               : [ Input] 輸入的int陣列
*a              : [Output] 陣列的平均數
*s              : [Output] 陣列的標準差

目的            : 計算vector<int>的平均數與標準差
******************************/
void averageStandard(std::vector<int> t, std::vector<double> * a, std::vector<double> * s) {
	std::vector<int>::iterator it_match;
	double averageMatch = 0, standardMatch = 0;

	for (it_match = t.begin(); it_match != t.end(); it_match++) {
		averageMatch += *it_match;
	}
	averageMatch /= (double)t.size();

	for (it_match = t.begin(); it_match != t.end(); it_match++) {
		standardMatch += pow(((double)*it_match - averageMatch), 2.0) / (double)t.size();
	}

	a->emplace_back(averageMatch);
	s->emplace_back(standardMatch);
}


/******************************
t               : [ Input] 輸入的double陣列
*a              : [Output] 陣列的平均數
*s              : [Output] 陣列的標準差

目的            : 計算vector<double>的平均數與標準差
******************************/
void averageStandard(std::vector<double> t, std::vector<double> * a, std::vector<double> * s) {
	std::vector<double>::iterator it_match;
	double averageMatch = 0, standardMatch = 0;

	for (it_match = t.begin(); it_match != t.end(); it_match++) {
		averageMatch += *it_match;
	}
	averageMatch /= (double)t.size();

	for (it_match = t.begin(); it_match != t.end(); it_match++) {
		standardMatch += pow(((double)*it_match - averageMatch), 2.0) / (double)t.size();
	}

	a->emplace_back(averageMatch);
	s->emplace_back(standardMatch);
}