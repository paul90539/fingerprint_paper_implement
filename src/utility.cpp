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
folder        : [ Input] �ؼи�Ƨ�
Output        : [Output] ��Ƨ����U�Ҧ����ɦW

�ت�          : ���o�ؼи�Ƨ����U�Ҧ����ɦW
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
*p            : [ Input] �ؼЦr��
sp            : [ Input] �n���Ϊ��ؼЦr��
Output        : [Output] ���Ϋ᪺�r��

�ت�          : ���Φr��
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
path          : [ Input] �ؼ�txt��
Output        : [Output] txt�ɪ����e�r��

�ت�          : Ū�Jtxt�ɤ��e�r��
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
csvPath         : [ Input] �ؼ�csv��
removeFirstLine : [ Input] �O�_�n�����Ĥ@��
Output          : [Output] csv�ɪ��r�ꤺ�e

�ت�            : Ū�Jcsv�ɪ��r�ꤺ�e
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
inputVec        : [ Input] �ؼ��ɦW�r��
subVec          : [ Input] ���O�d�����ɦW�r��
Output          : [Output] �L�o�᪺�ɦW�r��

�ت�            : Ū�Jcsv�ɪ��r�ꤺ�e
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
t               : [ Input] ��J��int�}�C
*a              : [Output] �}�C��������
*s              : [Output] �}�C���зǮt

�ت�            : �p��vector<int>�������ƻP�зǮt
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
t               : [ Input] ��J��double�}�C
*a              : [Output] �}�C��������
*s              : [Output] �}�C���зǮt

�ت�            : �p��vector<double>�������ƻP�зǮt
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