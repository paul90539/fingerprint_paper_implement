#ifndef UTILITY_H
#define UTILITY_H
#define WIN64

#include <string>
#include <vector>

// 系統性、功能性function都會放在這裡

// for test function => AREA_LIMIT_MATCHING
bool sortLengthFunc(std::vector<int> a, std::vector<int> b);

/******************************
folder        : 目標資料夾
Output        : 資料夾底下所有的檔名

目的          : 取得目標資料夾底下所有的檔名
******************************/
std::vector<std::string> get_all_files_names_within_folder(std::string folder);

/******************************
*p            : 目標字串
sp            : 要分割的目標字元
Output        : 分割後的字串

目的          : 分割字串
******************************/
std::vector<std::string> split(const char * p, char sp);

/******************************
path          : 目標txt檔
Output        : txt檔的內容字串

目的          : 讀入txt檔內容字串
******************************/
std::vector<std::string> getList(std::string path);

/******************************
csvPath         : 目標csv檔
removeFirstLine : 是否要移除第一行
Output          : csv檔的字串內容

目的            : 讀入csv檔的字串內容
******************************/
std::vector<std::vector<std::string>> csvLoader(std::string csvPath, bool removeFirstLine = true);

/******************************
inputVec        : [ Input] 目標檔名字串
subVec          : [ Input] 欲保留的副檔名字串
Output          : [Output] 過濾後的檔名字串

目的            : 讀入csv檔的字串內容
******************************/
std::vector<std::string> checkSubName(std::vector<std::string> inputVec, std::vector<std::string> subVec);

/******************************
t               : [ Input] 輸入的int陣列
*a              : [Output] 陣列的平均數
*s              : [Output] 陣列的標準差

目的            : 計算vector<int>的平均數與標準差
******************************/
void averageStandard(std::vector<int> t, std::vector<double> * a, std::vector<double> * s);

/******************************
t               : [ Input] 輸入的double陣列
*a              : [Output] 陣列的平均數
*s              : [Output] 陣列的標準差

目的            : 計算vector<double>的平均數與標準差
******************************/
void averageStandard(std::vector<double> t, std::vector<double> * a, std::vector<double> * s);

#endif