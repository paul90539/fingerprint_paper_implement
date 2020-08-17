#ifndef UTILITY_H
#define UTILITY_H
#define WIN64

#include <string>
#include <vector>

// �t�ΩʡB�\���function���|��b�o��

// for test function => AREA_LIMIT_MATCHING
bool sortLengthFunc(std::vector<int> a, std::vector<int> b);

/******************************
folder        : �ؼи�Ƨ�
Output        : ��Ƨ����U�Ҧ����ɦW

�ت�          : ���o�ؼи�Ƨ����U�Ҧ����ɦW
******************************/
std::vector<std::string> get_all_files_names_within_folder(std::string folder);

/******************************
*p            : �ؼЦr��
sp            : �n���Ϊ��ؼЦr��
Output        : ���Ϋ᪺�r��

�ت�          : ���Φr��
******************************/
std::vector<std::string> split(const char * p, char sp);

/******************************
path          : �ؼ�txt��
Output        : txt�ɪ����e�r��

�ت�          : Ū�Jtxt�ɤ��e�r��
******************************/
std::vector<std::string> getList(std::string path);

/******************************
csvPath         : �ؼ�csv��
removeFirstLine : �O�_�n�����Ĥ@��
Output          : csv�ɪ��r�ꤺ�e

�ت�            : Ū�Jcsv�ɪ��r�ꤺ�e
******************************/
std::vector<std::vector<std::string>> csvLoader(std::string csvPath, bool removeFirstLine = true);

/******************************
inputVec        : [ Input] �ؼ��ɦW�r��
subVec          : [ Input] ���O�d�����ɦW�r��
Output          : [Output] �L�o�᪺�ɦW�r��

�ت�            : Ū�Jcsv�ɪ��r�ꤺ�e
******************************/
std::vector<std::string> checkSubName(std::vector<std::string> inputVec, std::vector<std::string> subVec);

/******************************
t               : [ Input] ��J��int�}�C
*a              : [Output] �}�C��������
*s              : [Output] �}�C���зǮt

�ت�            : �p��vector<int>�������ƻP�зǮt
******************************/
void averageStandard(std::vector<int> t, std::vector<double> * a, std::vector<double> * s);

/******************************
t               : [ Input] ��J��double�}�C
*a              : [Output] �}�C��������
*s              : [Output] �}�C���зǮt

�ت�            : �p��vector<double>�������ƻP�зǮt
******************************/
void averageStandard(std::vector<double> t, std::vector<double> * a, std::vector<double> * s);

#endif