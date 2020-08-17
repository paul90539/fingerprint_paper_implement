#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <direct.h>
#include <io.h>

#include "utility.h"

#include ".\surbhi_mathur_implement_2016.h"
#include ".\yue_li_implement_2019.h"

using std::cin;
using std::cout;
using std::endl;

int main(int argc, char** argv) {



	std::string targetFolderName;
	if (argc > 1) {
		std::vector<std::string> tempStr = split(argv[1], '\\');
		targetFolderName = tempStr[tempStr.size() - 1];
	}
	else {
		targetFolderName = ".\\data"; // 指紋資料集位置
	}
	std::string folderTitle = "result"; // 輸出數據的資料夾名稱

	//testKnn();
	double totalTimeStart = (double)cv::getTickCount();

	//startSMathurTest(targetFolderName, folderTitle); // 2016 samung 方法
	//startYueLiTest(targetFolderName, folderTitle);   // 2019 yueli 方法

	// 計算總共比對時間
	double totalTimeEnd = (double)cv::getTickCount();
	double totalSpendTime = (totalTimeEnd - totalTimeStart) / cv::getTickFrequency();
	cout << "total spending time: " << (int)std::floor(totalSpendTime / 3600.0) << " h " << (int)((int)std::floor(totalSpendTime / 60.0) % 60) << " m " << (int)std::ceil(totalSpendTime) % 60 << " s" << endl;
	
	system("PAUSE");
	return 0;
}