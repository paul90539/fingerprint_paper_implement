#include <iostream>
#include <opencv2/opencv.hpp>

#include <direct.h>
#include <io.h>

#include "fingerprint_config.h"
#include "utility.h"


#include "yue_li_implement_2019.h"

/* ================================================================
論文名: ORB-based Fingerprint Matching Algorithm for Mobile Devices
年分  : 2019
作者  : Y. Li, G. Shi
Link  : https://ieeexplore.ieee.org/document/8989155
================================================================ */

using std::cin;
using std::cout;
using std::endl;

bool matchSortYueLi(std::vector< cv::DMatch > a, std::vector< cv::DMatch > b) { return (a[0].distance < b[0].distance); }
bool errorScoreSortYueLi(double a, double b) { return (a < b); }
double calcEuclideanDistanceYueLi(cv::Point2f a, cv::Point2f b) { return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)); }

// detectMode false => detector 與 extractor 分開
// detectMode last  => detector 與 extractor 合併
matcherYueLi::matcherYueLi(fingerprintOptions* options, bool detectMode = false) : _detectMode(detectMode), _options(options) {

	_feature_detector = cv::ORB::create();
	_feature_extractor = cv::ORB::create();
}

//first is qurey,second is train.
std::vector<std::vector<cv::DMatch>> matcherYueLi::_orbKnnMatch(cv::Mat desc_find, cv::Mat desc_data, int k = 2) {

	std::vector<std::vector<cv::DMatch>> output_matches;
	cv::BFMatcher matcher(cv::NORM_HAMMING, false);
	matcher.knnMatch(desc_find, desc_data, output_matches, 2);

	return output_matches;
}

double matcherYueLi::compareYueLi(cv::Mat tempImage, cv::Mat validImage, cv::Mat& resultImage = cv::Mat()) {

	std::vector<cv::KeyPoint> tempKeyPoint, validKeyPoint;
	cv::Mat tempDescriptor, validDescriptor;
	std::vector< cv::DMatch > desMatchesResult;

	std::vector< cv::DMatch > goodMatches;
	std::vector< std::vector< cv::DMatch > > matchesResult;

	desMatchesResult.clear();

	if (!_detectMode) {
		_feature_detector->detect(validImage, validKeyPoint);
		_feature_extractor->compute(validImage, validKeyPoint, validDescriptor);
		_feature_detector->detect(tempImage, tempKeyPoint);
		_feature_extractor->compute(tempImage, tempKeyPoint, tempDescriptor);
	}
	else {
		_feature_detector->detectAndCompute(validImage, cv::noArray(), validKeyPoint, validDescriptor);
		_feature_detector->detectAndCompute(tempImage, cv::noArray(), tempKeyPoint, tempDescriptor);
	}

	matchesResult = _orbKnnMatch(tempDescriptor, validDescriptor, 2);


	bool inSixArea = false;

	std::sort(matchesResult.begin(), matchesResult.end(), matchSortYueLi);

	for (int index = 0; index < matchesResult.size(); index++) {
		inSixArea = false;

		cv::DMatch matchePair = matchesResult[index][0];

		if (desMatchesResult.size() == 0) desMatchesResult.push_back(matchePair);
		else if (desMatchesResult.size() == 15) {
			break;
		}
		else {
			for (int desIdx = 0; desIdx < desMatchesResult.size(); desIdx++) {
				cv::Point2f newP = tempKeyPoint[matchePair.queryIdx].pt;
				cv::Point2f dataP = tempKeyPoint[desMatchesResult[desIdx].queryIdx].pt;

				if (std::abs(newP.x - dataP.x) <= 6.0 && std::abs(newP.y - dataP.y) <= 6.0) {
					inSixArea = true;
					break;
				}
			}
			if (!inSixArea) {
				desMatchesResult.push_back(matchePair);
			}
		}
	}

	//mask => 0 是濾除區域, 1 是保留區域
	std::vector< cv::DMatch > sixMatchesResult;
	std::vector<char> sixPointMask(desMatchesResult.size(), 1);
	std::vector<double> errorScore(desMatchesResult.size(), 0);
	std::vector<std::vector<double>> distanceDiff(desMatchesResult.size(), std::vector<double>(desMatchesResult.size(), 0));
	//cv::Mat distanceDiff = cv::Mat::zeros(cv::Size(desMatchesResult.size(), desMatchesResult.size()), CV_64FC1);
	std::vector<cv::KeyPoint> tempKPft, validKPft;
	tempKPft.clear();
	validKPft.clear();
	sixMatchesResult.clear();

	int countM = desMatchesResult.size();

	// topology 過濾剩六點 
	for (int desIdx = 0; desIdx < desMatchesResult.size(); desIdx++) {
		tempKPft.push_back(tempKeyPoint[desMatchesResult[desIdx].queryIdx]);
		validKPft.push_back(tempKeyPoint[desMatchesResult[desIdx].trainIdx]);
	}

	for (int iy = 0; iy < desMatchesResult.size() - 1; iy++) {
		for (int ix = iy + 1; ix < desMatchesResult.size(); ix++) {
			distanceDiff[iy][ix] = std::abs(calcEuclideanDistanceYueLi(tempKPft[iy].pt, tempKPft[ix].pt) - calcEuclideanDistanceYueLi(validKPft[iy].pt, validKPft[ix].pt));
			distanceDiff[ix][iy] = distanceDiff[iy][ix];
			errorScore[iy] = errorScore[iy] + distanceDiff[iy][ix];
			errorScore[ix] = errorScore[ix] + distanceDiff[ix][iy];
		}
	}

	int maxErrorIdx = 0;
	double maxErrorScore = 0;
	int k = 6;

	//std::sort(errorScore.begin(), errorScore.end(), errorScoreSortYueLi);
	//sixMatchesResult.assign()

	while (countM > k) {

		maxErrorIdx = 0;
		maxErrorScore = 0.0;
		for (int idx = 0; idx < desMatchesResult.size(); idx++) {
			if (sixPointMask[idx] == 0) continue;
			if (errorScore[idx] > maxErrorScore) {
				maxErrorScore = errorScore[idx];
				maxErrorIdx = idx;
			}
		}

		sixPointMask[maxErrorIdx] = 0;

		//for (int idx = 0; idx < desMatchesResult.size(); idx++) {
		//	if (sixPointMask[idx] == 0) continue;
		//	errorScore[idx] = errorScore[idx] - distanceDiff[maxErrorIdx][idx];
		//}
		countM -= 1;
	}

	double scoreText, scoreTopo, scoreTotal, wt;
	scoreText = 0.0;
	scoreTopo = 0.0;
	wt = 0.04;

	for (int desIdx = 0; desIdx < desMatchesResult.size(); desIdx++) {
		if (sixPointMask[desIdx] == 0) continue;
		scoreText += desMatchesResult[desIdx].distance;
		//cout << "Distance: " << desMatchesResult[desIdx].distance << endl;
	}
	scoreText /= k;

	for (int iy = 0; iy < desMatchesResult.size() - 1; iy++) {
		for (int ix = iy + 1; ix < desMatchesResult.size(); ix++) {
			if (sixPointMask[iy] == 0 || sixPointMask[ix] == 0) continue;
			scoreTopo += distanceDiff[iy][ix];
		}
	}

	//C6取2
	scoreTopo /= (k * 5 / 2);
	scoreTotal = wt * scoreText + (1 - wt) * scoreTopo;

	//cout << "scoreTotal: " << scoreTotal << endl;

	return scoreTotal;
}


// 初始化並篩選enroll image
int initFolderCacheYueLi(const std::string targetFolderName, const std::string verifyFolderName, fingerprintOptions* _options, bool ramdomFlag = true, int verifyValue = -1) {

	bool sameFolderFlag = targetFolderName.compare(verifyFolderName) == 0 ? true : false;
	cout << "sameFolderFlag: " << sameFolderFlag << endl;;

	// IQ分數輸出
	std::fstream iqTotalCsv, iqDropCsv;
	iqTotalCsv.open(_options->outputFolderPath + "\\iq_total_result.csv", std::ios::out);
	iqDropCsv.open(_options->outputFolderPath + "\\iq_drop_result.csv", std::ios::out);
	iqDropCsv << "FolderName,totalImageCount,dropImageCount" << endl;
	iqTotalCsv << "iq Score, ImagePath" << endl;

	// cache folder create
	cv::Size imageSize = cv::Size(160, 160);
	std::string cacheFolderName = targetFolderName + "_cache";
	std::string saveFolderName = targetFolderName + "_preprocessing";
	_mkdir(cacheFolderName.c_str());

	std::string subCacheFolderPath;
	std::string subImageFolderPath;
	std::string subImageVerifyFolderPath;
	std::string subSaveFolderPath;

	std::vector<std::string> subFolderName;
	std::vector<std::string> subVerifyFolderName;

	std::vector<std::string> filesName;
	std::vector<std::string> filesVerifyName;

	std::string enrollFolderPath;
	std::string verifyFolderPath;

	std::vector<std::string> subName = { ".bmp", ".BMP", ".jpg", ".JPG", "png", "PNG" };

	subFolderName = get_all_files_names_within_folder(targetFolderName);
	if (!sameFolderFlag) subVerifyFolderName = get_all_files_names_within_folder(verifyFolderName);

	for (int i = 0; i < subFolderName.size(); i++) {
		if (subFolderName[i] == "base") continue;
		subCacheFolderPath = cacheFolderName + "\\" + subFolderName[i];
		subImageFolderPath = targetFolderName + "\\" + subFolderName[i];
		if (!sameFolderFlag) subImageVerifyFolderPath = verifyFolderName + "\\" + subVerifyFolderName[i];
		_mkdir(subCacheFolderPath.c_str());


		filesName = get_all_files_names_within_folder(subImageFolderPath);
		if (!sameFolderFlag) filesVerifyName = get_all_files_names_within_folder(subImageVerifyFolderPath);

		filesName = checkSubName(filesName, subName);
		if (!sameFolderFlag) filesVerifyName = checkSubName(filesVerifyName, subName);

		if (ramdomFlag) std::random_shuffle(filesName.begin(), filesName.end());
		if (ramdomFlag && !sameFolderFlag) std::random_shuffle(filesVerifyName.begin(), filesVerifyName.end());

		std::vector<std::string> enrollFilesName(0, "");
		std::vector<std::string> verifyFilesName(0, "");
		std::vector<std::string> allVerifyFilesName(0, "");

		std::vector<std::string>::iterator filesIt;
		filesIt = filesName.begin();

		int verifyCount = 0, dropCount = 0, totalCount = 0;
		enrollFolderPath = subImageFolderPath;

		//======================================

		if (verifyValue < 0) {
			verifyValue = filesName.size();
		}

		for (int image_index = 0; image_index < filesName.size(); image_index++)
		{
			cv::Mat enrollImage = cv::imread(enrollFolderPath + "\\" + filesName[image_index], CV_LOAD_IMAGE_GRAYSCALE);

			if (enrollFilesName.size() == 16) {
				if (verifyFilesName.size() < verifyValue) {
					verifyFilesName.push_back(filesName[image_index]);
				}
				allVerifyFilesName.push_back(filesName[image_index]);
			}

			else{
				enrollFilesName.push_back(filesName[image_index]);
			}

		}
		// ====================================================================================================


		std::string enrollFilePath = subCacheFolderPath + "\\" + "enroll_file.txt";
		std::fstream enrollStream;
		enrollStream.open(enrollFilePath, std::ios::out);
		for (int index = 0; index < enrollFilesName.size(); index++) {
			if (index != 0) enrollStream << endl;
			enrollStream << subImageFolderPath << "\\" << enrollFilesName[index];
		}
		enrollStream.close();

		std::string verifyFilePath = subCacheFolderPath + "\\" + "verify_file.txt";
		std::fstream verifyStream;
		verifyStream.open(verifyFilePath, std::ios::out);
		for (int index = 0; index < verifyFilesName.size(); index++) {
			if (index != 0) verifyStream << endl;
			if (sameFolderFlag) verifyStream << subImageFolderPath << "\\" << verifyFilesName[index];
			else verifyStream << verifyFolderPath << "\\" << verifyFilesName[index];

		}
		verifyStream.close();

		std::string allVerifyFilePath = subCacheFolderPath + "\\" + "all_verify_file.txt";
		std::fstream allVerifyStream;
		allVerifyStream.open(allVerifyFilePath, std::ios::out);
		for (int index = 0; index < allVerifyFilesName.size(); index++) {
			if (index != 0) allVerifyStream << endl;
			if (sameFolderFlag) allVerifyStream << subImageFolderPath << "\\" << allVerifyFilesName[index];
			else allVerifyStream << verifyFolderPath << "\\" << allVerifyFilesName[index];
		}
		allVerifyStream.close();

	}
	return 0;
}

void allFolderMatchingYueLi(const std::string cacheFolderPath, fingerprintOptions* _options) {

	std::vector<std::string> subFolderName = get_all_files_names_within_folder(cacheFolderPath);

	matcherYueLi * orbMatch = new matcherYueLi(_options, false);

	//output file
	std::fstream totalResultCSV;
	totalResultCSV.open(_options->outputFolderPath + "\\compare_result.csv", std::ios::out);

	totalResultCSV << "Score,ScoreT,E_Folder,Enroll,V_Folder,Verify" << endl;

	for (int enrollFolderIndex = 0; enrollFolderIndex < subFolderName.size(); enrollFolderIndex++) {
		//image list load
		std::string enrollFilePath = cacheFolderPath + "\\" + subFolderName[enrollFolderIndex] + "\\enroll_file.txt";
		std::vector<std::string> enrollList = getList(enrollFilePath);

		cv::Size imageSize = cv::Size(160, 160);
		double outerStart = (double)cv::getTickCount();

		for (int verifyFolderIndex = 0; verifyFolderIndex < subFolderName.size(); verifyFolderIndex++) {

			double start = (double)cv::getTickCount();

			bool sameFlag = false;
			if (subFolderName[enrollFolderIndex] == subFolderName[verifyFolderIndex]) {
				sameFlag = true;
			}

			cout << "===== Now Processing =====" << endl << "enroll: " << subFolderName[enrollFolderIndex] << endl << "verify: " << subFolderName[verifyFolderIndex] << endl;
			//image list load
			std::string verifyFilePath;
			if (sameFlag) verifyFilePath = cacheFolderPath + "\\" + subFolderName[verifyFolderIndex] + "\\all_verify_file.txt";
			else verifyFilePath = cacheFolderPath + "\\" + subFolderName[verifyFolderIndex] + "\\verify_file.txt";

			std::vector<std::string> verifyList = getList(verifyFilePath);

			//matching strating
			for (int verifyElement = 0; verifyElement < verifyList.size(); verifyElement++) {

				//score parameter
				double totalScore = 0, minScore = 10000;
				std::string maxScoreEnrollImageName, maxScoreVerifyImageName;
				maxScoreVerifyImageName = verifyList[verifyElement];
				std::vector<double> enrollScoreStore;

				//cout << "verifyList[" << verifyElement << "]: " << verifyList[verifyElement] << endl;

				for (int enrollElement = 0; enrollElement < enrollList.size(); enrollElement++) {
					cv::Mat enrollImage, verifyImage, resultImage;
					enrollImage = cv::imread(enrollList[enrollElement], CV_LOAD_IMAGE_GRAYSCALE);
					verifyImage = cv::imread(verifyList[verifyElement], CV_LOAD_IMAGE_GRAYSCALE);
					//cv::resize(enrollImage, enrollImage, cv::Size(enrollImage.cols * 2, enrollImage.rows * 2), 0, 0, cv::INTER_CUBIC);
					//cv::resize(verifyImage, verifyImage, cv::Size(enrollImage.cols * 2, enrollImage.rows * 2), 0, 0, cv::INTER_CUBIC);
					double currentScore = orbMatch->compareYueLi(enrollImage, verifyImage, resultImage);
					//int currentScore = akazeMatch->compare_normal(enrollImage, verifyImage, resultImage);

					//cout << "currentScore: " << currentScore << endl;
					totalScore += currentScore;
					if (currentScore < minScore) {
						minScore = currentScore;
						maxScoreEnrollImageName = enrollList[enrollElement];
					}
					enrollScoreStore.emplace_back(currentScore);
					//system("PAUSE");
				}
				//cout << "maxScore: " << maxScore << endl;

				//score save
				totalResultCSV << minScore << ", " << totalScore << ", " << subFolderName[enrollFolderIndex] << ", "
					<< maxScoreEnrollImageName << ", " << subFolderName[verifyFolderIndex] << ", "
					<< maxScoreVerifyImageName << endl;
			}

			double end = (double)cv::getTickCount();
			double spendTime = (end - start) / cv::getTickFrequency();
			cout << "spendTime: " << spendTime << " s" << endl;
		}

		double outerEnd = (double)cv::getTickCount();
		double outerSpendTime = (outerEnd - outerStart) / cv::getTickFrequency();
		cout << "Finish one finger spend time: " << (int)std::floor(outerSpendTime / 60.0) << " m " << (int)std::ceil(outerSpendTime) % 60 << " s" << endl;
		double ETATime = outerSpendTime * (subFolderName.size() - enrollFolderIndex);
		cout << "ETA Folder: " << enrollFolderIndex + 1 << "/" << subFolderName.size() << endl;
		cout << "ETA time: " << (int)std::floor(ETATime / 3600.0) << " h " << (int)std::floor(((int)std::floor(ETATime) % 3600) / 60.0) << " m " << (int)std::ceil(ETATime) % 60 << " s" << endl;
	}

	totalResultCSV.close();
}

/*
論文名: ORB-based Fingerprint Matching Algorithm for Mobile Devices
年分  : 2019
作者  : Y. Li, G. Shi
Link  : https://ieeexplore.ieee.org/document/8989155
*/
void startYueLiTest(std::string targetFolderName, std::string outputFolderTitle) {

	fingerprintOptions options;
	cv::Size image_size = cv::Size(160, 160);


	std::string cacheFolderPath = targetFolderName + "_cache";

	struct tm now_time;
	time_t ltm;
	time(&ltm);
	localtime_s(&now_time, &ltm);
	char timeStr[1024] = "";

	sprintf_s(timeStr, "%d_%d_%d_%d_%d_%d", now_time.tm_year + 1900, now_time.tm_mon + 1, now_time.tm_mday, now_time.tm_hour, now_time.tm_min, now_time.tm_sec);
	options.outputFolderPath = targetFolderName + "_" + outputFolderTitle + "_" + timeStr;
	_mkdir(options.outputFolderPath.c_str());

	cout << "init start" << endl;

	//std::string verifyFolderName = ".\\data\\12_0521_FTS\\dry";
	std::string verifyFolderName = targetFolderName;

	initFolderCacheYueLi(targetFolderName, verifyFolderName, &options, false, 10);
	allFolderMatchingYueLi(cacheFolderPath, &options);
}