#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <direct.h>
#include <io.h>

#include "utility.h"

#include "fingerprint_config.h"

#include "fftm.h"
#include "surbhi_mathur_implement_2016.h"

#define ENROLL_SIZE 16

using std::cin;
using std::cout;
using std::endl;

/*
論文名: Methodology for Partial Fingerprint Enrollmentand Authenticationon Mobile Devices
年分  : 2016
作者  : S. Mathur, A. Vjay, J. Shah, S. Das and A. Malla
Link  : https://ieeexplore.ieee.org/document/7550093
*/

// detectMode false => detector 與 extractor 分開
// detectMode last  => detector 與 extractor 合併
matcherSMathur::matcherSMathur(fingerprintOptions* options, bool detectMode = false) : _detectMode(detectMode), _options(options) {

	_feature_detector  = cv::AKAZE::create();
	_feature_extractor = cv::AKAZE::create();
}

//first is qurey,second is train.
std::vector<std::vector<cv::DMatch>> matcherSMathur::_akazeKnnMatchSMathur(cv::Mat desc_find, cv::Mat desc_data, int k = 2) {

	std::vector<std::vector<cv::DMatch>> output_matches;
	cv::BFMatcher matcher(cv::NORM_HAMMING, false);
	matcher.knnMatch(desc_find, desc_data, output_matches, 2);

	return output_matches;
}

// test function samsung method
bool responseSortSMathur(cv::KeyPoint a, cv::KeyPoint b) { return (a.response > b.response); }

// test function samsung method
bool matchSortSMathur(std::vector< cv::DMatch > a, std::vector< cv::DMatch > b) { return (a[0].distance < b[0].distance); }

// test function samsung method
double calcEuclideanDistanceSMathur(cv::Point2f a, cv::Point2f b) { return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)); }

// test function samsung method
cv::Mat matcherSMathur::clacMaskSMathur(cv::Mat inputImage, cv::Size blockSize) {

	cv::Size imageSize = inputImage.size();

	//mask => 0 是濾除區域, 1 是保留區域
	cv::Mat outputMask = cv::Mat::zeros(imageSize, CV_8UC1);
	cv::Mat local_partition = cv::Mat::zeros(blockSize, CV_8UC1);
	int partition_X = 0, partition_Y = 0, partition_width = 0, partition_height = 0;

	cv::Mat meanResult, stdResult;
	double nowVariance = 0.0, maxVariance = 0.0, thVariance, centerVeriance, meanValue;

	for (int y = 0; y < imageSize.height; y++)
	{
		for (int x = 0; x < imageSize.width; x++)
		{
			maxVariance = 0.0;

			partition_X = x - blockSize.width / 2, partition_Y = y - blockSize.height / 2;
			partition_width = blockSize.width, partition_height = blockSize.height;
			if (partition_X < 0)
			{
				partition_width = blockSize.width + partition_X;
				partition_X = 0;
			}
			if (partition_Y < 0)
			{
				partition_height = blockSize.height + partition_Y;
				partition_Y = 0;
			}
			if ((partition_X + blockSize.width) >(imageSize.width - 1))
			{
				partition_width = imageSize.width - partition_X;
			}
			if ((partition_Y + blockSize.height) > (imageSize.height - 1))
			{
				partition_height = imageSize.height - partition_Y;
			}

			local_partition = cv::Mat(inputImage, cv::Rect(partition_X, partition_Y, partition_width, partition_height));

			meanValue = cv::mean(local_partition)[0];

			for (int ly = 0; ly < local_partition.rows; ly++) {
				for (int lx = 0; lx < local_partition.cols; lx++) {

					nowVariance = (int)local_partition.ptr<uchar>(ly)[lx] - meanValue;
					nowVariance = nowVariance * nowVariance;
					if (nowVariance > maxVariance) {
						maxVariance = nowVariance;
					}
				}
			}

			thVariance = maxVariance * 0.2;
			centerVeriance = inputImage.ptr<uchar>(y)[x] - meanValue;
			centerVeriance = centerVeriance * centerVeriance;
			if (centerVeriance > thVariance) {
				outputMask.ptr<uchar>(y)[x] = 1;
			}
			else {
				outputMask.ptr<uchar>(y)[x] = 0;
			}

		} // x
	} // y

	return outputMask;
}

// test function samsung method
double matcherSMathur::compareSMathur(cv::Mat tempImage, cv::Mat validImage, cv::Mat& resultImage = cv::Mat()) {

	cv::Mat oriTemp = tempImage.clone();

	std::vector<cv::KeyPoint> tempKeyPoint, validKeyPoint;
	cv::Mat tempDescriptor, validDescriptor;

	std::vector< cv::DMatch > goodMatches, ransacMatches, filterAngleMatches;
	std::vector< std::vector< cv::DMatch > > matchesResult;
	std::vector< cv::DMatch > desMatchesResult;

	cv::Mat tempMask = clacMaskSMathur(tempImage, cv::Size(11, 11));
	cv::Mat vaildMask = clacMaskSMathur(validImage, cv::Size(11, 11));

	desMatchesResult.clear();

	if (!_detectMode) {
		_feature_detector->detect(validImage, validKeyPoint, vaildMask);

		std::vector<cv::KeyPoint> temp_kp;
		temp_kp.clear();

		// new kp
		std::sort(validKeyPoint.begin(), validKeyPoint.end(), responseSortSMathur);
		if (validKeyPoint.size() > 300) {
			validKeyPoint.assign(validKeyPoint.begin(), validKeyPoint.begin() + 300);
		}

		_feature_extractor->compute(validImage, validKeyPoint, validDescriptor);
		_feature_detector->detect(tempImage, tempKeyPoint, tempMask);

		// new kp
		std::sort(tempKeyPoint.begin(), tempKeyPoint.end(), responseSortSMathur);
		if (tempKeyPoint.size() > 300) {
			tempKeyPoint.assign(tempKeyPoint.begin(), tempKeyPoint.begin() + 300);
		}


		_feature_extractor->compute(tempImage, tempKeyPoint, tempDescriptor);
	}
	else {
		_feature_detector->detectAndCompute(validImage, cv::noArray(), validKeyPoint, validDescriptor);
		_feature_detector->detectAndCompute(tempImage, cv::noArray(), tempKeyPoint, tempDescriptor);
	}

	matchesResult = _akazeKnnMatchSMathur(tempDescriptor, validDescriptor, 2);

	// 6x6 drop and 15 points
	bool inSixArea = false;

	std::sort(matchesResult.begin(), matchesResult.end(), matchSortSMathur);

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
	std::vector<char> sixPointMask(desMatchesResult.size(), 1);
	std::vector<double> errorScore(desMatchesResult.size(), 0);
	std::vector<std::vector<double>> distanceDiff(desMatchesResult.size(), std::vector<double>(desMatchesResult.size(), 0));
	//cv::Mat distanceDiff = cv::Mat::zeros(cv::Size(desMatchesResult.size(), desMatchesResult.size()), CV_64FC1);
	std::vector<cv::KeyPoint> tempKPft, validKPft;
	tempKPft.clear();
	validKPft.clear();

	int countM = desMatchesResult.size();

	// topology 過濾剩六點 
	for (int desIdx = 0; desIdx < desMatchesResult.size(); desIdx++) {
		tempKPft.push_back(tempKeyPoint[desMatchesResult[desIdx].queryIdx]);
		validKPft.push_back(tempKeyPoint[desMatchesResult[desIdx].trainIdx]);
	}

	for (int iy = 0; iy < desMatchesResult.size() - 1; iy++) {
		for (int ix = iy + 1; ix < desMatchesResult.size(); ix++) {
			distanceDiff[iy][ix] = std::abs(calcEuclideanDistanceSMathur(tempKPft[iy].pt, tempKPft[ix].pt) - calcEuclideanDistanceSMathur(validKPft[iy].pt, validKPft[ix].pt));
			distanceDiff[ix][iy] = distanceDiff[iy][ix];
			errorScore[iy] = errorScore[iy] + distanceDiff[iy][ix];
			errorScore[ix] = errorScore[ix] + distanceDiff[ix][iy];
		}
	}

	int maxErrorIdx = 0;
	double maxErrorScore = 0;
	while (countM > 6) {

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

		for (int idx = 0; idx < desMatchesResult.size(); idx++) {
			if (sixPointMask[idx] == 0) continue;
			errorScore[idx] = errorScore[idx] - distanceDiff[maxErrorIdx][idx];
		}
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
	scoreText /= 6;

	for (int iy = 0; iy < desMatchesResult.size() - 1; iy++) {
		for (int ix = iy + 1; ix < desMatchesResult.size(); ix++) {
			if (sixPointMask[iy] == 0 || sixPointMask[ix] == 0) continue;
			scoreTopo += distanceDiff[iy][ix];
		}
	}

	//C6取2
	scoreTopo /= (6 * 5 / 2);
	scoreTotal = wt * scoreText + (1 - wt) * scoreTopo;

	//cout << "scoreTotal: " << scoreTotal << endl;

	return scoreTotal;
}


double FMTOverlapRatio(cv::Mat im0, cv::Mat im1, double canny_threshold1, double canny_threshold2) {

	double cover_ratio = 0;

	cv::Size image_size = cv::Size(160, 160);
	cv::RotatedRect rotated_rectangle;
	cv::Point2f rectangle_points[4];
	std::vector<cv::Point> contour;

	// get rotate point and fix
	rotated_rectangle = LogPolarFFTTemplateMatch(im0, im1, 200, 100);
	rotated_rectangle.points(rectangle_points);
	for (int j = 0; j < 4; j++)
	{
		if (rectangle_points[j].x < 0)
			rectangle_points[j].x = 0;
		if (rectangle_points[j].y < 0)
			rectangle_points[j].y = 0;
		if (rectangle_points[j].x >= (float)image_size.width)
			rectangle_points[j].x = (float)(image_size.width - 1);
		if (rectangle_points[j].y >= (float)image_size.height)
			rectangle_points[j].y = (float)(image_size.height - 1);
	}

	// compute ratio
	contour.clear();
	for (int j = 0; j < 4; j++)
	{
		contour.push_back(rectangle_points[j]);
	}
	cover_ratio = (cv::contourArea(contour, false) / (image_size.width * image_size.height)) * 100;

	return cover_ratio;
}

// 初始化並篩選enroll image
int initFolderCacheSMathur(const std::string targetFolderName, const std::string verifyFolderName, fingerprintOptions* _options, bool ramdomFlag = true, int verifyValue = -1) {

	bool sameFolderFlag = targetFolderName.compare(verifyFolderName) == 0 ? true : false;
	cout << "sameFolderFlag: " << sameFolderFlag << endl;;

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
		int rejectCount = 0;
		std::vector<cv::Mat> enrollImageVector(0, cv::Mat());
		bool cover_reject = false;
		double cover_ratio = 0.0;
		double cover_threshold = 85.0;

		if (verifyValue < 0) {
			verifyValue = filesName.size();
		}

		for (int image_index = 0; image_index < filesName.size(); image_index++)
		{
			cv::Mat enrollImage = cv::imread(enrollFolderPath + "\\" + filesName[image_index], CV_LOAD_IMAGE_GRAYSCALE);

			cover_reject = false;

			if (enrollImageVector.size() == 16) {
				if (verifyFilesName.size() < verifyValue) {
					verifyFilesName.push_back(filesName[image_index]);
				}
				allVerifyFilesName.push_back(filesName[image_index]);
			}

			else if (enrollFilesName.size() == 0)
			{
				enrollFilesName.push_back(filesName[image_index]);
				enrollImageVector.push_back(enrollImage);
			}
			else {
				for (int enroll_image_index = 0; enroll_image_index < enrollFilesName.size(); enroll_image_index++) {

					cover_ratio = FMTOverlapRatio(enrollImage, enrollImageVector[enroll_image_index], 200, 100);


					if (cover_ratio >= cover_threshold)
					{
						cover_reject = true;
						break;
					}
				}

				if (cover_reject == false)
				{
					enrollFilesName.push_back(filesName[image_index]);
					enrollImageVector.push_back(enrollImage);
				}
				else {
					rejectCount++;
					if (verifyFilesName.size() < verifyValue) {
						verifyFilesName.push_back(filesName[image_index]);
					}
					allVerifyFilesName.push_back(filesName[image_index]);
				}
			}
		}

		// ====================================================================================================

		cout << subFolderName[i] << " => FMT Fault: " << rejectCount << endl;

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

void allFolderMatchingSMathur(const std::string cacheFolderPath, fingerprintOptions* _options) {

	std::vector<std::string> subFolderName = get_all_files_names_within_folder(cacheFolderPath);

	matcherSMathur * akazeMatch = new matcherSMathur(_options, false);

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
				std::string minScoreEnrollImageName, minScoreVerifyImageName;
				minScoreVerifyImageName = verifyList[verifyElement];
				std::vector<double> enrollScoreStore;

				//cout << "verifyList[" << verifyElement << "]: " << verifyList[verifyElement] << endl;

				for (int enrollElement = 0; enrollElement < enrollList.size(); enrollElement++) {
					cv::Mat enrollImage, verifyImage, resultImage;
					enrollImage = cv::imread(enrollList[enrollElement], CV_LOAD_IMAGE_GRAYSCALE);
					verifyImage = cv::imread(verifyList[verifyElement], CV_LOAD_IMAGE_GRAYSCALE);
					//cv::resize(enrollImage, enrollImage, cv::Size(enrollImage.cols * 2, enrollImage.rows * 2), 0, 0, cv::INTER_CUBIC);
					//cv::resize(verifyImage, verifyImage, cv::Size(enrollImage.cols * 2, enrollImage.rows * 2), 0, 0, cv::INTER_CUBIC);
					double currentScore = akazeMatch->compareSMathur(enrollImage, verifyImage, resultImage);
					//int currentScore = akazeMatch->compare_normal(enrollImage, verifyImage, resultImage);


					//cout << "currentScore: " << currentScore << endl;
					totalScore += currentScore;
					if (currentScore < minScore) {
						minScore = currentScore;
						minScoreEnrollImageName = enrollList[enrollElement];
					}
					enrollScoreStore.emplace_back(currentScore);
					//system("PAUSE");
				}
				//cout << "maxScore: " << maxScore << endl;

				//score save
				totalResultCSV << minScore << ", " << totalScore << ", " << subFolderName[enrollFolderIndex] << ", "
					<< minScoreEnrollImageName << ", " << subFolderName[verifyFolderIndex] << ", "
					<< minScoreVerifyImageName << endl;

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
論文名: Methodology for Partial Fingerprint Enrollmentand Authenticationon Mobile Devices
年分  : 2016
作者  : S. Mathur, A. Vjay, J. Shah, S. Das and A. Malla
Link  : https://ieeexplore.ieee.org/document/7550093
*/
void startSMathurTest(std::string targetFolderName, std::string outputFolderTitle) {

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

	initFolderCacheSMathur(targetFolderName, verifyFolderName, &options, false, 10);

	allFolderMatchingSMathur(cacheFolderPath, &options);
}