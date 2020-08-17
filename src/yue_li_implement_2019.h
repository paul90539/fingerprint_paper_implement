#ifndef YUE_LI_IMPLEMENT_2019_H
#define YUE_LI_IMPLEMENT_2019_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "fingerprint_config.h"

/* ================================================================
論文名: ORB-based Fingerprint Matching Algorithm for Mobile Devices
年分  : 2019
作者  : Y. Li, G. Shi
Link  : https://ieeexplore.ieee.org/document/8989155
================================================================ */

class matcherYueLi {
public:
	matcherYueLi(fingerprintOptions* options, bool detectMode);
	double compareYueLi(cv::Mat tempImage, cv::Mat validImage, cv::Mat& resultImage);

private:
	bool  _detectMode = false;
	fingerprintOptions* _options;

	cv::Ptr<cv::ORB> _feature_detector;
	cv::Ptr<cv::ORB> _feature_extractor;

	std::vector<std::vector<cv::DMatch>> _orbKnnMatch(cv::Mat desc_find, cv::Mat desc_data, int k);

};

bool matchSortYueLi(std::vector< cv::DMatch > a, std::vector< cv::DMatch > b);
bool errorScoreSortYueLi(double a, double b);
double calcEuclideanDistanceYueLi(cv::Point2f a, cv::Point2f b);

int initFolderCacheYueLi(const std::string targetFolderName, const std::string verifyFolderName, cv::Mat* background, cv::Mat* mask, fingerprintOptions* _options, bool ramdomFlag, int verifyValue);
void allFolderMatchingYueLi(const std::string cacheFolderPath, cv::Mat* background, cv::Mat* mask, fingerprintOptions* _options);

/* ================================================================
論文名: ORB-based Fingerprint Matching Algorithm for Mobile Devices
年分  : 2019
作者  : Y. Li, G. Shi
Link  : https://ieeexplore.ieee.org/document/8989155
================================================================ */
void startYueLiTest(std::string targetFolderName, std::string outputFolderTitle);

#endif