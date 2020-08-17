#ifndef SURBHI_MATHUR_IMPLEMENT_2016
#define SURBHI_MATHUR_IMPLEMENT_2016

#include <iostream>
#include <opencv2/opencv.hpp>

#include "fingerprint_config.h"

/*
論文名: Methodology for Partial Fingerprint Enrollmentand Authenticationon Mobile Devices
年分  : 2016
作者  : S. Mathur, A. Vjay, J. Shah, S. Das and A. Malla
Link  : https://ieeexplore.ieee.org/document/7550093
*/


class matcherSMathur {
public:
	matcherSMathur(fingerprintOptions* options, bool detectMode);
	cv::Mat clacMaskSMathur(cv::Mat inputImage, cv::Size blockSize);
	double compareSMathur(cv::Mat tempImage, cv::Mat validImage, cv::Mat& resultImage);

private:
	bool  _detectMode = false;
	fingerprintOptions* _options;

	cv::Ptr<cv::AKAZE> _feature_detector;
	cv::Ptr<cv::AKAZE> _feature_extractor;

	std::vector<std::vector<cv::DMatch>> _akazeKnnMatchSMathur(cv::Mat desc_find, cv::Mat desc_data, int k);

};

bool responseSortSMathur(cv::KeyPoint a, cv::KeyPoint b);
bool matchSortSMathur(std::vector< cv::DMatch > a, std::vector< cv::DMatch > b);
double calcEuclideanDistanceSMathur(cv::Point2f a, cv::Point2f b);
double FMTOverlapRatio(cv::Mat im0, cv::Mat im1, double canny_threshold1, double canny_threshold2);

int initFolderCacheSMathur(const std::string targetFolderName, const std::string verifyFolderName, fingerprintOptions* _options, bool ramdomFlag, int verifyValue);
void allFolderMatchingSMathur(const std::string cacheFolderPath, fingerprintOptions* _options);

/*
論文名: Methodology for Partial Fingerprint Enrollmentand Authenticationon Mobile Devices
年分  : 2016
作者  : S. Mathur, A. Vjay, J. Shah, S. Das and A. Malla
Link  : https://ieeexplore.ieee.org/document/7550093
*/
void startSMathurTest(std::string targetFolderName, std::string outputFolderTitle);

#endif