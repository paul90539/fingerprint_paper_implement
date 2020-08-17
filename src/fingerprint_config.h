#ifndef FINGERPRINT_CONFIG_H
#define FINGERPRINT_CONFIG_H

struct fingerprintOptions {

	fingerprintOptions()
		: imageMatchingDisplay(false)
		, outputFolderPath("")
	{
	}
	bool imageMatchingDisplay;     // 是否顯示影像匹配的過程圖
	std::string outputFolderPath;  // 輸出檔案位置
};

#endif