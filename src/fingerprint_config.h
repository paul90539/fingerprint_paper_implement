#ifndef FINGERPRINT_CONFIG_H
#define FINGERPRINT_CONFIG_H

struct fingerprintOptions {

	fingerprintOptions()
		: imageMatchingDisplay(false)
		, outputFolderPath("")
	{
	}
	bool imageMatchingDisplay;     // �O�_��ܼv���ǰt���L�{��
	std::string outputFolderPath;  // ��X�ɮצ�m
};

#endif