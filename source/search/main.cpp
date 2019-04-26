#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define MAX_NUM_PER_CLASS 2000
#define PCA_NAMES_FILE    "pca.names"
#define PCA_FC_FILE       "pca_fc6.bin"
#define PCA_POOL_FILE     "pca_pool5.bin"
#define HIST_FILE         ".hist"
#define HIST_CHANNELS     10
#define TOP_NUM_GLOBAL    200
#define TOP_NUM_LOCAL     5
#define INVALID_SCORE     -100

struct RelatedScore
{
	int id;
	float score;
	RelatedScore()
	{
		id = -1;
		score = -1.f;
	}

	RelatedScore(int i, float v)
	{
		id = i;
		score = v;
	}
};

bool cmpScore(const RelatedScore& n1, const RelatedScore& n2) { if (n1.score == n2.score) return n1.id < n2.id;  return n1.score < n2.score; }

bool readNamesPca(vector<string>& fileList, const char* fileName)
{
	FILE *fp = fopen(fileName, "r");
	if (fp == NULL) return false;

	char str[32];
	int id = 0;
	for (int i = 0; i < MAX_NUM_PER_CLASS; ++i)
	{
		int val = fscanf(fp, "%s\n", str);
		if (val == EOF) break;

		fileList[id++] = str;
	}
	fileList.resize(id - 1);

	fclose(fp);
	return true;
}

bool getPcaData(const char* ftrDir, const char* className, int fNum, int imgLen,
	float*& fpools, int& channelPool, float*& fcs, int& channelFc, float*& hists, int& channelHist)
{
	string inDir(ftrDir);
	inDir = inDir + string(className) + "\\";

	string nameFile = inDir + PCA_NAMES_FILE;
	string fcFile   = inDir + PCA_FC_FILE;
	string poolFile = inDir + PCA_POOL_FILE;
	string histFile = inDir + string(className) + HIST_FILE;

	int imgNum = imgLen * imgLen;

	FILE *fp = fopen(poolFile.c_str(), "rb");
	if (fp == NULL) return false;
	fread(&channelPool, sizeof(int), 1, fp);
	int offset = channelPool * imgNum;
	fpools = (float*)malloc(offset * fNum * sizeof(float));
	fread(fpools, sizeof(float), offset * fNum, fp);
	fclose(fp);

	fp = fopen(fcFile.c_str(), "rb");
	if (fp == NULL) return false;
	fread(&channelFc, sizeof(int), 1, fp);
	fcs = (float*)malloc(channelFc * fNum * sizeof(float));
	fread(fcs, sizeof(float), channelFc * fNum, fp);
	fclose(fp);

	fp = fopen(histFile.c_str(), "rb");
	if (fp == NULL) return false;
	channelHist = HIST_CHANNELS;
	hists = (float*)malloc(channelHist * imgNum * fNum * sizeof(float));
	fread(hists, sizeof(float), channelHist * imgNum * fNum, fp);
	fclose(fp);

	return true;
}

void getHistograms(float*& histData, const Mat& smlImg, int imgLen)
{
	int minCols = smlImg.cols;
	int minRows = smlImg.rows;

	int l_bins = HIST_CHANNELS;
	float l_ranges[] = { 0, 256 };
	int histSize[] = { l_bins };
	const float* ranges[] = { l_ranges };
	int channels[] = { 0 };

	int dx = int(minCols / (float)imgLen + 0.5f);
	int dy = int(minRows / (float)imgLen + 0.5f);
	int offx = 16;
	int offy = 16;

	histData = (float*)malloc(imgLen * imgLen * l_bins * sizeof(float));
	for (int iy = 0; iy < imgLen; ++iy)
	{
		int sy = max(iy * dy, offy);
		int h = min(minRows - sy - offy, (iy + 1) * dy - sy);
		for (int ix = 0; ix < imgLen; ++ix)
		{
			int sx = max(ix * dx, offx);
			int w = min(minCols - sx - offx, (ix + 1) * dx - sx);
			int id = iy * imgLen + ix;
			Mat region = smlImg(Rect(sx, sy, w, h));

			MatND hist;
			calcHist(&region, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
			memcpy(&histData[id * l_bins], hist.data, hist.cols * hist.rows * sizeof(float));
		}
	}
}

float distByHistogram(const vector<unsigned int>& idMap, float* hists0, float* hists1, int imgLen)
{
	int histNum = imgLen * imgLen;
	float dist = 0.f;

	for (int h = 0; h < histNum; ++h)
	{
		int h1 = h;
		if (idMap.size())
		{
			h1 = idMap[h1];
		}

		cv::MatND hist0(HIST_CHANNELS, 1, CV_32F, &(hists0[h * HIST_CHANNELS]));
		cv::MatND hist1(HIST_CHANNELS, 1, CV_32F, &(hists1[h1 * HIST_CHANNELS]));

		double histDist = compareHist(hist0, hist1, CV_COMP_CORREL);
		dist += 1.f - float(histDist);
	}

	return dist;
}


void getTops(int* rrids, float* rscrs, const float* srcfpools, const float* srcfcs, float* srchists, int fNum, int imgLen,
	float* fpools, int& channelPool, float* fcs, int& channelFc, float* hists, int& channelHist,
	float fWeight, float hWeight)
{
	vector<RelatedScore> distFc(fNum);
	vector<int> ids;

	int imgNum = imgLen * imgLen;
	int offset = channelPool * imgNum;

#pragma omp parallel for
	for (int f = 0; f < fNum; ++f)
	{
		int sid1 = f * channelFc;
		distFc[f].id = f;
		distFc[f].score = 0.f;
		for (int c = 0; c < channelFc; ++c)
		{
			distFc[f].score -= srcfcs[c] * fcs[c + sid1];
		}
	}
	sort(distFc.begin(), distFc.end(), cmpScore);

	int topFc = min(TOP_NUM_GLOBAL, fNum / 2);
	for (int i = 0; i < topFc; ++i)
	{
		ids.push_back(distFc[i].id);
	}

	vector<RelatedScore> distPool(topFc);
	vector<vector<unsigned int>> idMap(fNum);

#pragma omp parallel for
	for (int f = 0; f < topFc; ++f)
	{
		distPool[f].id = distFc[f].id;

		int sid1 = distFc[f].id * offset;
		idMap[distFc[f].id].resize(imgNum);

		float sum = 0;
		for (int i0 = 0; i0 < imgNum; ++i0)
		{
			float distMin = 0.f;
			for (int i1 = 0; i1 < imgNum; ++i1)
			{
				float dist = 0.f;
				for (int c = 0; c < channelPool; ++c)
				{
					dist -= srcfpools[c * imgNum + i0] * fpools[sid1 + c * imgNum + i1];
				}
				if (distMin > dist)
				{
					distMin = dist;
					idMap[distFc[f].id][i0] = i1;
				}
			}

			sum += distMin;
		}

		float dist0 = sum / imgNum + 1.f;

		float dist1 = (distByHistogram(idMap[distFc[f].id], srchists, &hists[distFc[f].id * channelHist * imgNum], imgLen) / imgNum + 1.f) * 0.5;

		distPool[f].score = dist0 * fWeight + dist1 * hWeight;
	}

	sort(distPool.begin(), distPool.end(), cmpScore);

	for (int i = 0; i < TOP_NUM_LOCAL; ++i)
	{
		rrids[i] = distPool[i].id;
		rscrs[i] = distPool[i].score;
	}
}

__declspec(dllexport) void __stdcall search(const float* pcaFc6, const float* pcaPool5, 
	const char* srcFtrDir, const char* inDir, const char* outDir, const char* name0, const char* name1,
	int minLen, int ftrLen, float fWeight, float hWeight)
{
	printf("Search strating.\n"); 
	
	string ftrDir(srcFtrDir);
	string inImgDir(inDir);
	string outFileDir(outDir);
	string strName0(name0);
	string strName1(name1);
	
	string outFile = outFileDir + "\\" + strName0 + ".pairs";
	FILE* outfp = fopen(outFile.c_str(), "w");

	string fileName = ftrDir + strName1 + "\\" + PCA_NAMES_FILE;
	vector<string> fileList(MAX_NUM_PER_CLASS);
	bool res = readNamesPca(fileList, fileName.c_str());
	if (!res)
	{
		printf("Fatal Error 1: %s\n", name1);
		return;
	}

	int fNum = int(fileList.size());
	float* fcs = NULL;
	float* fpools = NULL;
	float* hists = NULL;
	float* hist = NULL;

	int channelFc = 0;
	int channelPool = 0;
	int channelHist = 0;

	res = getPcaData(ftrDir.c_str(), name1, fNum, ftrLen, fpools, channelPool, fcs, channelFc, hists, channelHist);
	if (!res)
	{
		if (hist) free(hist);
		if (fpools) free(fpools);
		if (fcs) free(fcs);
		if (hists) free(hists);
		return;
	}

	fileName = inImgDir + strName0;
	Mat img = imread(fileName.c_str());

	float scale = float(minLen) / float(min(img.cols, img.rows));
	int min_width = int(img.cols * scale + 0.5f);
	int min_height = int(img.rows * scale + 0.5f);
	resize(img, img, cv::Size(min_width, min_height));

	// convert to Lab
	Mat labCol;
	cvtColor(img, labCol, COLOR_BGR2Lab);

	// compute color histograms
	getHistograms(hist, labCol, ftrLen);

	// search top N based on global and local similarity
	int tops[TOP_NUM_LOCAL];
	float scores[TOP_NUM_LOCAL];
	getTops(tops, scores, pcaPool5, pcaFc6, hist, fNum, ftrLen, fpools, channelPool, fcs, channelFc, hists, channelHist, fWeight, hWeight);

	if (scores[0] < INVALID_SCORE)
	{
		printf("Fatal Error 4: %s %s\n", name0, name1);
		return;
	}

	for (int ti = 0; ti < TOP_NUM_LOCAL; ++ti)
	{
		fprintf(outfp, "%s %s %2.4f\n", name0, fileList[tops[ti]].c_str(), 1.f - scores[ti]);
		printf("Search Result: %s %s %2.4f\n", name0, fileList[tops[ti]].c_str(), 1.f - scores[ti]);
	}
	fprintf(outfp, "\n");

	if (hist) free(hist);
	if (fpools) free(fpools);
	if (fcs) free(fcs);
	if (hists) free(hists);

	fclose(outfp);
	printf("Search done.\n");
}
