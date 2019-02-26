#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_BINS 64

typedef struct
{
	float bin[NUM_BINS * 3];
} histogram;

typedef struct
{
	int count[NUM_BINS * 3];
	int total;
} energy;

typedef struct
{
	bool decision[NUM_BINS * 3];
} decisionMap;

extern "C" void cuda_fd_update(unsigned char* image, energy* energies, int width, int height, int yOffset, dim3 gridSize, dim3 blockSize);

extern "C" void cuda_fd_classify(unsigned char* image, unsigned char* strictMask, unsigned char* looseMask, decisionMap* strictMaps, decisionMap* looseMaps, int width, int height, dim3 gridSize, dim3 blockSize);

extern "C" void cuda_fd_train(energy* energies, histogram* histograms, decisionMap* strictMaps, decisionMap* looseMaps, int width, int height, int xOffset, int yOffset, float alpha, float strictThreshold, int strictRange, float looseThreshold, int looseRange, dim3 gridSize, dim3 blockSize);

extern "C" void cuda_fd_cleanup(unsigned char* mask, int width, int height, int size, dim3 gridSize, dim3 blockSize);

extern "C" void cuda_fd_fill(unsigned char* strictMask, unsigned char* looseMask, int width, int height, int size, dim3 gridSize, dim3 blockSize);