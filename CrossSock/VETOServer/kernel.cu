//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include "kernel.h"


__global__ void fd_classify(unsigned char* image, unsigned char* strictMask, unsigned char* looseMask, decisionMap* strictMaps, decisionMap* looseMaps, int width, int height)
{
	//get the yIndex and xIndex associated with this pixel
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//guard against pixels outside of the image
	if (xIndex < width && yIndex < height)
	{
		//get the image index associated with this pixel
		int i = xIndex + yIndex * width;

		//get the current pixel's values
		int r = image[i * 3 + 2] / (int)(256.0f / (float)NUM_BINS);
		int g = image[i * 3 + 1] / (int)(256.0f / (float)NUM_BINS);
		int b = image[i * 3] / (int)(256.0f / (float)NUM_BINS);
		int y = (int)((float)r * 0.3f + (float)g * 0.6f + (float)b * 0.1f);
		int cr = 0;
		int cb = 0;
		if (r + g + b != 0)
		{
			cr = r * NUM_BINS / (r + g + b);
			cb = b * NUM_BINS / (r + g + b);
		}

		//if the pixel is strictly in the foreground...
		if (strictMaps[i].decision[cr * 3] || strictMaps[i].decision[cb * 3 + 1] || strictMaps[i].decision[y * 3 + 2])
		{
			//set the mask color to white
			strictMask[i] = 255;
		}
		else //else if the pixel is in the background...
		{
			//set the mask color to black
			strictMask[i] = 0;
		}

		//if the pixel is loosely in the foreground...
		if (looseMaps[i].decision[cr * 3] || looseMaps[i].decision[cb * 3 + 1] || looseMaps[i].decision[y * 3 + 2])
		{
			//set the mask color to white
			looseMask[i] = 255;
		}
		else //else if the pixel is in the background...
		{
			//set the mask color to black
			looseMask[i] = 0;
		}
	}
}

__global__ void fd_update(unsigned char* image, energy* energies, int width, int height, int yOffset)
{
	//get the yIndex and xIndex associated with this pixel
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = (blockIdx.y + yOffset) * blockDim.y + threadIdx.y;

	//guard against pixels outside of the image
	if (xIndex < width && yIndex < height)
	{
		//get the image index associated with this pixel
		int i = xIndex + yIndex * width;

		//get the current pixel's values
		int r = image[i * 3 + 2] / (int)(256.0f / (float)NUM_BINS);
		int g = image[i * 3 + 1] / (int)(256.0f / (float)NUM_BINS);
		int b = image[i * 3] / (int)(256.0f / (float)NUM_BINS);
		int y = (int)((float)r * 0.3f + (float)g * 0.6f + (float)b * 0.1f);
		int cr = 0;
		int cb = 0;
		if (r + g + b != 0)
		{
			cr = r * NUM_BINS / (r + g + b);
			cb = b * NUM_BINS / (r + g + b);
		}

		//Increment this pixels counters
		energies[i].count[cr * 3]++;
		energies[i].count[cb * 3 + 1]++;
		energies[i].count[y * 3 + 2]++;
		energies[i].total++;
	}
}

__global__ void fd_train(energy* energies, histogram* histograms, decisionMap* strictMaps, decisionMap* looseMaps, int width, int height, int xOffset, int yOffset, float alpha, float strictThreshold, int strictRange, float looseThreshold, int looseRange)
{
	//get the yIndex and xIndex associated with this pixel
	int xIndex = (blockIdx.x + xOffset) * blockDim.x + threadIdx.x;
	int yIndex = (blockIdx.y + yOffset) * blockDim.y + threadIdx.y;

	//guard against pixels outside of the image
	if (xIndex < width && yIndex < height)
	{
		//get the image index associated with this pixel
		int i = xIndex + yIndex * width;

		//Get the sum of each pixel counter
		int cSum = energies[i].total;
		energies[i].total = 0;

		//Get a copy of the relevant histogram in local memory
		histogram h;
		for (int x = 0; x < NUM_BINS; x++)
		{
			h.bin[x * 3] = histograms[i].bin[x * 3];
			h.bin[x * 3 + 1] = histograms[i].bin[x * 3 + 1];
			h.bin[x * 3 + 2] = histograms[i].bin[x * 3 + 2];
		}

		//Set the new bin value based on alpha and the normalized count for this pixel, sum each new bin value, and reset the count for this pixel
		float crSum = 0.0f;
		float cbSum = 0.0f;
		float ySum = 0.0f;
		for (int x = 0; x < NUM_BINS; x++)
		{
			h.bin[x * 3] = ((float)energies[i].count[x * 3] / (float)cSum) * alpha + h.bin[x * 3] * (1.0f - alpha);
			crSum += h.bin[x * 3];
			energies[i].count[x * 3] = 0;

			h.bin[x * 3 + 1] = ((float)energies[i].count[x * 3 + 1] / (float)cSum) * alpha + h.bin[x * 3 + 1] * (1.0f - alpha);
			cbSum += h.bin[x * 3 + 1];
			energies[i].count[x * 3 + 1] = 0;

			h.bin[x * 3 + 2] = ((float)energies[i].count[x * 3 + 2] / (float)cSum) * alpha + h.bin[x * 3 + 2] * (1.0f - alpha);
			ySum += h.bin[x * 3 + 2];
			energies[i].count[x * 3 + 2] = 0;
		}

		//Divide each bin by bSum to normalize them, and then find the global max
		float crMax = 0.0f;
		float crMin = 1.0f;
		float cbMax = 0.0f;
		float cbMin = 1.0f;
		float yMax = 0.0f;
		float yMin = 1.0f;
		for (int x = 0; x < NUM_BINS; x++)
		{
			h.bin[x * 3] /= crSum;//scan[0]
			if (h.bin[x * 3] > crMax)
				crMax = h.bin[x * 3];
			if (h.bin[x * 3] < crMin)
				crMin = h.bin[x * 3];

			h.bin[x * 3 + 1] /= cbSum;//scan[1]
			if (h.bin[x * 3 + 1] > cbMax)
				cbMax = h.bin[x * 3 + 1];
			if (h.bin[x * 3 + 1] < cbMin)
				cbMin = h.bin[x * 3 + 1];

			h.bin[x * 3 + 2] /= ySum;//scan[2]
			if (h.bin[x * 3 + 2] > yMax)
				yMax = h.bin[x * 3 + 2];
			if (h.bin[x * 3 + 2] < yMin)
				yMin = h.bin[x * 3 + 2];
		}

		//Calculate the strict and loose thresholds for each classification criteria
		float crStrictThreshold = crMax * strictThreshold + crMin * (1.0f - strictThreshold);
		float cbStrictThreshold = cbMax * strictThreshold + cbMin * (1.0f - strictThreshold);
		float yStrictThreshold = yMax * strictThreshold + yMin * (1.0f - strictThreshold);
		float crLooseThreshold = crMax * looseThreshold + crMin * (1.0f - looseThreshold);
		float cbLooseThreshold = cbMax * looseThreshold + cbMin * (1.0f - looseThreshold);
		float yLooseThreshold = yMax * looseThreshold + yMin * (1.0f - looseThreshold);

		//Create the new decision maps on local memory
		decisionMap strictMap;
		decisionMap looseMap;

		//Perform the first strict decision map generation pass
		int crCount = 0;
		int cbCount = 0;
		int yCount = 0;
		for (int x = 0; x < NUM_BINS; x++)
		{
			if (h.bin[x * 3] > crStrictThreshold)
				crCount = strictRange;
			else
				crCount--;

			if (crCount > 0)
				strictMap.decision[x * 3] = false;
			else
				strictMap.decision[x * 3] = true;

			if (h.bin[x * 3 + 1] > cbStrictThreshold)
				cbCount = strictRange;
			else
				cbCount--;

			if (cbCount > 0)
				strictMap.decision[x * 3 + 1] = false;
			else
				strictMap.decision[x * 3 + 1] = true;

			if (h.bin[x * 3 + 2] > yStrictThreshold)
				yCount = strictRange;
			else
				yCount--;

			if (yCount > 0)
				strictMap.decision[x * 3 + 2] = false;
			else
				strictMap.decision[x * 3 + 2] = true;
		}

		//Perform the second strict decision map generation pass
		crCount = 0;
		cbCount = 0;
		yCount = 0;
		for (int x = NUM_BINS - 1; x >= 0; x--)
		{
			if (h.bin[x * 3] > crStrictThreshold)
				crCount = strictRange;
			else
				crCount--;

			if (crCount > 0)
				strictMap.decision[x * 3] = false;

			if (h.bin[x * 3 + 1] > cbStrictThreshold)
				cbCount = strictRange;
			else
				cbCount--;

			if (cbCount > 0)
				strictMap.decision[x * 3 + 1] = false;

			if (h.bin[x * 3 + 2] > yStrictThreshold)
				yCount = strictRange;
			else
				yCount--;

			if (yCount > 0)
				strictMap.decision[x * 3 + 2] = false;
		}

		//Perform the first loose decision map generation pass
		crCount = 0;
		cbCount = 0;
		yCount = 0;
		for (int x = 0; x < NUM_BINS; x++)
		{
			if (h.bin[x * 3] > crLooseThreshold)
				crCount = looseRange;
			else
				crCount--;

			if (crCount > 0)
				looseMap.decision[x * 3] = false;
			else
				looseMap.decision[x * 3] = true;

			if (h.bin[x * 3 + 1] > cbLooseThreshold)
				cbCount = looseRange;
			else
				cbCount--;

			if (cbCount > 0)
				looseMap.decision[x * 3 + 1] = false;
			else
				looseMap.decision[x * 3 + 1] = true;

			if (h.bin[x * 3 + 2] > yLooseThreshold)
				yCount = looseRange;
			else
				yCount--;

			if (yCount > 0)
				looseMap.decision[x * 3 + 2] = false;
			else
				looseMap.decision[x * 3 + 2] = true;
		}

		//Perform the second loose decision map generation pass
		crCount = 0;
		cbCount = 0;
		yCount = 0;
		for (int x = NUM_BINS - 1; x >= 0; x--)
		{
			if (h.bin[x * 3] > crLooseThreshold)
				crCount = looseRange;
			else
				crCount--;

			if (crCount > 0)
				looseMap.decision[x * 3] = false;

			if (h.bin[x * 3 + 1] > cbLooseThreshold)
				cbCount = looseRange;
			else
				cbCount--;

			if (cbCount > 0)
				looseMap.decision[x * 3 + 1] = false;

			if (h.bin[x * 3 + 2] > yLooseThreshold)
				yCount = looseRange;
			else
				yCount--;

			if (yCount > 0)
				looseMap.decision[x * 3 + 2] = false;
		}

		//Transfer the strict decision map to global memory
		for (int x = 0; x < NUM_BINS; x++)
		{
			strictMaps[i].decision[x * 3] = strictMap.decision[x * 3];
			strictMaps[i].decision[x * 3 + 1] = strictMap.decision[x * 3 + 1];
			strictMaps[i].decision[x * 3 + 2] = strictMap.decision[x * 3 + 2];
		}

		//Transfer the loose decision map to global memory
		for (int x = 0; x < NUM_BINS; x++)
		{
			looseMaps[i].decision[x * 3] = looseMap.decision[x * 3];
			looseMaps[i].decision[x * 3 + 1] = looseMap.decision[x * 3 + 1];
			looseMaps[i].decision[x * 3 + 2] = looseMap.decision[x * 3 + 2];
		}

		//Transfer the histogram to global memory
		for (int x = 0; x < NUM_BINS; x++)
		{
			histograms[i].bin[x * 3] = h.bin[x * 3];
			histograms[i].bin[x * 3 + 1] = h.bin[x * 3 + 1];
			histograms[i].bin[x * 3 + 2] = h.bin[x * 3 + 2];
		}
	}
}

__global__ void fd_cleanup(unsigned char* mask, int width, int height, int size)
{
	//get the yIndex and xIndex associated with this pixel
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//guard against pixels outside of the image
	if (xIndex < width && yIndex < height)
	{
		//get the image index associated with this pixel
		int i = xIndex + yIndex * width;

		//if this pixel is currently white...
		if (mask[i])
		{
			int whiteCount = 0;
			int blackCount = 0;
			for (int y = -1 * size; y <= size; y++)
			{
				for (int x = -1 * size; x <= size; x++)
				{
					//if this stencil pixel is valid
					if (xIndex + x < width && xIndex + x >= 0 && yIndex + y < height && yIndex + y >= 0)
					{
						//get the stencil pixel's index
						int j = (xIndex + x) + (yIndex + y) * width;

						//if the stencil pixel is white, increment white count
						if (mask[j])
							whiteCount++;
						else //Else increment black count
							blackCount++;
					}
				}
			}

			//syncronize the threads
			syncthreads();

			//move this pixel to background if too small of an object
			if (whiteCount < blackCount)
			{
				mask[i] = 0;
			}
		}
	}
}

__global__ void fd_fill(unsigned char* strictMask, unsigned char* looseMask, int width, int height, int size)
{
	//get the yIndex and xIndex associated with this pixel
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//guard against pixels outside of the image
	if (xIndex < width && yIndex < height)
	{
		//get the image index associated with this pixel
		int i = xIndex + yIndex * width;

		//if this pixel is strictly background but loosely foreground...
		if (looseMask[i] && !strictMask[i])
		{
			int whiteCount = 0;
			int blackCount = 0;
			for (int y = -1 * size; y <= size; y++)
			{
				for (int x = -1 * size; x <= size; x++)
				{
					//if this stencil pixel is valid
					if (xIndex + x < width && xIndex + x >= 0 && yIndex + y < height && yIndex + y >= 0)
					{
						//get the stencil pixel's index
						int j = (xIndex + x) + (yIndex + y) * width;

						//if the stencil pixel is white, increment white count
						if (strictMask[j])
							whiteCount++;
						else //Else increment black count
							blackCount++;
					}
				}
			}

			//syncronize the threads
			syncthreads();

			//If enough strict pixels where white, move this pixel to foreground
			if (whiteCount >= blackCount)
			{
				strictMask[i] = 255;
			}
		}
	}
}


extern "C" void cuda_fd_classify(unsigned char* image, unsigned char* strictMask, unsigned char* looseMask, decisionMap* strictMaps, decisionMap* looseMaps, int width, int height, dim3 gridSize, dim3 blockSize)
{
	fd_classify << < gridSize, blockSize >> > (image, strictMask, looseMask, strictMaps, looseMaps, width, height);
}

extern "C" void cuda_fd_train(energy* energies, histogram* histograms, decisionMap* strictMaps, decisionMap* looseMaps, int width, int height, int xOffset, int yOffset, float alpha, float strictThreshold, int strictRange, float looseThreshold, int looseRange, dim3 gridSize, dim3 blockSize)
{
	fd_train << < gridSize, blockSize >> > (energies, histograms, strictMaps, looseMaps, width, height, xOffset, yOffset, alpha, strictThreshold, strictRange, looseThreshold, looseRange);
}

extern "C" void cuda_fd_update(unsigned char* image, energy* energies, int width, int height, int yOffset, dim3 gridSize, dim3 blockSize)
{
	fd_update << < gridSize, blockSize >> > (image, energies, width, height, yOffset);
}

extern "C" void cuda_fd_cleanup(unsigned char* mask, int width, int height, int size, dim3 gridSize, dim3 blockSize)
{
	fd_cleanup << < gridSize, blockSize >> > (mask, width, height, size);
}

extern "C" void cuda_fd_fill(unsigned char* strictMask, unsigned char* looseMask, int width, int height, int size, dim3 gridSize, dim3 blockSize)
{
	fd_fill << < gridSize, blockSize >> > (strictMask, looseMask, width, height, size);
}
