#include <stdio.h>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace std;

__global__
void convert(char *input, char *output, int len, int step)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//if (index < size) *(output + (2 - index % 3) * len + index / 3) = *(input+index);

    char * oR = output + index * step, *oG = oR + len, *oB = oG + len;
    const char *iPos = (const char *)input + index * step * 3;

    for(int i = 0; i < step; i += 1){
        *(oB++) = *(iPos++);
        *(oG++) = *(iPos++);
        *(oR++) = *(iPos++);
    }
}


extern "C" void rgbConvert(char* input, int size, unsigned char * pBuffer){
    int len = size / 3;
    char * cinput = (char *) pBuffer;
    char * coutput = (char *) pBuffer + size;
    int block = 128;
    int thread = 128;

    cudaMemcpy(pBuffer, input, size, cudaMemcpyHostToDevice );
    cudaMemset(pBuffer + len * 6, 0, len);

    cudaDeviceSynchronize();

    if(len/block/thread > 0){
        convert<<<len/block/thread, thread>>>(cinput, coutput, len, block);
        convert<<<len%(block*thread), 1>>>(cinput + len/block/thread*block*thread*3, coutput + len/block/thread*block*thread, len, 1);
    }else{
        convert<<<len%(block*thread), 1>>>(cinput , coutput , len, 1);
    }


    cudaDeviceSynchronize();

}


__global__
void convertBack(char *input, char *output, int len, int step)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//if (index < size) *(output + (2 - index % 3) * len + index / 3) = *(input+index);

    char * iR = input + index * step , *iG = iR + len, *iB = iG + len;
    char *oPos = output + index * step * 3;

    for (int i = 0; i < step; i++){
        *(oPos++) = *(iB++);
        *(oPos++) = *(iG++);
        *(oPos++) = *(iR++);
    }
}


extern "C" void rgbConvertBack(int len, unsigned char * pBuffer){

    int block = 128;
    int thread = 128;

    char * cinput = (char *) pBuffer;
    char * coutput = (char *) pBuffer + len * 4;

    if(len/block/thread > 0){
        convertBack<<<len/block/thread, thread>>>(cinput, coutput, len, thread);
        convertBack<<<len%(block*thread), 1>>>(cinput + len/block/thread*block*thread, coutput + len/block/thread*block*thread*3, len, 1);
    }else{
        convertBack<<<len%(block*thread), 1>>>(cinput, coutput, len, 1);
    }
    cudaDeviceSynchronize();
}


__global__
void convert128(char *input, char *output, int len, int step)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//if (index < size) *(output + (2 - index % 3) * len + index / 3) = *(input+index);

    char * oR = output + index * step, *oG = oR + len, *oB = oG + len;
    const char *iPos = (const char *)input + index * step * 3;

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

        *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

        *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

        *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

        *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

        *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);

    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
    *(oB++) = *(iPos++);
    *(oG++) = *(iPos++);
    *(oR++) = *(iPos++);
}

/*
int main(){

    int size = 1811520*3;
	char * input = (char*) malloc(size);
	memset(input, 2, size);

    char* output = rgbConvert(input, size);
    cudaFree(output);
	return EXIT_SUCCESS;

}
*/