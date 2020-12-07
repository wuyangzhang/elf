#include <iostream>
#include <stdio.h>
#include <cstdio>

#include <chrono>
#include <cuda_runtime_api.h>
#include "nvjpeg.h"
#include <string.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;


extern "C" void rgbConvert(char*, int, unsigned char *);
extern "C" void rgbConvertBack(int, unsigned char *);

class NvJPEG{
private:
    struct encode_params_t {
          std::string input_dir;
          std::string output_dir;
          std::string format;
          std::string subsampling;
          int quality;
          int dev;
    };

    static int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
    static int dev_free(void *p) { return (int)cudaFree(p); }

    nvjpegEncoderParams_t encode_params;
    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t jpeg_state;
    nvjpegEncoderState_t encoder_state;
    int max_width, max_height;
    unsigned char * pBuffer = NULL;
    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};


public:
    NvJPEG(int max_width1, int max_height1){
            max_width = max_width1;
            max_height = max_height1;
        }

    ~NvJPEG(){
        cudaFree(pBuffer);
        nvjpegEncoderParamsDestroy(encode_params);
        nvjpegEncoderStateDestroy(encoder_state);
        nvjpegJpegStateDestroy(jpeg_state);
        nvjpegDestroy(nvjpeg_handle);
    }

    void init(){
        cudaMalloc((void **)&pBuffer, 7 * max_width * max_height);
        nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &nvjpeg_handle);
        nvjpegJpegStateCreate(nvjpeg_handle, &jpeg_state);
        nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, NULL);
        nvjpegEncoderParamsCreate(nvjpeg_handle, &encode_params, NULL);
        nvjpegEncoderParamsSetQuality(encode_params, 70, NULL);
        nvjpegEncoderParamsSetOptimizedHuffman(encode_params, 1, NULL);
        nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_420, NULL);
    }

    py::bytes encode(py::array_t<uint8_t> input) {

            py::buffer_info buf = input.request();
            int width = buf.shape[1];
            int height = buf.shape[0];
            int len = buf.size / 3;

            // option 1. cpu version RGB format transmission.
//            char * nv_buf = (char *) malloc(width * height * 4);
//            char * oR = nv_buf, *oG = oR + len, *oB = oG + len, *oA = oB + len;
//            const char *iPos = (const char *)buf.ptr;
//
//            for (int i = 0; i < len; i++) {
//                *(oB++) = *(iPos++);
//                *(oG++) = *(iPos++);
//                *(oR++) = *(iPos++);
//                *(oA++) = 0;
//            }
//
//            cudaMemcpy(pBuffer, nv_buf, NVJPEG_MAX_COMPONENT * width * height, cudaMemcpyHostToDevice);
//            cudaDeviceSynchronize();
//            nvjpegImage_t imgdesc = {
//                {
//                    (uint8_t *) pBuffer + width * height * 0,
//                    (uint8_t *) pBuffer + width * height * 1,
//                    (uint8_t *) pBuffer + width * height * 2,
//                    (uint8_t *) pBuffer + width * height * 3
//                },
//                {
//                    (unsigned int)width,
//                    (unsigned int)width,
//                    (unsigned int)width,
//                    (unsigned int)width
//                }
//            };

            //option 2: use cuda to accelerate rgb convert.
            rgbConvert((char*)buf.ptr, buf.size, pBuffer);

            nvjpegImage_t imgdesc = {
                {
                    (uint8_t *) pBuffer + width * height * 3,
                    (uint8_t *) pBuffer + width * height * 4,
                    (uint8_t *) pBuffer + width * height * 5,
                    (uint8_t *) pBuffer + width * height * 6
                },
                {
                    (unsigned int)width,
                    (unsigned int)width,
                    (unsigned int)width,
                    (unsigned int)width
                }
            };


            nvjpegEncodeImage(nvjpeg_handle,
                            encoder_state,
                            encode_params,
                            &imgdesc,
                            NVJPEG_INPUT_RGB,
                            width,
                            height,
                            NULL);

            std::vector<unsigned char> obuffer;
            size_t length;
            nvjpegEncodeRetrieveBitstream(
                        nvjpeg_handle,
                        encoder_state,
                        NULL,
                        &length,
                        NULL);

            obuffer.resize(length);
            nvjpegEncodeRetrieveBitstream(
                      nvjpeg_handle,
                      encoder_state,
                      obuffer.data(),
                      &length,
                      NULL);

            std::string str(obuffer.data(), obuffer.data() + obuffer.size() / sizeof obuffer[0]);
            return py::bytes(str);
        }


    py::array_t<uint8_t> decode(std::string input) {
        size_t size = input.length();
        unsigned char * dpImage = (unsigned char *)input.c_str();
        int nComponent = 0;
        nvjpegChromaSubsampling_t subsampling;
        int widths[NVJPEG_MAX_COMPONENT];
        int heights[NVJPEG_MAX_COMPONENT];
        if (NVJPEG_STATUS_SUCCESS != nvjpegGetImageInfo(nvjpeg_handle, dpImage, size, &nComponent, &subsampling, widths, heights)){
           std::cerr << "fail to get image info!" << std::endl;
        }
        nvjpegImage_t imgdesc = {
             {
                 pBuffer,
                 pBuffer + widths[0]*heights[0],
                 pBuffer + widths[0]*heights[0]*2,
                 pBuffer + widths[0]*heights[0]*3
             },
             {
                 (unsigned int)widths[0],
                 (unsigned int)widths[0],
                 (unsigned int)widths[0],
                 (unsigned int)widths[0]
             }
         };
        cudaDeviceSynchronize();

        int nReturnCode = nvjpegDecode(nvjpeg_handle, jpeg_state, dpImage, size, NVJPEG_OUTPUT_RGB, &imgdesc, NULL);
        cudaDeviceSynchronize();

        // results are in pBuffer.

        int len = widths[0] * heights[0];
        py::array_t<uint8_t> output = py::array_t<uint8_t>(len * 3);
        auto buf = output.request();

        rgbConvertBack(len, pBuffer);
        cudaMemcpy(buf.ptr, pBuffer + widths[0] * heights[0] * 4, widths[0] * heights[0] * 3, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // option 1. cpu based format transfer
//        char * res = (char *) malloc(widths[0] * heights[0] * NVJPEG_MAX_COMPONENT);

//        const char *iR = (const char *)res, *iG = iR + len, *iB = iG + len;
//        char *oPos = (char *)buf.ptr;
//        for (int i = 0; i < len; i++){
//            *(oPos++) = *(iB++);
//            *(oPos++) = *(iG++);
//            *(oPos++) = *(iR++);
//        }
//
//        free(res);
        output.resize({heights[0], widths[0], 3});
        return output;
    }
};


PYBIND11_MODULE(py_nvjpeg, m) {
    py::class_<NvJPEG>(m, "NvJPEG")
        .def(py::init<const int, const int>())
        .def("init", &NvJPEG::init)
        .def("encode", &NvJPEG::encode)
        .def("decode", &NvJPEG::decode);
}

//int main(){
//    NvJPEG nvj = NvJPEG(1000, 2600);
//    nvj.init();
//
//    int len = 370 * 1224;
//    py::array_t<uint8_t> input = py::array_t<uint8_t>(len * 3);
//    py::buffer_info buf = input.request();
//    FILE* fp = fopen("img.data", "w+");
//    fread(buf.ptr, len*3, 1, fp);
//    fclose(fp);
//
//    nvj.encode(input);
//    nvj.encode(input);
//}
