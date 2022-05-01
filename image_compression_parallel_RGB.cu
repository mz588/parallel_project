// Please note that this code takes in a command line argumentL THRESHOLD;
// nvcc image_compression_parallel_RGB.cu -o parallel_RGB -lm -lcufft -w
// ./parallel_RGB 0.1

#include<stdint.h>
#include<stdlib.h>
#include<stdio.h>
#include<cufft.h>
#include<math.h>
#include<thrust/device_vector.h>
#include<thrust/copy.h>
#include<thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/complex.h>
#include <thrust/extrema.h>

// Necessary libs for reading in and writing to image files
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// #define STB_IMAGE_WRITE_IMPLEMENTATION
#define INPUTFILE "./input_images/image.png"
#define OUTPUTFILE_JPG "./result_images/result_RGB.jpg"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__host__ __device__ bool operator> (const cufftDoubleComplex& a, const cufftDoubleComplex& b){
  return sqrt(a.x * a.x + a.y * a.y) > sqrt(b.x * b.x + b.y * b.y);
}

__host__ __device__ bool operator< (const cufftDoubleComplex& a, const cufftDoubleComplex& b){
  return sqrt(a.x * a.x + a.y * a.y) < sqrt(b.x * b.x + b.y * b.y);
}


void printMatrix(double* m, int row, int col){
  for(int i = 0; i < row; i++){
    printf("[");
    for(int j = 0; j < col; j++){
      printf("%E", m[i*row+j]);
      if(j != col-1) printf(", ");
    }
    printf("]\n");
  }
  printf("\n");
}

void printImageData(unsigned char *m, int row, int col, int channels){
  for(int i = 0; i < row; i++){
    printf("[");
    for(int j = 0; j < col; j++){
      printf("(");
      for(int k = 0; k < channels; k++){
        printf("%d", (int)m[i*col*channels+j*channels+k]);
        if(k != channels - 1) printf(", ");
      }
      printf(")");
      if(j != col - 1) printf(", ");
    }
    printf("]\n");
  }
  printf("\n");
}

void printMatrixChars(unsigned char*m, int row, int col){
  for(int i = 0; i < row; i++){
    printf("[");
    for(int j = 0; j < col; j++){
      printf("%d", (int)m[IDX2C(i,j,row)]);
      if(j != col - 1) printf(", ");
    }
    printf("]\n");
  }
  printf("\n");
}

void printComplexMatrix(cufftDoubleComplex *m, int row, int col){
  for(int i = 0; i < row; i++) {
    printf("[");
    for(int j = 0; j < col; j++){
      printf("(%E, %E)", m[IDX2C(i,j,row)].x, m[IDX2C(i,j,row)].y);
      if(j != col - 1) printf(", ");
    }
    printf("]\n");
  }
  printf("\n");
}

__global__ void generateResultImage(cufftDoubleComplex *IFFT, cufftDoubleComplex max, unsigned char*result, int row, int col) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(row_index < row && col_index < col) {
    int result_index = row_index * col + col_index;
    int IFFT_index = IDX2C(row_index, col_index, row);
    result[result_index] = (unsigned char)(IFFT[IFFT_index].x/max.x*255);
  }
}

__global__ void generateResultImageRGB(cufftDoubleComplex *IFFT, cufftDoubleComplex max, unsigned char*result, int row, int col, int number_of_channles, int channel){
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(row_index < row && col_index < col) {
    int result_index = (row_index * col + col_index ) *number_of_channles;
    int IFFT_index = IDX2C(row_index, col_index, row);
    result[result_index+channel] = (unsigned char)(IFFT[IFFT_index].x/max.x*255);
  }
}

__global__ void toGrayScaleImage(unsigned char*src, unsigned char*dest, int row, int col, int origin_channels, int new_channels){
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(row_index < row && col_index < col) {
    int src_starting = (row_index*col + col_index) * origin_channels;
    int dest_starting = (row_index*col + col_index) * new_channels;
    unsigned char red, green, blue;
    red = src[src_starting];
    green = src[src_starting+1];
    blue = src[src_starting+2];
    dest[dest_starting] = red*0.3+blue*0.11+green*0.59; 
    if(origin_channels == 4) dest[dest_starting+1] = src[src_starting+3];
  }
}

__global__ void toGrayScaleData(unsigned char*src, double*dest, int row, int col, int origin_channels){
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(row_index < row && col_index < col) {
    int src_starting = (row_index*col + col_index) * origin_channels;
    int dest_starting = row_index*col + col_index;
    double red, green, blue;
    red = (double)src[src_starting];
    green = (double)src[src_starting+1];
    blue = (double)src[src_starting+2]; 
    dest[dest_starting] = (red*0.3+blue*0.11+green*0.59)/255; 
  }
}

__global__ void getChannelData(unsigned char *src, cufftDoubleComplex *dest, int row, int col, int origin_channels, int target_channel){
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(row_index < row && col_index < col) {
    assert(target_channel < origin_channels);
    int src_starting = (row_index*col + col_index) * origin_channels;
    int dest_starting = IDX2C(row_index, col_index, row);
    dest[dest_starting].x = (double)src[src_starting+target_channel]; 
  }
}

__global__ void copyToComplex(double*src, cufftDoubleComplex*dest, int row, int col){
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(row_index < row && col_index < col) {
    int index = row_index * col + col_index;
    int dest_index = IDX2C(row_index, col_index, row);
    dest[dest_index].x = src[index];
  }
}

__global__ void cutoff(cufftDoubleComplex*src, cufftDoubleComplex*sorted, int thresholdIdx, int row, int col){
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  if(row_index < row && col_index < col) {
    int index = row_index*col + col_index;
    double threshold_val = sqrt(sorted[thresholdIdx].x * sorted[thresholdIdx].x + sorted[thresholdIdx].y * sorted[thresholdIdx].y);
    double cur_val = sqrt(src[index].x * src[index].x + src[index].y * src[index].x);
    if(cur_val < threshold_val) src[index].x = src[index].y = 0.0;
  }
}

void generateArray(cufftDoubleComplex*dest, int row, int col){
  int index = 0;
  for(int i = 0; i < row; i++){
    for(int j = 0; j < col; j++) dest[IDX2C(i,j,row)].x = index++;
  }
}

int main(int argc, char* argv[]) {

  assert(argc == 2);
  double THRESHOLD = atof(argv[1]);

  // Pointer to the memory of image on device
  cufftDoubleComplex *fft_result, *Ifft_result, *red_dev, *green_dev, *blue_dev;
  unsigned char *rgb_image_dev, *final_result_RGB_dev, *final_result_RGB;
  float elapsed_time;
  // declare cufft handle, use in each cufft call
  cufftHandle planZ2Z, planIZ2Z;
  // for checking if cufft fails
  cufftResult cuError;
  // for checking memory allocation on device fails
  cudaError_t cudaStat = cudaSuccess;
  // for measuring the time
  cudaEvent_t start,stop;

  int width, height, origin_channels;

  // read in the image file
  unsigned char *rgb_image_chars = stbi_load(INPUTFILE, &width, &height, &origin_channels, 0);
  assert(rgb_image_chars != NULL);
  printf("width: %d, height: %d, origin_channels: %d\n", width, height, origin_channels);

  // Allocate memory on host
  final_result_RGB = (unsigned char*)calloc(height*width*3, sizeof(unsigned char));

  // Allocate memory on device
  cudaStat = cudaMalloc((void**)&red_dev,sizeof(cufftDoubleComplex) * height * width);
  assert(cudaStat == cudaSuccess);
  cudaStat = cudaMalloc((void**)&green_dev,sizeof(cufftDoubleComplex) * height * width);
  assert(cudaStat == cudaSuccess);
  cudaStat = cudaMalloc((void**)&blue_dev,sizeof(cufftDoubleComplex) * height * width);
  assert(cudaStat == cudaSuccess);
  cudaStat = cudaMalloc((void**)&rgb_image_dev, sizeof(unsigned char)*height*width*origin_channels);
  assert(cudaStat == cudaSuccess);
  cudaStat = cudaMalloc((void**)&final_result_RGB_dev, sizeof(unsigned char)*width*height*3);

  // allocate memory for result
  fft_result = (cufftDoubleComplex *)calloc(height*width, sizeof(cufftDoubleComplex));
  Ifft_result = (cufftDoubleComplex*)calloc(width*height, sizeof(cufftDoubleComplex));

  // create plans
  cuError = cufftPlan2d(&planZ2Z, width, height, CUFFT_Z2Z);
  assert(cuError == CUFFT_SUCCESS);
  cuError = cufftPlan2d(&planIZ2Z, width, height, CUFFT_Z2Z);
  assert(cuError == CUFFT_SUCCESS);

  //define block and grid dimensions
	const dim3 dimGrid((int)ceil((width)/16), (int)ceil((height)/16));
	const dim3 dimBlock(16, 16);

  // create event
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start the timer
  cudaEventRecord(start);

  // copy rgb image to device
  cudaStat = cudaMemcpy(rgb_image_dev, rgb_image_chars, sizeof(unsigned char)*height*width*origin_channels, cudaMemcpyHostToDevice);
  assert(cudaStat == cudaSuccess);
  
  // Get RGB three channel data (changed alittle bit, no need intermediate red_doubles_dev)
  getChannelData<<<dimGrid, dimBlock>>>(rgb_image_dev, red_dev, height, width, origin_channels, 0);
  getChannelData<<<dimGrid, dimBlock>>>(rgb_image_dev, green_dev, height, width, origin_channels, 1);
  getChannelData<<<dimGrid, dimBlock>>>(rgb_image_dev, blue_dev, height, width, origin_channels, 2);
  // if(origin_channels == 4) getChannelData<<<dimGrid, dimBlock>>>(rgb_image_dev, alpha_dev, height, width, origin_channels, 3);

  // Perform FFT on RGB channels separately
  // Red:
  thrust::device_vector<cufftDoubleComplex> fft_result_red_dev(height*width);
  cufftDoubleComplex *_fft_result_red_dev = (cufftDoubleComplex *)thrust::raw_pointer_cast(fft_result_red_dev.data());
  cuError = cufftExecZ2Z(planZ2Z, red_dev, _fft_result_red_dev, CUFFT_FORWARD);
  // Green:
  thrust::device_vector<cufftDoubleComplex> fft_result_blue_dev(height*width);
  cufftDoubleComplex *_fft_result_blue_dev = (cufftDoubleComplex *)thrust::raw_pointer_cast(fft_result_blue_dev.data());
  cuError = cufftExecZ2Z(planZ2Z, blue_dev, _fft_result_blue_dev, CUFFT_FORWARD);
  // Blue:
  thrust::device_vector<cufftDoubleComplex> fft_result_green_dev(height*width);
  cufftDoubleComplex *_fft_result_green_dev = (cufftDoubleComplex *)thrust::raw_pointer_cast(fft_result_green_dev.data());
  cuError = cufftExecZ2Z(planZ2Z, green_dev, _fft_result_green_dev, CUFFT_FORWARD);

  // sorting on the copy of fft_result_dev (fft_result_sorted_dev)
  // Red:
  thrust::device_vector<cufftDoubleComplex> fft_result_red_sorted_dev(height*width);
  thrust::copy(fft_result_red_dev.begin(), fft_result_red_dev.end(), fft_result_red_sorted_dev.begin());
  cufftDoubleComplex * _fft_result_red_sorted_dev = (cufftDoubleComplex *)thrust::raw_pointer_cast(fft_result_red_sorted_dev.data());
  thrust::sort(fft_result_red_sorted_dev.begin(), fft_result_red_sorted_dev.end(), thrust::greater<cufftDoubleComplex>());
  // Blue:
  thrust::device_vector<cufftDoubleComplex> fft_result_blue_sorted_dev(height*width);
  thrust::copy(fft_result_blue_dev.begin(), fft_result_blue_dev.end(), fft_result_blue_sorted_dev.begin());
  cufftDoubleComplex * _fft_result_blue_sorted_dev = (cufftDoubleComplex *)thrust::raw_pointer_cast(fft_result_blue_sorted_dev.data());
  thrust::sort(fft_result_blue_sorted_dev.begin(), fft_result_blue_sorted_dev.end(), thrust::greater<cufftDoubleComplex>());
  // Green:
  thrust::device_vector<cufftDoubleComplex> fft_result_green_sorted_dev(height*width);
  thrust::copy(fft_result_green_dev.begin(), fft_result_green_dev.end(), fft_result_green_sorted_dev.begin());
  cufftDoubleComplex * _fft_result_green_sorted_dev = (cufftDoubleComplex *)thrust::raw_pointer_cast(fft_result_green_sorted_dev.data());
  thrust::sort(fft_result_green_sorted_dev.begin(), fft_result_green_sorted_dev.end(), thrust::greater<cufftDoubleComplex>());
  
  // preserve only the values that are larger than the threshold value for all RGB channels
  int cutoffIndex = height*width*THRESHOLD;
  cutoff<<<dimGrid, dimBlock>>>(_fft_result_red_dev, _fft_result_red_sorted_dev, cutoffIndex, height, width);
  cutoff<<<dimGrid, dimBlock>>>(_fft_result_green_dev, _fft_result_green_sorted_dev, cutoffIndex, height, width);
  cutoff<<<dimGrid, dimBlock>>>(_fft_result_blue_dev, _fft_result_blue_sorted_dev, cutoffIndex, height, width);

  // // Do the IFFT on the fft_result_dev
  // Red:
  thrust::device_vector<cufftDoubleComplex> Ifft_result_red_dev(height*width);
  cufftDoubleComplex * _Ifft_result_red_dev = (cufftDoubleComplex *)thrust::raw_pointer_cast(Ifft_result_red_dev.data());
  cuError = cufftExecZ2Z(planIZ2Z, _fft_result_red_dev, _Ifft_result_red_dev, CUFFT_INVERSE);
  assert(cuError == CUFFT_SUCCESS);
  // Blue:
  thrust::device_vector<cufftDoubleComplex> Ifft_result_blue_dev(height*width);
  cufftDoubleComplex * _Ifft_result_blue_dev = (cufftDoubleComplex *)thrust::raw_pointer_cast(Ifft_result_blue_dev.data());
  cuError = cufftExecZ2Z(planIZ2Z, _fft_result_blue_dev, _Ifft_result_blue_dev, CUFFT_INVERSE);
  assert(cuError == CUFFT_SUCCESS);
  // Green:
  thrust::device_vector<cufftDoubleComplex> Ifft_result_green_dev(height*width);
  cufftDoubleComplex * _Ifft_result_green_dev = (cufftDoubleComplex *)thrust::raw_pointer_cast(Ifft_result_green_dev.data());
  cuError = cufftExecZ2Z(planIZ2Z, _fft_result_green_dev, _Ifft_result_green_dev, CUFFT_INVERSE);
  assert(cuError == CUFFT_SUCCESS);

  // Find max element of IFFT for RGB channels
  thrust::device_vector<cufftDoubleComplex>::iterator IFFT_red_max_iter = thrust::max_element(Ifft_result_red_dev.begin(), Ifft_result_red_dev.end());
  thrust::device_vector<cufftDoubleComplex>::iterator IFFT_green_max_iter = thrust::max_element(Ifft_result_green_dev.begin(), Ifft_result_green_dev.end());
  thrust::device_vector<cufftDoubleComplex>::iterator IFFT_blue_max_iter = thrust::max_element(Ifft_result_blue_dev.begin(), Ifft_result_blue_dev.end());

  // Scale every data with the max 
  // Red:
  cufftDoubleComplex IFFT_red_max = *IFFT_red_max_iter;
  generateResultImageRGB<<<dimGrid, dimBlock>>>(_Ifft_result_red_dev, IFFT_red_max, final_result_RGB_dev, height, width,3,0);
  // Green:
  cufftDoubleComplex IFFT_green_max = *IFFT_green_max_iter;
  generateResultImageRGB<<<dimGrid, dimBlock>>>(_Ifft_result_green_dev, IFFT_green_max, final_result_RGB_dev, height, width,3,1);
  // Blue:
  cufftDoubleComplex IFFT_blue_max = *IFFT_blue_max_iter;
  generateResultImageRGB<<<dimGrid, dimBlock>>>(_Ifft_result_blue_dev, IFFT_blue_max, final_result_RGB_dev, height, width,3,2);
  
  // back to host
  cudaStat = cudaMemcpy(final_result_RGB, final_result_RGB_dev, sizeof(unsigned char)*height*width*3, cudaMemcpyDeviceToHost);
  assert(cudaStat == cudaSuccess);

  // stop the timer and compute time consumption
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time,start,stop);

  // write to the image file
  stbi_write_jpg(OUTPUTFILE_JPG, width, height, 3, final_result_RGB, 100);
  printf("Finishes writing to %s\n", OUTPUTFILE_JPG);
  printf("Total time consumption: %f\n",elapsed_time);

  stbi_image_free(rgb_image_chars); 
  free(fft_result); free(Ifft_result); free(final_result_RGB);
  cudaFree(red_dev); cudaFree(blue_dev); cudaFree(green_dev);
  cudaFree(rgb_image_dev); cudaFree(final_result_RGB_dev); 
  return 0;
}