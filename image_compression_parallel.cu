// nvcc image_compression_parallel.cu -o parallel -lm -lcufft -w
// ./parallel 1

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
#define OUTPUT_GRAY_PNG "./result_images/image_gray.png"
#define OUTPUTFILE_JPG "./result_images/result_gray.jpg"
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
  cufftDoubleComplex *gray_image_dev, *fft_result, *Ifft_result;
  unsigned char *rgb_image_dev, *gray_image_chars_dev, *gray_image_chars;
  unsigned char *final_result_dev, *final_result;
  double *gray_image_doubles_dev;
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
  int gray_channels = origin_channels == 4?2:1;
  gray_image_chars = (unsigned char*)calloc(height*width*gray_channels, sizeof(unsigned char));
  final_result = (unsigned char*)calloc(height*width, sizeof(unsigned char));

  // Allocate memory on device
  cudaStat = cudaMalloc((void**)&gray_image_doubles_dev, sizeof(double) * height * width);
  assert(cudaStat == cudaSuccess);
  cudaStat = cudaMalloc((void**)&gray_image_chars_dev, sizeof(unsigned char)*height*width*gray_channels);
  assert(cudaStat == cudaSuccess);
  cudaStat = cudaMalloc((void**)&rgb_image_dev, sizeof(unsigned char)*height*width*origin_channels);
  assert(cudaStat == cudaSuccess);
  cudaStat = cudaMalloc((void**)&gray_image_dev , sizeof(cufftDoubleComplex)*width*height);
  assert(cudaStat == cudaSuccess);
  cudaStat = cudaMalloc((void**)&final_result_dev, sizeof(unsigned char)*width*height);
  assert(cudaStat == cudaSuccess);

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
  
  // Generate gray scale image
  toGrayScaleImage<<<dimGrid,dimBlock>>>(rgb_image_dev, gray_image_chars_dev, height, width, origin_channels, gray_channels);
  cudaStat = cudaMemcpy(gray_image_chars, gray_image_chars_dev, sizeof(unsigned char)*height*width*gray_channels, cudaMemcpyDeviceToHost);
  assert(cudaStat == cudaSuccess);
  // save the result as a comparison
  stbi_write_png(OUTPUT_GRAY_PNG, width, height, gray_channels, gray_image_chars, width*gray_channels);

  // convert to double data 
  toGrayScaleData<<<dimGrid, dimBlock>>>(rgb_image_dev, gray_image_doubles_dev, height, width, origin_channels);

  // convert to cufftComplex matrix
  copyToComplex<<<dimGrid, dimBlock>>>(gray_image_doubles_dev, gray_image_dev, height, width);

  // // Perform FFT on image
  thrust::device_vector<cufftDoubleComplex> fft_result_dev(height*width);
  cufftDoubleComplex *_fft_result_dev = (cufftDoubleComplex *)thrust::raw_pointer_cast(fft_result_dev.data());
  cuError = cufftExecZ2Z(planZ2Z, gray_image_dev, _fft_result_dev, CUFFT_FORWARD);
  assert(cuError == CUFFT_SUCCESS);

  // sorting on the copy of fft_result_dev (fft_result_sorted_dev)
  thrust::device_vector<cufftDoubleComplex> fft_result_sorted_dev(height*width);
  thrust::copy(fft_result_dev.begin(), fft_result_dev.end(), fft_result_sorted_dev.begin());
  cufftDoubleComplex * _fft_result_sorted_dev = (cufftDoubleComplex *)thrust::raw_pointer_cast(fft_result_sorted_dev.data());
  thrust::sort(fft_result_sorted_dev.begin(), fft_result_sorted_dev.end(), thrust::greater<cufftDoubleComplex>());
  
  // preserve only the values that are larger than the threshold value
  int cutoffIndex = height*width*(THRESHOLD);
  cutoff<<<dimGrid, dimBlock>>>(_fft_result_dev, _fft_result_sorted_dev, cutoffIndex, height, width);

  // Do the IFFT on the fft_result_dev
  thrust::device_vector<cufftDoubleComplex> Ifft_result_dev(height*width);
  cufftDoubleComplex * _Ifft_result_dev = (cufftDoubleComplex *)thrust::raw_pointer_cast(Ifft_result_dev.data());
  cuError = cufftExecZ2Z(planIZ2Z, _fft_result_dev, _Ifft_result_dev, CUFFT_INVERSE);
  assert(cuError == CUFFT_SUCCESS);
  // find the max IFFT value
  thrust::device_vector<cufftDoubleComplex>::iterator IFFT_max_iter = thrust::max_element(Ifft_result_dev.begin(), Ifft_result_dev.end());

  // Scale every data with the max 
  cufftDoubleComplex IFFT_max = *IFFT_max_iter;
  generateResultImage<<<dimGrid, dimBlock>>>(_Ifft_result_dev, IFFT_max, final_result_dev, height, width);
  cudaStat = cudaMemcpy(final_result, final_result_dev, sizeof(unsigned char)*height*width, cudaMemcpyDeviceToHost);
  assert(cudaStat == cudaSuccess);

  // stop the timer and compute time consumption
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time,start,stop);

  // write to the image file
  stbi_write_jpg(OUTPUTFILE_JPG, width, height, 1, final_result, 100);
  printf("Finishes writing to %s\n", OUTPUTFILE_JPG);
  printf("Total time consumption: %f\n",elapsed_time);

  stbi_image_free(rgb_image_chars); 
  free(gray_image_chars); free(fft_result); free(Ifft_result); free(final_result);
  cudaFree(gray_image_dev); cudaFree(final_result_dev);
  cudaFree(rgb_image_dev); cudaFree(gray_image_chars_dev);
  cudaFree(gray_image_doubles_dev);

  return 0;
}