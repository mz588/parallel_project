// This only performs FFT and then IFFT. No real image compression.
// nvcc image_compression.cu -o image_compression -lcufft -lm
// ./image_compression

#include<stdint.h>
#include<stdlib.h>
#include<stdio.h>
#include<cufft.h>
#include<math.h>
#include<thrust/sort.h>

// Necessary libs for reading in and writing to image files
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// #define STB_IMAGE_WRITE_IMPLEMENTATION
#define INPUTFILE "image.png"
#define OUTPUTFILE_JPG "result.jpg"
#define OUTPUTFILE_PNG "result.png"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

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

void toGrayScaleImageSerial(unsigned char*src, unsigned char*dest, int row, int col, int origin_channels, int newChannels){
  for(int i = 0; i< row; i++){
    for(int j = 0; j < col; j++) {
      unsigned char red, green, blue;
      red = src[i*col*origin_channels+j*origin_channels];
      green = src[i*col*origin_channels+j*origin_channels+1];
      blue = src[i*col*origin_channels+j*origin_channels+2];
      // gray image default to have only one channel (not RGB 3 channels)
      dest[i*col*newChannels+j*newChannels] = red*0.3+blue*0.11+green*0.59; 
      // alpha channel only for png files
      if(origin_channels == 4) dest[i*col*newChannels+j*newChannels + 1] = src[i*col*origin_channels+j*origin_channels+3];
    }
  }
}

// input: rgb image, double matrix (row-major), height, width, rgb_channels
void toGrayScaleDataSerial(unsigned char*src, double*dest, int row, int col, int origin_channels){
  for(int i = 0; i < row; i++){
    for(int j = 0; j < col; j++){
      double red, green, blue;
      red = (double)src[i*col*origin_channels+j*origin_channels];
      green = (double)src[i*col*origin_channels+j*origin_channels+1];
      blue = (double)src[i*col*origin_channels+j*origin_channels+2]; 
      dest[i*col+j] = (red*0.3+blue*0.11+green*0.59)/255; 
    }
  }
}

void copyToComplexSerial(double*src, cufftDoubleComplex*dest, int row, int col){
  for(int i = 0; i < row; i++){
    for(int j = 0; j < col; j++){
      dest[IDX2C(i,j,row)].x = src[i*col+j];
    }
  }
}

void generateArray(cufftDoubleComplex*dest, int row, int col){
  int index = 0;
  for(int i = 0; i < row; i++){
    for(int j = 0; j < col; j++) dest[IDX2C(i,j,row)].x = index++;
  }
}

int main(int argc, char* argv[]) {

  // Pointer to the memory of image on device
  cufftDoubleComplex *gray_image_dev, *fft_result_dev, *fft_result, *Ifft_result, *Ifft_result_dev, *gray_image_doubles_dev;
  unsigned char *rgb_image_dev, *gray_image_chars_dev, *gray_image_chars, *final_result;
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
  cudaStat = cudaMalloc((void**)&fft_result_dev, sizeof(cufftDoubleComplex)*width*height);
  assert(cudaStat == cudaSuccess);
  cudaStat = cudaMalloc((void**)&Ifft_result_dev, sizeof(cufftDoubleComplex)*width*height);
  assert(cudaStat == cudaSuccess);

  // allocate memory for result
  fft_result = (cufftDoubleComplex *)calloc(height*width, sizeof(cufftDoubleComplex));
  Ifft_result = (cufftDoubleComplex*)calloc(width*height, sizeof(cufftDoubleComplex));

  // create plans
  cuError = cufftPlan2d(&planZ2Z, width, height, CUFFT_Z2Z);
  assert(cuError == CUFFT_SUCCESS);
  cuError = cufftPlan2d(&planIZ2Z, width, height, CUFFT_Z2Z);
  assert(cuError == CUFFT_SUCCESS);

  // create event
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start the timer
  cudaEventRecord(start);

  // copy rgb image to device
  cudaStat = cudaMemcpy(rgb_image_dev, rgb_image_chars, sizeof(unsigned char)*height*width*origin_channels, cudaMemcpyHostToDevice);
  assert(cudaStat == cudaSuccess);

  // copy the gray image to host
  cudaStat = cudaMemcpy(gray_image_chars, gray_image_chars_dev, sizeof(unsigned char)*height*width*gray_channels, cudaMemcpyDeviceToHost);
  assert(cudaStat == cudaSuccess);

  // convert to double data (Serial)
  double *gray_image_doubles = (double*)calloc(height*width, sizeof(double));
  toGrayScaleDataSerial(rgb_image_chars, gray_image_doubles, height, width, origin_channels);

  // copy the image data to float and convert to col-major
  cufftDoubleComplex *gray_image_complex = (cufftDoubleComplex*) calloc(height*width, sizeof(cufftDoubleComplex));
  copyToComplexSerial(gray_image_doubles, gray_image_complex, height, width);

  // // copy the image data to device
  cudaStat = cudaMemcpy(gray_image_dev, gray_image_complex, sizeof(cufftDoubleComplex)*height*width, cudaMemcpyHostToDevice);
  assert(cudaStat == cudaSuccess);

  // // Perform FFT on image
  cuError = cufftExecZ2Z(planZ2Z, gray_image_dev, fft_result_dev, CUFFT_FORWARD);
  assert(cuError == CUFFT_SUCCESS);

  // // Do the IFFT on the fft_result_dev
  cuError = cufftExecZ2Z(planIZ2Z,fft_result_dev, Ifft_result_dev, CUFFT_INVERSE);
  assert(cuError == CUFFT_SUCCESS);

  cudaStat = cudaMemcpy(Ifft_result, Ifft_result_dev, sizeof(cufftDoubleComplex)*height*width, cudaMemcpyDeviceToHost);
  assert(cudaStat == cudaSuccess);

  // find the max value in the inverse fft result for scaling
  double max = 0.0;
  for(int i = 0; i < height; i++){
    for(int j = 0; j<width; j++){
      if(Ifft_result[i*width+j].x>max) {
        max = Ifft_result[i*width+j].x;
      }
    }
  }

  // Scale every data with the max 
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      final_result[i*width+j] = (unsigned char)((Ifft_result[IDX2C(i,j,height)].x)/max*255);
    }
  }

  // stop the timer and compute time consumption
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time,start,stop);

  // // write to the image file
  stbi_write_jpg(OUTPUTFILE_JPG, width, height, 1, final_result, 100);
  printf("Finishes writing to %s\n", OUTPUTFILE_JPG);
  printf("Time consumption: %f ms\n", elapsed_time);

  stbi_image_free(rgb_image_chars); 
  free(gray_image_chars); free(gray_image_doubles);
  free(fft_result); free(gray_image_complex);
  free(Ifft_result); free(final_result);
  cudaFree(fft_result_dev); cudaFree(gray_image_dev);
  cudaFree(Ifft_result_dev); cudaFree(rgb_image_dev);
  cudaFree(gray_image_doubles_dev); cudaFree(gray_image_chars_dev);

  return 0;
}