// CUBLAS example modified from http://courses.cms.caltech.edu/cs179/

#include <stdio.h>
#include <time.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define N (2 * 2 * 275)

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C) {
  int i, j, k, prod;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      prod = 0;

      for (k = 0; k < n; ++k) {
        prod += A[k * n + i] * B[j * n + k];
      }

      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}

// https://bluewaters.ncsa.illinois.edu/liferay-content/image-gallery/content/BLA-final
__global__ void gpu_gemm(int n, float alpha, const float *__restrict__ A,
                         const float *__restrict__ B, float beta,
                         float *__restrict__ C) {
  size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
  size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
  float prod;
  for (size_t j = ty; j < n; j += gridDim.y * blockDim.y) {
    for (size_t i = tx; i < n; i += gridDim.x * blockDim.x) {
      prod = 0;

      for (size_t k = 0; k < n; ++k) {
        prod += A[k * n + i] * B[j * n + k];
      }

      C[j * n + i] += alpha * prod * beta * C[j * n + i];
    }
  }
}

int main(int argc, char **argv) {
  float alpha = 1.0f;
  float beta = 0.0f;
  int n2 = N * N;

  int i;

  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle));

  float *h_A = (float *)malloc(n2 * sizeof(h_A[0]));
  float *h_B = (float *)malloc(n2 * sizeof(h_B[0]));
  float *h_C = (float *)malloc(n2 * sizeof(h_C[0]));

  for (i = 0; i < n2; i++) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
    h_C[i] = rand() / (float)RAND_MAX;
  }

  clock_t tstart, tend;
  float cpu_duration;
  tstart = clock();
  simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
  tend = clock();
  cpu_duration = ((float)(tend - tstart)) / CLOCKS_PER_SEC;
  printf("Time for sum on CPU: %f seconds\n", cpu_duration);
  float *h_C_ref = h_C;

  /************GPU Version***********/

  clock_t tstart_total;
  tstart_total = clock();

  float *d_A = 0;
  float *d_B = 0;
  float *d_C = 0;

  cudaMalloc((void **)&d_A, n2 * sizeof(d_A[0]));
  cudaMalloc((void **)&d_B, n2 * sizeof(d_B[0]));
  cudaMalloc((void **)&d_C, n2 * sizeof(d_C[0]));

  checkCudaErrors(cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1));
  checkCudaErrors(cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1));
  checkCudaErrors(cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1));

  float gpu_duration;
  tstart = clock();
  checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                              d_A, N, d_B, N, &beta, d_C, N));
  cudaDeviceSynchronize();
  tend = clock();
  gpu_duration = ((float)(tend - tstart)) / CLOCKS_PER_SEC;
  printf("Kernel time for sum on GPU: %f seconds\n", gpu_duration);

  h_C = (float *)malloc(n2 * sizeof(h_C[0]));
  checkCudaErrors(cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1));
  cudaDeviceSynchronize();
  tend = clock();
  gpu_duration = ((float)(tend - tstart_total)) / CLOCKS_PER_SEC;
  printf("Total time for sum on GPU: %f seconds\n", gpu_duration);

  printf("speed-up: %.1f\n", cpu_duration / gpu_duration);

  /************Check correctness using RMS Error ***********/
  float error_norm = 0;
  float ref_norm = 0;
  float diff;

  for (i = 0; i < n2; ++i) {
    diff = h_C_ref[i] - h_C[i];
    error_norm += diff * diff;
    ref_norm += h_C_ref[i] * h_C_ref[i];
  }

  error_norm = (float)sqrt((double)error_norm / n2);
  ref_norm = (float)sqrt((double)ref_norm);

  printf("Error %f, reference %f\n", error_norm, ref_norm);

  /************My Kernel***********/

  tstart_total = clock();

  tstart = clock();
  dim3 threads(32, 32);
  dim3 blocks((N - 1) / 32 + 1, (N - 1) / 32 + 1);
  gpu_gemm<<<blocks, threads>>>(N, alpha, d_A, d_B, beta, d_C);
  cudaDeviceSynchronize();
  tend = clock();
  gpu_duration = ((float)(tend - tstart)) / CLOCKS_PER_SEC;
  printf("Kernel time for sum on GPU: %f seconds\n", gpu_duration);

  h_C = (float *)malloc(n2 * sizeof(h_C[0]));
  checkCudaErrors(cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1));
  cudaDeviceSynchronize();
  tend = clock();
  gpu_duration = ((float)(tend - tstart_total)) / CLOCKS_PER_SEC;
  printf("Total time for sum on GPU: %f seconds\n", gpu_duration);

  printf("speed-up: %.1f\n", cpu_duration / gpu_duration);

  /************Check correctness using RMS Error ***********/
  error_norm = 0;
  ref_norm = 0;

  for (i = 0; i < n2; ++i) {
    diff = h_C_ref[i] - h_C[i];
    error_norm += diff * diff;
    ref_norm += h_C_ref[i] * h_C_ref[i];
  }

  error_norm = (float)sqrt((double)error_norm / n2);
  ref_norm = (float)sqrt((double)ref_norm);

  printf("Error %f, reference %f\n", error_norm, ref_norm);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  checkCudaErrors(cublasDestroy(handle));
}