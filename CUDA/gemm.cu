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

      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}

template <int TILE_EXT_Y = 32, int TILE_EXT_X = 32, int TILE_EXT_K = 64>
__global__ void gpu_gemm_sh_nn(int n, float alpha, const float *__restrict__ A,
                               const float *__restrict__ B, float beta,
                               float *__restrict__ C) {
  __shared__ float abuf[TILE_EXT_K][TILE_EXT_X], bbuf[TILE_EXT_Y][TILE_EXT_K];

  // tile offset in Y
  for (int y_pos = blockIdx.y * blockDim.y; y_pos < n;
       y_pos += gridDim.y * blockDim.y) {
    // tile offset in X
    for (int x_pos = blockIdx.x * blockDim.x; x_pos < n;
         x_pos += gridDim.x * blockDim.x) {
      float tmp = 0.0;  // accumulator

      // k_pos is the position of the CUDA thread along the K dimension
      for (int k_pos = 0; k_pos < n; k_pos += TILE_EXT_K) {
        int k_end = k_pos + TILE_EXT_K;
        if (k_end > n) {
          k_end = n;
        }

        // Load a tile of matrix A(x_pos:TILE_EXT_X, k_pos:TILE_EXT_K):
        if (x_pos + threadIdx.x < n) {
          for (int k_loc = k_pos + threadIdx.y; k_loc < k_end;
               k_loc += blockDim.y) {
            abuf[k_loc - k_pos][threadIdx.x] =
                A[k_loc * n + (x_pos + threadIdx.x)];
          }
        }

        // Load a tile of matrix B(k_pos:TILE_EXT_K, y_pos:TILE_EXT_Y):
        if (y_pos + threadIdx.y < n) {
          for (int k_loc = k_pos + threadIdx.x; k_loc < k_end;
               k_loc += blockDim.x) {
            bbuf[threadIdx.y][k_loc - k_pos] =
                B[(y_pos + threadIdx.y) * n + k_loc];
          }
        }
        __syncthreads();

        // Multiply two loaded tiles to produce a tile of matrix
        // C(x_pos:TILE_EXT_X, y_pos:TILE_EXT_Y):
        if (x_pos + threadIdx.x < n && y_pos + threadIdx.y < n) {
          // number of loop iterations is known at compile time: Unroll it
          if (k_end - k_pos == TILE_EXT_K) {
#pragma unroll
            for (int l = 0; l < TILE_EXT_K; ++l) {
              tmp += abuf[l][threadIdx.x] * bbuf[threadIdx.y][l];
            }
          } else {  // number of loop iterations is not known at compile time
            for (int l = 0; l < (k_end - k_pos); ++l) {
              tmp += abuf[l][threadIdx.x] * bbuf[threadIdx.y][l];
            }
          }
        }
        __syncthreads();

      }  // k_pos

      // Store element of the C matrix in global memory:
      if (x_pos + threadIdx.x < n && y_pos + threadIdx.y < n) {
        int c_offset = (y_pos + threadIdx.y) * n + (x_pos + threadIdx.x);
        C[c_offset] = alpha * tmp + beta * C[c_offset];
      }

    }  // x_pos

  }  // y_pos

  return;
}

template <int TILE_EXT_Y = 32, int TILE_EXT_X = 32, int TILE_EXT_K = 16>
__global__ void gpu_gemm_sh_reg_nn(int n, float alpha,
                                   const float *__restrict__ A,
                                   const float *__restrict__ B, float beta,
                                   float *__restrict__ C) {
  __shared__ float abuf[TILE_EXT_K][TILE_EXT_X], bbuf[TILE_EXT_Y][TILE_EXT_K];

  // tile offset in Y dimension
  for (int y_pos = blockIdx.y * TILE_EXT_Y; y_pos < n;
       y_pos += gridDim.y * TILE_EXT_Y) {
    int y_end = y_pos + TILE_EXT_Y;
    if (y_end > n) {
      y_end = n;
    }

    // tile offset in X dimension
    for (int x_pos = blockIdx.x * TILE_EXT_X; x_pos < n;
         x_pos += gridDim.x * TILE_EXT_X) {
      int x_end = x_pos + TILE_EXT_X;
      if (x_end > n) {
        x_end = n;
      }

      // complete tile C(TILE_EXT_X, TILE_EXT_Y)
      if ((x_end - x_pos == TILE_EXT_X) && (y_end - y_pos == TILE_EXT_Y)) {
        // Initialize registers to zero:
        float creg[4][4] = {0};
        float breg[4] = {0};
        float areg[4] = {0};

        // k_pos is the position of the CUDA thread along the K dimension
        for (int k_pos = 0; k_pos < n; k_pos += TILE_EXT_K) {
          int k_end = k_pos + TILE_EXT_K;
          if (k_end > n) {
            k_end = n;
          }

          // Load a tile of matrix A(x_pos:TILE_EXT_X, k_pos:TILE_EXT_K):
          for (int x_loc = x_pos + threadIdx.x; x_loc < x_end;
               x_loc += blockDim.x) {
            for (int k_loc = k_pos + threadIdx.y; k_loc < k_end;
                 k_loc += blockDim.y) {
              abuf[k_loc - k_pos][x_loc - x_pos] = A[k_loc * n + x_loc];
            }
          }

          // Load a tile of matrix B(k_pos:TILE_EXT_K, y_pos:TILE_EXT_Y):
          for (int y_loc = y_pos + threadIdx.y; y_loc < y_end;
               y_loc += blockDim.y) {
            for (int k_loc = k_pos + threadIdx.x; k_loc < k_end;
                 k_loc += blockDim.x) {
              bbuf[y_loc - y_pos][k_loc - k_pos] = B[y_loc * n + k_loc];
            }
          }
          __syncthreads();

          // Multiply two loaded tiles to produce a tile of matrix
          // C(x_pos:TILE_EXT_X, y_pos:TILE_EXT_Y):
          if (k_end - k_pos == TILE_EXT_K) {
#pragma unroll
            for (int l = 0; l < TILE_EXT_K; ++l) {
#pragma unroll
              for (int j = 0; j < 4; ++j)
                breg[j] = bbuf[threadIdx.y + blockDim.y * j][l];
#pragma unroll
              for (int j = 0; j < 4; ++j)
                areg[j] = abuf[l][threadIdx.x + blockDim.x * j];
#pragma unroll
              for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                  creg[j][i] += areg[i] * breg[j];
                }
              }
            }
          } else {
            for (int l = 0; l < (k_end - k_pos); ++l) {
#pragma unroll
              for (int j = 0; j < 4; ++j)
                breg[j] = bbuf[threadIdx.y + blockDim.y * j][l];
#pragma unroll
              for (int j = 0; j < 4; ++j)
                areg[j] = abuf[l][threadIdx.x + blockDim.x * j];
#pragma unroll
              for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                  creg[j][i] += areg[i] * breg[j];
                }
              }
            }
          }
          __syncthreads();

        }  // k_pos

        // Store elements of the C matrix in global memory:
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            int c_offset = (y_pos + threadIdx.y + blockDim.y * j) * n +
                           (x_pos + threadIdx.x + blockDim.x * i);
            C[c_offset] = alpha * creg[j][i] + beta * C[c_offset];
          }
        }

      } else {  // incomplete tile of C

        // Initialize registers to zero:
        float creg[4][4] = {0};
        float breg[4] = {0};
        float areg[4] = {0};

        // k_pos is the position of the CUDA thread along the K dimension
        for (int k_pos = 0; k_pos < n; k_pos += TILE_EXT_K) {
          int k_end = k_pos + TILE_EXT_K;
          if (k_end > n) {
            k_end = n;
          }

          // Load a tile of matrix A(x_pos:TILE_EXT_X, k_pos:TILE_EXT_K):
          for (int x_loc = x_pos + threadIdx.x; x_loc < x_end;
               x_loc += blockDim.x) {
            for (int k_loc = k_pos + threadIdx.y; k_loc < k_end;
                 k_loc += blockDim.y) {
              abuf[k_loc - k_pos][x_loc - x_pos] = A[k_loc * n + x_loc];
            }
          }

          // Load a tile of matrix B(k_pos:TILE_EXT_K, y_pos:TILE_EXT_Y):
          for (int y_loc = y_pos + threadIdx.y; y_loc < y_end;
               y_loc += blockDim.y) {
            for (int k_loc = k_pos + threadIdx.x; k_loc < k_end;
                 k_loc += blockDim.x) {
              bbuf[y_loc - y_pos][k_loc - k_pos] = B[y_loc * n + k_loc];
            }
          }
          __syncthreads();

          // Multiply two loaded tiles to produce a tile of matrix
          // C(x_pos:TILE_EXT_X,y_pos:TILE_EXT_Y):
          for (int l = 0; l < (k_end - k_pos); ++l) {
            for (int i = 0, j = threadIdx.y; j < y_end - y_pos;
                 j += blockDim.y, i++)
              breg[i] = bbuf[j][l];
            for (int i = 0, j = threadIdx.x; j < x_end - x_pos;
                 j += blockDim.x, i++)
              areg[i] = abuf[l][j];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
              for (int i = 0; i < 4; ++i) {
                creg[j][i] += areg[i] * breg[j];
              }
            }
          }
          __syncthreads();

        }  // k_pos

        // Store element of the C matrix in global memory:
        for (int j = 0, y_loc = y_pos + threadIdx.y; y_loc < y_end;
             y_loc += blockDim.y, j++) {
          for (int i = 0, x_loc = x_pos + threadIdx.x; x_loc < x_end;
               x_loc += blockDim.x, i++) {
            int c_offset = y_loc * n + x_loc;
            C[c_offset] = alpha * creg[j][i] + beta * C[c_offset];
          }
        }
      }

    }  // x_pos

  }  // y_pos
  return;
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
  printf("********My Kernal********\n");

  tstart_total = clock();

  tstart = clock();
  dim3 threads(32, 32);
  dim3 blocks((N - 1) / 32 + 1, (N - 1) / 32 + 1);
  gpu_gemm<<<blocks, threads>>>(N, alpha, d_A, d_B, beta, d_C);
  cudaDeviceSynchronize();
  tend = clock();
  gpu_duration = ((float)(tend - tstart)) / CLOCKS_PER_SEC;
  printf("Kernel time for sum on GPU: %f seconds\n", gpu_duration);

  // h_C = (float *)malloc(n2 * sizeof(h_C[0]));
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

  /************shared memory***********/
  printf("********shared memory********\n");

  tstart_total = clock();

  tstart = clock();
  // dim3 threads(32, 32);
  // dim3 blocks((N - 1) / 32 + 1, (N - 1) / 32 + 1);
  gpu_gemm_sh_nn<<<blocks, threads>>>(N, alpha, d_A, d_B, beta, d_C);
  cudaDeviceSynchronize();
  tend = clock();
  gpu_duration = ((float)(tend - tstart)) / CLOCKS_PER_SEC;
  printf("Kernel time for sum on GPU: %f seconds\n", gpu_duration);

  // h_C = (float *)malloc(n2 * sizeof(h_C[0]));
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

  /************shared memory and register***********/
  printf("********shared memory and register********\n");

  tstart_total = clock();

  tstart = clock();
  // dim3 threads(32, 32);
  // dim3 blocks((N - 1) / 32 + 1, (N - 1) / 32 + 1);
  gpu_gemm_sh_nn<<<blocks, threads>>>(N, alpha, d_A, d_B, beta, d_C);
  cudaDeviceSynchronize();
  tend = clock();
  gpu_duration = ((float)(tend - tstart)) / CLOCKS_PER_SEC;
  printf("Kernel time for sum on GPU: %f seconds\n", gpu_duration);

  // h_C = (float *)malloc(n2 * sizeof(h_C[0]));
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

  /************Cleaning up ***********/

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  checkCudaErrors(cublasDestroy(handle));
}