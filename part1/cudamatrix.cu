#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#define BLOCK_SIZE 24

__global__ void square_dgemm(float* devM, float* devN, float* devP, int width)
{
  __shared__ float B[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float A[BLOCK_SIZE][BLOCK_SIZE];
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  float sum = 0;
  for( int i = 0; i < width / BLOCK_SIZE; i++ ){
        B[threadIdx.y][threadIdx.x] = devM[row * width + (i * BLOCK_SIZE + threadIdx.x)];
        A[threadIdx.y][threadIdx.x] = devN[col + (i * BLOCK_SIZE + threadIdx.y) * width];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k){
                sum += B[threadIdx.y][k] * A[k][threadIdx.x];
                __syncthreads();
        }
  }
  devP[row * width + col] = sum;
}

double timer()
{
    static bool initialized = false;
    static struct timeval start_event;
    struct timeval end_event;
    if( !initialized )
    {
        gettimeofday( &start_event, NULL );
        initialized = true;
    }
    gettimeofday( &end_event, NULL );
    return 1.0e-6 * (end_event.tv_usec - start_event.tv_usec) + (end_event.tv_sec - start_event.tv_sec);
}

void fill( float *p, int n){
    for (int i = 0; i < n; i++)
        p[i] = 2 * (float) drand48() - 1;
}

bool check( float *C, int n, int k) {
	for (int i = 0; i < n; ++i){
		for (int j = 0; j < k; ++j){
			if (C[i * k + j] != C[i * k + j]){
				return false;
			}
		}
	}
  return true;
}

int main( int argc, char **argv )
{
	int n = 1600;
	int m = 1600;
	int k = 1600;
	float *A, *B, *C;

  // Mem aloc
	A = (float *)malloc( n * n * sizeof(float) );
 	B = (float *)malloc( n * n * sizeof(float) );
  C = (float *)malloc( n * n * sizeof(float) );
  float *A_cuda, *B_cuda, *C_cuda;
	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
  dim3 dimGrid((n / BLOCK_SIZE), (n / BLOCK_SIZE));
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	cudaMalloc((void **) &A_cuda, sizeof(float) * n * m);
	cudaMalloc((void **) &B_cuda, sizeof(float) * k * n);
	cudaMalloc((void **) &C_cuda, sizeof(float) * k * m);
	fill(A, n * n);
	fill(B, n * n);
	fill(C, n * n);

  // Timer: copy time
  double time_cpu = -1.0;
  double time_total = timer();
	cudaMemcpy(A_cuda, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
	cudaMemcpy(B_cuda, B, sizeof(float) * m * n, cudaMemcpyHostToDevice);
	time_total = timer() - time_total;
  double time_copy = time_total;

  // Timer: CPU time; Gflops
  double Gigaflops = 0.0, Gigaflops_noCopy = 0.0;
	for (int fresh = 1; time_cpu < 0.1;	fresh *= 2){
    square_dgemm<<<dimGrid,dimBlock>>>(A_cuda, B_cuda, C_cuda, n);
    time_cpu = timer();
  	for(int i = 0; i < fresh; i++){
    	square_dgemm<<<dimGrid,dimBlock>>>(A_cuda, B_cuda, C_cuda, n);
    	time_cpu = timer() - time_cpu;
  	}
    Gigaflops_noCopy = (2e-9 * n * n * n * fresh) / (time_cpu);
	}
	cudaMemcpy(C, C_cuda, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
	time_total += time_cpu;
  Gigaflops = (Gigaflops_noCopy * time_cpu) / (time_total);

	cudaThreadSynchronize();
	cudaEventSynchronize(stop_event);

  // Info
  printf("Total CPU time is %f s\n", time_total);
  printf("GPU CPU time is %f s\n", time_cpu);
  printf("Copy time is %f s\n", time_copy);
  printf("Total GPU Gigaflops is %f \n", Gigaflops);
  printf("No copy GPU Gigaflops is %f \n", Gigaflops_noCopy);

  // Check
	bool check_matrix = check(C, m, k);
	if (!check_matrix){
		printf("Wrong\n");
	}

  cudaFree( A_cuda );
  cudaFree( B_cuda );
  cudaFree( C_cuda );
 	free(A);
	free(B);
	free(C);
  return 0;
}
