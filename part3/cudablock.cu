#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#define BLOCK_SIZE 16

__global__ void square_dgemm(float* devM, float* devN, float* devP, int width)
{
  __shared__ float sM[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float sN[BLOCK_SIZE][BLOCK_SIZE];
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;

  float sum = 0;


  for( int i = 0; i < width / BLOCK_SIZE; i++ ){
        sM[threadIdx.y][threadIdx.x] = devM[row * width + (i * BLOCK_SIZE + threadIdx.x)];
        sN[threadIdx.y][threadIdx.x] = devN[col + (i * BLOCK_SIZE + threadIdx.y) * width];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k){
                sum += sM[threadIdx.y][k] * sN[k][threadIdx.x];
                __syncthreads();
        }
  }

  devP[row * width + col] = sum;
}

/* Helper functions */

double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }

    gettimeofday( &end, NULL );

    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

void fill( float *p, int n){
    for (int i = 0; i < n; i++)
        p[i] = 2 * (float) drand48() - 1;
}
/* The benchmarking program */

int main( int argc, char **argv )
{
	int n = 1600;
	int m = 1600;
	int k = 1600;
	float *A, *B, *C;

	A = (float *)malloc( n * n * sizeof(float) );
 	B = (float *)malloc( n * n * sizeof(float) );
  C = (float *)malloc( n * n * sizeof(float) );

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  int grid_rows = n / BLOCK_SIZE;
  int grid_cols = n / BLOCK_SIZE;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	float *d_a, *d_b, *d_c;
	cudaMalloc((void **) &d_a, sizeof(float)*m*n);
	cudaMalloc((void **) &d_b, sizeof(float)*n*k);
	cudaMalloc((void **) &d_c, sizeof(float)*m*k);

	fill(A, n*n);
	fill(B, n*n);
	fill(C, n*n);

  double time_total = read_timer();

	cudaMemcpy(d_a, A, sizeof(float)*m*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, sizeof(float)*m*n, cudaMemcpyHostToDevice);
	time_total = read_timer()-time_total;
  double time_copy = time_total;
	double time_cpu=-1.0;
	double Gigaflops = 0.0, Gigaflops_noCopy = 0.0;
	for (int n_iterations = 1; time_cpu<0.1;	n_iterations*=2){
    //warmup
    square_dgemm<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, n);
    //measure
    time_cpu = read_timer();
  	for(int i=0; i<n_iterations;i++){
    	square_dgemm<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, n);
    	        time_cpu = read_timer()-time_cpu;
  	}
    Gigaflops_noCopy = (2e-9 * n * n * n * n_iterations)/(time_cpu);
	}

	cudaMemcpy(C, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);

	time_total = time_cpu+time_total;
	cudaThreadSynchronize();
	cudaEventSynchronize(stop);

	Gigaflops = (Gigaflops_noCopy * time_cpu) / (time_total);

  printf("Total CPU time is %f s\n", time_total);
  printf("GPU CPU time is %f s\n", time_cpu);
  printf("Copy time is %f s\n", time_copy);
  printf("Total GPU Gigaflops is %f \n", Gigaflops);
  printf("No copy GPU Gigaflops is %f \n", Gigaflops_noCopy);

	int check = 0;
	for (int i = 0; i < m; ++i){
		for (int j = 0; j < k; ++j){
			if (C[i * k + j] != C[i * k + j]){
				check = 1;
			}

		}
	}

	if (check){
		printf("Wrong\n");
	}

  cudaFree( d_a );
  cudaFree( d_b );
  cudaFree( d_c );

 	free(A);
	free(B);
	free(C);

  return 0;
}
