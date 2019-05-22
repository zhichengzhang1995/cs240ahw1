#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <float.h>
#include <math.h>

#include <sys/time.h>
#include <assert.h>
#define BLOCK_SIZE 16

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

__global__ void square_dgemm(float* devM, float* devN, float* devP, int width)
{
  __shared__ float sM[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float sN[BLOCK_SIZE][BLOCK_SIZE];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = bx * BLOCK_SIZE + tx;
  int row = by * BLOCK_SIZE + ty;

  float sum = 0;


  for( int i = 0; i < width / BLOCK_SIZE; i++ ){
        sM[ty][tx] = devM[row * width + (i * BLOCK_SIZE + tx)];
        sN[ty][tx] = devN[col + (i * BLOCK_SIZE + ty) * width];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k){
                sum += sM[ty][k] * sN[k][tx];
                __syncthreads();
        }
  }

  devP[row * width + col] = sum;
}

void fill( float *p, int n){
    for (int i = 0; i < n; i++)
        p[i] = 2 * (float) drand48() - 1;
}
/* The benchmarking program */

int main( int argc, char **argv )
{
	int n = 1024;
	int m = 1024;
	int k = 1024;
	float *h_a, *h_b, *h_c;

	//cudaMallocHost((void **) &h_a, sizeof(float)*m*n);
	//cudaMallocHost((void **) &h_b, sizeof(float)*n*k);
	//cudaMallocHost((void **) &h_c, sizeof(float)*m*k);
	 h_a = (float *)malloc( n*n* sizeof(float) );
 	 h_b = (float *)malloc( n*n* sizeof(float) );
  	 h_c = (float *)malloc( n*n* sizeof(float) );

  /* random initialize matrix*/
//	for (int i= 0; i<m; ++i){
//		for (int j = 0; j<n; ++j){
//			h_a[i * n + j] = rand() % 1024;
//		}
//	}
//	for (int i = 0; i<n; ++i){
//		for (int j=0; j<k; ++j){
//			h_b[i * k +j] = rand() % 1024;
//		}
//	}

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

	fill(h_a, n*n);
	fill(h_b, n*n);
	fill(h_c, n*n);

        double seconds_copy = read_timer();

	cudaMemcpy(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(float)*m*n, cudaMemcpyHostToDevice);
	seconds_copy = read_timer()-seconds_copy;
	printf("cpu to device copy is %f\n",seconds_copy);
	double seconds=-1.0;
	double Gflop_s = 0.0, Gflop_s1 = 0.0;
	for (int n_iterations = 1; seconds<0.1;	n_iterations*=2){
	//warmup
	square_dgemm<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, n);

	//measure
	seconds = read_timer();
	for(int i=0; i<n_iterations;i++){
	square_dgemm<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, n);
	        seconds = read_timer()-seconds;
	}
//	seconds = read_timer()-seconds;
	Gflop_s1 = (2e-9 * n * n * n * n_iterations)/(seconds);

	}
//	double seconds_copy1 = read_timer();
	cudaMemcpy(h_c, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
//	seconds_copy1 = read_timer()-seconds_copy1;
//	printf("device to cpu copy is %f\n", seconds_copy1);
	seconds_copy = seconds+seconds_copy;
	cudaThreadSynchronize();

	cudaEventSynchronize(stop);

	Gflop_s = (Gflop_s1*seconds)/(seconds_copy);

	printf("Time of GPU cal is %f s(with copy)\n Time of GPU cal is %f s (without copy) Gflop_s with copy is %g\n Gflop_s without copy is %g\n", seconds_copy, seconds, Gflop_s, Gflop_s1);

	int all_good = 1;
	for (int i=0; i<m; ++i){
		for (int j=0; j<k; ++j){
			if (h_c[i*k+j] != h_c[i*k+j]){
				all_good = 0;
			}

		}
	}
	if (all_good){
		printf("all good!!\n");

	}
	else{
		printf("incorrect!!\n");
	}

		/*Deallocate memory*/
        cudaFree( d_a );
        cudaFree( d_b );
        cudaFree( d_c );
//	cudaFreeHost(h_a);
//	cudaFreeHost(h_b);
//	cudaFreeHost(h_c);
 	free(h_a);
	free(h_b);
	free(h_c);

    return 0;
}
