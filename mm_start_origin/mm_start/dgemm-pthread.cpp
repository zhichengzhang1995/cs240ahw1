/*In case you're wondering, dgemm stands for Double-precision, GEneral Matrix-Matrix multiplication.*/
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
const char* dgemm_desc = "blocked dgemm pthread.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a,b) (((a)<(b))?(a):(b))

struct matrixmul {
		int count;	//numt
		int rank;	//id
		int l;		//lda
		double *a;
		double *b;
		double *c;
	};

void * dgemm_per_thread(void * mm_t);


void do_block( int lda, int M, int N, int K, double *A, double *B, double *C )
{
/* To optimize this, think about loop unrolling and software
    pipelining.  Hint:  For the majority of the matmuls, you
    know exactly how many iterations there are (the block size)...  */
	for( int i = 0; i < M; i++ )
		for( int j = 0; j < N; j++ ) 
		{
			double cij = C[i+j*lda];
			for( int k = 0; k < K; k++ )
				cij += A[i+k*lda] * B[k+j*lda];
			C[i+j*lda] = cij;
       }
	
}

void square_dgemm( int lda, double *A, double *B, double *C )
{	
	
	pthread_t *thread_handles;
	int thread_count = 4;
	thread_handles = (pthread_t *)malloc(thread_count *sizeof(pthread_t));
	
	matrixmul mm[thread_count];
	for(int i=0; i<thread_count; i++)
	{
		mm[i].count = thread_count;
		mm[i].rank = i;
		mm[i].l = lda;
		mm[i].a = A;
		mm[i].b = B;
		mm[i].c = C;
		pthread_create(&thread_handles[i], NULL, dgemm_per_thread, (void*)(&mm[i]));
	}

	//printf("%d threads created\n", thread_count);
	for(int i=0; i<thread_count; i++)
		pthread_join(thread_handles[i], NULL);
	free(thread_handles);

}

//function to be called by each thread, to be modified
	/*For each block combination*/
	
void * dgemm_per_thread(void * mm_t)	{
		matrixmul * pmm = (matrixmul*) mm_t;
		int numt=pmm->count;
		int id=pmm->rank;
		int lda =pmm->l;
		int col_num;	//number of block column
		int col_per_thread;	//number of block column per thread

		if(lda % BLOCK_SIZE ==0)	col_num=lda/BLOCK_SIZE;
		else						col_num=lda/BLOCK_SIZE+1;
		if(col_num % numt ==0)		col_per_thread = col_num/numt;
		else						col_per_thread = col_num/numt+1;
		
		int start_col = id*col_per_thread*BLOCK_SIZE;
		int end_col = min((start_col+col_per_thread*BLOCK_SIZE),lda);

		for( int i = 0; i < lda; i += BLOCK_SIZE )
			for( int j = start_col; j < end_col; j += BLOCK_SIZE )
				for( int k = 0; k < lda; k += BLOCK_SIZE )
				{
				/*This gets the correct block size (for fringe blocks also)*/
				int M = min( BLOCK_SIZE, lda-i );
				int N = min( BLOCK_SIZE, lda-j );
				int K = min( BLOCK_SIZE, lda-k );
				
				do_block(lda, M, N, K, (pmm->a) + i + k*lda, (pmm->b) + k + j*lda, (pmm->c) + i + j*lda);
				}

	}
