#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
//#include <acml.h> //assumes AMD platform
#include <mkl.h>

/* Your function must have the following signature: */

extern const char* dgemm_desc;

extern void square_dgemm(int M, double *A, double *B, double *C, int *argc, char ***argv);

/* Helper functions */

double read_timer() {
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if (!initialized) {
        gettimeofday(&start, NULL);
        initialized = true;
    }

    gettimeofday(&end, NULL);

    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

void fill(double *p, int n) {
    for (int i = 0; i < n; i++)
        p[i] = 2 * drand48() - 1;
}

void absolute_value(double *p, int n) {
    for (int i = 0; i < n; i++)
        p[i] = fabs(p[i]);
}


/* The benchmarking program */

int main(int argc, char **argv) {
    printf("Description:\t%s\n\n", dgemm_desc);
    /* These sizes should highlight performance dips at multiples of certain powers-of-two */
    int test_sizes[] = {1600};
    /*For each test size*/

    for (int isize = 0; isize < sizeof(test_sizes) / sizeof(test_sizes[0]); isize++) {
        /*Craete and fill 3 random matrices A,B,C*/
        int n = test_sizes[isize];

        double *A = (double *) malloc(n * n * sizeof(double));
        double *B = (double *) malloc(n * n * sizeof(double));
        double *C = (double *) malloc(n * n * sizeof(double));

        fill(A, n * n);
        fill(B, n * n);
        fill(C, n * n);

        /*  measure Mflop/s rate; time a sufficiently long sequence of calls to eliminate noise*/
        double Mflop_s, seconds = -1.0;
        double seconds_per_iteration = 0;
        const double BETA = 1.0;
        const double negOne = -1.0;
        char *ntran = "N";
        double neg3EpsilonN = -3.0 * DBL_EPSILON * n;

        for (int n_iterations = 1; seconds < 0.1; n_iterations *= 2) {
            /* warm-up */
            square_dgemm( n, A, B, C, &argc, &argv );
            /*  measure time */
            seconds = read_timer();
            for (int i = 0; i < n_iterations; i++)
                square_dgemm(n, A, B, C, &argc, &argv);
            seconds = read_timer() - seconds;

            /*  compute Mflop/s rate */
            Mflop_s = 2e-6 * n_iterations * n * n * n / seconds;
            seconds_per_iteration = seconds / n_iterations;
        }
        printf ("Size: %d\tMflop/s: %g\tTime: %g\n", n, Mflop_s, seconds_per_iteration);

        int taskid;
        MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
        printf("Task=%d\n", taskid);
        printf("Size: %d\tMflop/s: %g\n", n, Mflop_s);
        printf("Time: %g\n", seconds);
        MPI_Finalize();

        /*Deallocate memory*/
        free(C);
        free(B);
        free(A);
    }

    return 0;
}
