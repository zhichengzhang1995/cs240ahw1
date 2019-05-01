/*In case you're wondering, dgemm stands for Double-precision, GEneral Matrix-Matrix multiplication.*/
#include <mpi.h>
#include <stdio.h>
#include <string.h>

const char *dgemm_desc = "Simple dgemm mpi.";

MPI_Status status;

void square_dgemm(int n, double *A, double *B, double *C, int *argc, char ***argv) {

    int rank, cores, tasks, source, numoftask, mtype;
    int cols, per_col, per_row, index;
    int i, j, k;
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank)
    MPI_Comm_size(MPI_COMM_WORLD, &cores)
    tasks = cores - 1;

    //master
    if (rank == 0) {
        printf("core= %d \n", cores);
        per_row = n % tasks;
        per_col = n / tasks;

        index = 0;
        for (numoftask = 1; numoftask <= tasks; numoftask++) {
            cols = (numoftask <= per_row) ? per_col + 1 : per_col;
            MPI_Send(
                    /* data         = */ &index,
                    /* count        = */ 1,
                    /* datatype     = */ MPI_INT,
                    /* destination  = */ numoftask,
                    /* tag          = */ 1,
                    /* communicator = */ MPI_COMM_WORLD);
            MPI_Send(
                    /* data         = */ &cols,
                    /* count        = */ 1,
                    /* datatype     = */ MPI_INT,
                    /* destination  = */ numoftask,
                    /* tag          = */ 1,
                    /* communicator = */ MPI_COMM_WORLD);
            MPI_Send(
                    /* data         = */ &B[index * n],
                    /* count        = */ cols * n,
                    /* datatype     = */ MPI_DOUBLE,
                    /* destination  = */ numoftask,
                    /* tag          = */ 1,
                    /* communicator = */ MPI_COMM_WORLD);
            MPI_Send(
                    /* data         = */ A,
                    /* count        = */ n * n,
                    /* datatype     = */ MPI_DOUBLE,
                    /* destination  = */ numoftask,
                    /* tag          = */ 1,
                    /* communicator = */ MPI_COMM_WORLD);
            index += cols;
        }

        for (i = 1; i <= tasks; i++) {
            source = i;
            MPI_Recv(
                    /* data         = */ &index,
                    /* count        = */ 1,
                    /* datatype     = */ MPI_INT,
                    /* source       = */ source,
                    /* tag          = */ 2,
                    /* communicator = */ MPI_COMM_WORLD,
                    /* status       = */ &status);
            MPI_Recv(
                    /* data         = */ &cols,
                    /* count        = */ 1,
                    /* datatype     = */ MPI_INT,
                    /* source       = */ source,
                    /* tag          = */ 2,
                    /* communicator = */ MPI_COMM_WORLD,
                    /* status       = */ &status);
            MPI_Recv(
                    /* data         = */ &C[index * n],
                    /* count        = */ cols * n,
                    /* datatype     = */ MPI_DOUBLE,
                    /* source       = */ source,
                    /* tag          = */ 2,
                    /* communicator = */ MPI_COMM_WORLD,
                    /* status       = */ &status);
        }

    }
    if (rank > 0) {
        MPI_Recv(
                /* data         = */ &index,
                /* count        = */ 1,
                /* datatype     = */ MPI_INT,
                /* source       = */ 0,
                /* tag          = */ 1,
                /* communicator = */ MPI_COMM_WORLD,
                /* status       = */ &status);
        MPI_Recv(
                /* data         = */ &cols,
                /* count        = */ 1,
                /* datatype     = */ MPI_INT,
                /* source       = */ 0,
                /* tag          = */ 1,
                /* communicator = */ MPI_COMM_WORLD,
                /* status       = */ &status);
        MPI_Recv(
                /* data         = */ &B[index * n],
                /* count        = */ cols * n,
                /* datatype     = */ MPI_DOUBLE,
                /* source       = */ 0,
                /* tag          = */ 1,
                /* communicator = */ MPI_COMM_WORLD,
                /* status       = */ &status);
        MPI_Recv(
                /* data         = */ A,
                /* count        = */ n * n,
                /* datatype     = */ MPI_DOUBLE,
                /* source       = */ 0,
                /* tag          = */ 1,
                /* communicator = */ MPI_COMM_WORLD,
                /* status       = */ &status);
        for (i = 0; i < n; i++)
            for (j = index; j < index + cols; j++) {
                double cij = 0;
                for (k = 0; k < n; k++)
                    cij += A[i + k * n] * B[k + j * n];
                C[i + j * n] = cij;
            }

        MPI_Send(
                /* data         = */ &index,
                /* count        = */ 1,
                /* datatype     = */ MPI_INT,
                /* destination  = */ 0,
                /* tag          = */ 2,
                /* communicator = */ MPI_COMM_WORLD);
        MPI_Send(
                /* data         = */ &cols,
                /* count        = */ 1,
                /* datatype     = */ MPI_INT,
                /* destination  = */ 0,
                /* tag          = */ 2,
                /* communicator = */ MPI_COMM_WORLD);
        MPI_Send(
                /* data         = */ &C[index * n],
                /* count        = */ cols * n,
                /* datatype     = */ MPI_DOUBLE,
                /* destination  = */ 0,
                /* tag          = */ 2,
                /* communicator = */ MPI_COMM_WORLD);
    }
    
}
