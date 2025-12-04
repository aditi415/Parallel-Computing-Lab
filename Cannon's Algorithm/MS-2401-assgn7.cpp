// cannon_mpi.cpp
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 3) {
        if (world_rank == 0) fprintf(stderr, "Usage: %s input.txt output.txt\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = 0; 
    int *A = nullptr;
    int *B = nullptr;
    int *C = nullptr;

    if (world_rank == 0) {
        FILE *fin = fopen(argv[1], "r");
        if (!fin) { perror("fopen input"); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fscanf(fin, "%d", &N) != 1) { fprintf(stderr, "Failed to read N\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        A = (int*)malloc(sizeof(int) * N * N);
        B = (int*)malloc(sizeof(int) * N * N);
        C = (int*)calloc(N * N, sizeof(int));
        for (int i = 0; i < N*N; ++i) {
            if (fscanf(fin, "%d", &A[i]) != 1) { fprintf(stderr, "Failed to read A[%d]\n", i); MPI_Abort(MPI_COMM_WORLD, 1); }
        }
        for (int i = 0; i < N*N; ++i) {
            if (fscanf(fin, "%d", &B[i]) != 1) { fprintf(stderr, "Failed to read B[%d]\n", i); MPI_Abort(MPI_COMM_WORLD, 1); }
        }
        fclose(fin);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int q = (int)round(sqrt((double)world_size));
    if (q * q != world_size) {
        if (world_rank == 0) fprintf(stderr, "Number of processes must be a perfect square. Got %d\n", world_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (N % q != 0) {
        if (world_rank == 0) fprintf(stderr, "Matrix size N=%d must be divisible by q=%d\n", N, q);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm Aditi;
    int dims[2] = { q, q };
    int periods[2] = { 1, 1 };
    int reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &Aditi);

    int newRank;
    MPI_Comm_rank(Aditi, &newRank);

    int coords[2];
    MPI_Cart_coords(Aditi, newRank, 2, coords);
    int prow = coords[0], pcol = coords[1];

    int block = N / q;     
    int blockSize = block * block;

    int *localA = (int*)calloc(blockSize, sizeof(int));
    int *localB = (int*)calloc(blockSize, sizeof(int));
    int *localC = (int*)calloc(blockSize, sizeof(int));

    if (newRank == 0) {
        for (int r = 0; r < world_size; ++r) {
            int rc[2];
            MPI_Cart_coords(Aditi, r, 2, rc);
            int brow = rc[0], bcol = rc[1];

            vector<int> tmpA(blockSize), tmpB(blockSize);
            for (int i = 0; i < block; ++i) {
                for (int j = 0; j < block; ++j) {
                    int gi = brow * block + i;
                    int gj = bcol * block + j;
                    tmpA[i*block + j] = A[gi * N + gj];
                    tmpB[i*block + j] = B[gi * N + gj];
                }
            }
            if (r == 0) {
                memcpy(localA, tmpA.data(), sizeof(int) * blockSize);
                memcpy(localB, tmpB.data(), sizeof(int) * blockSize);
            } else {
                MPI_Send(tmpA.data(), blockSize, MPI_INT, r, 0, Aditi);
                MPI_Send(tmpB.data(), blockSize, MPI_INT, r, 0, Aditi);
            }
        }
    } else {
        MPI_Recv(localA, blockSize, MPI_INT, 0, 0, Aditi, MPI_STATUS_IGNORE);
        MPI_Recv(localB, blockSize, MPI_INT, 0, 0, Aditi, MPI_STATUS_IGNORE);
    }

    int src, dst;
    MPI_Cart_shift(Aditi, 1, -prow, &src, &dst);
    for (int s = 0; s < prow; ++s) {
        MPI_Sendrecv_replace(localA, blockSize, MPI_INT, dst, 111, src, 111, Aditi, MPI_STATUS_IGNORE);
    }

    MPI_Cart_shift(Aditi, 0, -pcol, &src, &dst);
    for (int s = 0; s < pcol; ++s) {
        MPI_Sendrecv_replace(localB, blockSize, MPI_INT, dst, 222, src, 222, Aditi, MPI_STATUS_IGNORE);
    }

    for (int step = 0; step < q; ++step) {
        for (int i = 0; i < block; ++i) {
            for (int k = 0; k < block; ++k) {
                int a = localA[i*block + k];
                for (int j = 0; j < block; ++j) {
                    localC[i*block + j] += a * localB[k*block + j];
                }
            }
        }

        MPI_Cart_shift(Aditi, 1, -1, &src, &dst);
        MPI_Sendrecv_replace(localA, blockSize, MPI_INT, dst, 333, src, 333, Aditi, MPI_STATUS_IGNORE);

        MPI_Cart_shift(Aditi, 0, -1, &src, &dst);
        MPI_Sendrecv_replace(localB, blockSize, MPI_INT, dst, 444, src, 444, Aditi, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(Aditi);
    for (int r = 0; r < world_size; ++r) {
        if (r == newRank) {
            cout << "local C: rank " << newRank << " coords(" << prow << "," << pcol << ")\n";
            for (int i = 0; i < block; ++i) {
                for (int j = 0; j < block; ++j) {
                    cout << localC[i*block + j] << " ";
                }
                cout << "\n";
            }
            cout << flush;
        }
        MPI_Barrier(Aditi);
    }

    if (newRank == 0) {
        for (int i = 0; i < block; ++i)
            for (int j = 0; j < block; ++j)
                C[i * N + j] = localC[i*block + j];

        for (int r = 1; r < world_size; ++r) {
            vector<int> tmp(blockSize);
            MPI_Recv(tmp.data(), blockSize, MPI_INT, r, 555, Aditi, MPI_STATUS_IGNORE);
            int rc[2];
            MPI_Cart_coords(Aditi, r, 2, rc);
            int brow = rc[0], bcol = rc[1];
            for (int i = 0; i < block; ++i)
                for (int j = 0; j < block; ++j) {
                    int gi = brow * block + i;
                    int gj = bcol * block + j;
                    C[gi * N + gj] = tmp[i*block + j];
                }
        }

        FILE *fout = fopen(argv[2], "w");
        if (!fout) { perror("fopen output"); MPI_Abort(MPI_COMM_WORLD, 1); }
        fprintf(fout, "%d\n", N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                fprintf(fout, "%d ", C[i*N + j]);
            fprintf(fout, "\n");
        }
        fclose(fout);
    } else {
        MPI_Send(localC, blockSize, MPI_INT, 0, 555, Aditi);
    }

    free(localA); free(localB); free(localC);
    if (newRank == 0) {
        free(A); free(B); free(C);
    }

    MPI_Comm_free(&Aditi);
    MPI_Finalize();
    return 0;
}
