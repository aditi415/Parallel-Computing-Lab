#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
using namespace std;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    vector<double> A, b;

    if (rank == 0) {
        ifstream fin("input.txt");
        fin >> n;

        A.resize(n * n);
        b.resize(n);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                fin >> A[i*n + j];

        for (int i = 0; i < n; i++)
            fin >> b[i];
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        A.resize(n*n);
        b.resize(n);
    }

    MPI_Bcast(A.data(), n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int k = 0; k < n-1; k++) {
        if (fabs(A[k*n + k]) < 1e-12) {
            if (rank == 0)
                cout << "Zero pivot encountered.
";
            MPI_Finalize();
            return 0;
        }

        for (int i = k+1; i < n; i++) {
            if (i % size == rank) {
                double m = A[i*n + k] / A[k*n + k];
                for (int j = k; j < n; j++)
                    A[i*n + j] -= m * A[k*n + j];
                b[i] -= m * b[k];
            }
        }

        MPI_Bcast(A.data(), n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    vector<double> x(n);
    if (rank == 0) {
        for (int i = n-1; i >= 0; i--) {
            double sum = b[i];
            for (int j = i+1; j < n; j++)
                sum -= A[i*n + j] * x[j];
            x[i] = sum / A[i*n + i];
        }

        ofstream fout("output.txt");
        fout << "Solution:
";
        for (int i = 0; i < n; i++)
            fout << x[i] << "
";
    }

    MPI_Finalize();
    return 0;
}
