#include <iostream>
#include <mpi.h>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;  

int partition(vector<int> &arr, int start, int end)
{
    int pivot = arr[end];
    int i = start - 1;
    for (int j = start; j < end; j++) 
    {
        if (arr[j] <= pivot)
        {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[end]);
    return i + 1;
}

void quicksort(vector<int> &arr, int start, int end) 
{
    if (start < end)
    {
        int pi = partition(arr, start, end);
        quicksort(arr, start, pi - 1);
        quicksort(arr, pi + 1, end);
    }
}

int main(int argc, char *argv[]) 
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    vector<int> arr;

    if (rank == 0) 
    {
        ifstream fin("input.txt");
        if (!fin.is_open()) 
        {
            cerr << "Error: Cannot open input.txt" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fin >> n;
        arr.resize(n);
        for (int i = 0; i < n; i++) 
          fin >> arr[i];
        fin.close();
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = n / size;
    vector<int> local_data(local_n);
    
    MPI_Scatter(arr.data(), local_n, MPI_INT,local_data.data(), local_n, MPI_INT,0, MPI_COMM_WORLD);

    quicksort(local_data, 0, local_n - 1);

    MPI_Gather(local_data.data(), local_n, MPI_INT,arr.data(), local_n, MPI_INT,0, MPI_COMM_WORLD);

    if (rank == 0) 
    {
        sort(arr.begin(), arr.end());

        ofstream fout("output.txt");
        fout << "Sorted elements:\n";
        for (int i = 0; i < n; i++) fout << arr[i] << " ";
        fout << endl;
        fout.close();

        cout << " Sorting completed successfully.\nOutput written to output.txt" << endl;
    }

    MPI_Finalize();
    return 0;
}

