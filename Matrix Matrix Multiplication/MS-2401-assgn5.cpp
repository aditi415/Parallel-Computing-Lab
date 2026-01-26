#include<iostream>
#include<mpi.h>

#include<vector>
#include<fstream>
using namespace std;

int main(int argc,char *argv[])
{
    MPI_Init(&argc,&argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int m,n,p;
    vector<int> matrix_A,matrix_B;

    if(rank==0){
        ifstream fin("input.txt");
        if(!fin.is_open()){
            cerr<<"Error opening input.txt"<<endl;
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        fin>>m>>n>>p;
        matrix_A.resize(m*n);
        matrix_B.resize(n*p);
        for(int i=0;i<m*n;i++)
            fin>>matrix_A[i];
            for(int i=0;i<n*p;i++)
              fin>>matrix_B[i];
      fin.close();
    }

    MPI_Bcast(&m,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&p,1,MPI_INT,0,MPI_COMM_WORLD);

    if(m%size!=0){
        if(rank==0)
            cerr<<"Error: rows of matrix A not divisible by processors"<<endl;
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    int row=m/size;
    vector<int> local_A(row*n,0);
    if(rank!=0)
        matrix_B.resize(n*p);

    MPI_Scatter(matrix_A.data(),row*n,MPI_INT,local_A.data(),row*n,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(matrix_B.data(),n*p,MPI_INT,0,MPI_COMM_WORLD);

    vector<int> local_C(row*p,0);
    for(int i=0;i<row;i++){
        for(int j=0;j<p;j++){
            int sum=0;
            for(int k=0;k<n;k++)
                sum+=local_A[i*n+k]*matrix_B[k*p+j];
            local_C[i*p+j]=sum;
        }
    }

    vector<int> matrix_C;
    if(rank==0)
        matrix_C.resize(m*p);
    MPI_Gather(local_C.data(),row*p,MPI_INT,matrix_C.data(),row*p,MPI_INT,0,MPI_COMM_WORLD);

    if(rank==0){
        ofstream fout("output.txt");
        fout<<"Resultant Matrix C ("<<m<<"x"<<p<<"):\n";
        for(int i=0;i<m;i++){
            for(int j=0;j<p;j++)
                fout<<matrix_C[i*p+j]<<" ";
            fout<<"\n";
        }
        fout.close();
        cout<<"Matrix multiplication completed. Result stored in output.txt"<<endl;
    }

    MPI_Finalize();
    return 0;
}

