#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>


int main(int argc, char** argv) {
    
    int mpi_rank, mpi_ranks;
    unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
    unsigned long long global_sum;
	unsigned long long pixels = 0;
    // unsigned long long y;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);


    #pragma omp parallel for schedule(guided, 20)hy reduction(+:pixels)
    for (unsigned long long x = mpi_rank; x < r; x += mpi_ranks){
		pixels += ceil(sqrtl(r*r - x*x));
		// pixels += y;
	}
    pixels %= k;

    MPI_Reduce(&pixels, &global_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    if(mpi_rank==0) printf("%llu\n", (4 * global_sum)%k );
    return 0;
}
