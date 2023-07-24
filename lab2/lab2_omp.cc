#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
    int omp_threads, omp_thread;
    unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long global_sum;
	unsigned long long pixels = 0;
    
    #pragma omp parallel for schedule(static, 20) reduction(+:pixels)
    for (unsigned long long x=0; x<r; x++){
        unsigned long long y = ceil(sqrtl( r*r-x*x ));
        pixels += y;
    }
    
    pixels %= k;
    printf("%llu\n", (4 * pixels)%k );

    return 0;
}
