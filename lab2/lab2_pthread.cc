#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>

typedef struct Data{
    int id;
    unsigned long long r;
	unsigned long long k;
    unsigned long long ncpus;
}data;
unsigned long long global_sum;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* hello(void* arg) {
	unsigned long long pixels = 0;
    data *d;
    d = (data*)arg;
    
    for (unsigned long long x = d->id; x < d->r; x += d->ncpus){
		unsigned long long y = ceil(sqrtl( d->r*d->r - x*x ));
		pixels += y;
		
	}
	pixels %= d->k;

    pthread_mutex_lock(&mutex);
	global_sum += pixels;
	pthread_mutex_unlock(&mutex);

    pthread_exit(NULL);
}


int main(int argc, char** argv) {
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);
    pthread_t threads[ncpus];

	pthread_mutex_init(&mutex, 0);

    int rc;
    data ID[ncpus];
    int t;
    for (t = 0; t < ncpus; t++) {
        ID[t].id = t;
        ID[t].k = k;
        ID[t].r = r;
        ID[t].ncpus = ncpus;

        rc = pthread_create(&threads[t], NULL, hello, &ID[t]);
    }

    for (t=0; t<ncpus; t++) {
		pthread_join(threads[t], NULL);
	}

    printf("%llu\n", (4 * global_sum) % k);

    pthread_mutex_destroy(&mutex);
	pthread_exit(NULL);
}

