#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>

#include <emmintrin.h> //SSE2(include xmmintrin.h)
// #include <smmintrin.h>//SSE4.1(include tmmintrin.h)
#define debug 0

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m; //vertex, edge
static int Dist[V][V];
unsigned long long ncpus;

int main(int argc, char* argv[]) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

    ncpus = CPU_COUNT(&cpu_set);
    pthread_t threads[ncpus];
    if(debug) printf("num threads:%d\n", ncpus);
    int id[ncpus];

    input(argv[1]);
    int B = 4;
    block_FW(B);
    output(argv[2]);
    return 0;
}

void display(void)
{
    printf("==========\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) printf(" -1 ");
            else printf("%3d ", Dist[i][j]);
        }
        printf("\n");
    }
    printf("==========\n");
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int* buffer = (int*)malloc(m*3*sizeof(int));
    fread(buffer, sizeof(int), m*3, file);

    // #pragma omp parallel for schedule(static)
    #pragma omp parallel num_threads(ncpus) 
    {
        #pragma omp for schedule(static)
        for (int i=0; i<m; i++){
            Dist[buffer[i*3]][buffer[i*3+1]] = buffer[i*3+2];
        }
    }

    if(debug) display();
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");

    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    if(debug) display();
    fclose(outfile);

}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {
    int round = ceil(n, B);

    for (int r = 0; r < round; ++r) {
        // printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        cal(B, r, r, r, 1, 1);

        /* Phase 2*/
        cal(B, r, r, 0, r, 1);
        cal(B, r, r, r + 1, round - r - 1, 1);
        cal(B, r, 0, r, 1, r);
        cal(B, r, r + 1, r, 1, round - r - 1);

        /* Phase 3*/
        cal(B, r, 0, 0, r, r);
        cal(B, r, 0, r + 1, round - r - 1, r);
        cal(B, r, r + 1, 0, r, round - r - 1);
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    int Round_B = Round * B;
    int Round_1_B = (Round + 1) * B;

    #pragma omp parallel num_threads(ncpus) 
    {
        #pragma omp for schedule(static)
        for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
            for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
                for (int k = Round_B; k < Round_1_B && k < n; ++k) {

                    int block_internal_start_x = b_i * B;
                    int block_internal_end_x = (b_i + 1) * B;
                    int block_internal_start_y = b_j * B;
                    int block_internal_end_y = (b_j + 1) * B;

                    if (block_internal_end_x > n) block_internal_end_x = n;
                    if (block_internal_end_y > n) block_internal_end_y = n;
                  
                    for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                        for (int j = block_internal_start_y; j < block_internal_end_y; j+=1) {
                            if (Dist[i][k] + Dist[k][j] < Dist[i][j]) {
                                Dist[i][j] = Dist[i][k] + Dist[k][j];
                            }
                        }
                    }
                }
            }
        }
    }
}
