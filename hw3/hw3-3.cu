#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>
#include <chrono>


//======================
#define DEV_NO 0
#define debug 0
#define BS 64
#define half_BS BS/2
// cudaDeviceProp prop;

const int INF = ((1 << 30) - 1);
// const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);
void block_FW(int B);
int ceil(int a, int b);

__device__ void assign_to_share_mem(int share[BS][BS], int* dist, int offset, int n, int x, int y);
__device__ void assign_to_device(int share[BS][BS], int* dist, int offset, int n, int x, int y);
__global__ void cal1(int *dist, int n, int Round, int B);
__global__ void cal2(int *dist, int n, int Round, int B);
__global__ void cal3(int *dist, int n, int Round, int B, int row_offset);

int n, m, origin_n;
// static int Dist[V][V];
int* Dist = NULL;


void display(void)
{
    printf("==========\n");
    for (int i = 0; i < origin_n; ++i) {
        for (int j = 0; j < origin_n; ++j) {
            if (Dist[i*n+j] >= INF) printf(" -1 ");
            else printf("%3d ", Dist[i*n+j]);
        }
        printf("\n");
    }
    printf("==========\n");
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&origin_n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    //padding
    n = origin_n + (BS-(origin_n%BS));
    Dist = (int*) malloc(sizeof(int)*n*n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i*n+j] = 0;
            } else {
                Dist[i*n+j] = INF;
            }
        }
    }

    int* buffer = (int*)malloc(m*3*sizeof(int));
    fread(buffer, sizeof(int), m*3, file);

    #pragma omp parallel for schedule(static)
    for (int i=0; i<m; i++){
        Dist[buffer[i*3]*n+buffer[i*3+1]] = buffer[i*3+2];
    }
    

    if(debug) display();
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < origin_n; ++i) {
        for (int j = 0; j < origin_n; ++j) {
            if (Dist[i*n+j] >= INF) Dist[i*n+j] = INF;
        }
        fwrite(&Dist[i*n], sizeof(int), origin_n, outfile);
    }
    if(debug) display();
    fclose(outfile);

}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {

    int* dist[2];  //multi device
    unsigned long Size = n * n * sizeof(int);


    int round = ceil(n, BS);
    int row_offset=0;
    if((n/BS) % 2 == 0) row_offset = n/2;
    else row_offset = (n/BS/2)*BS + BS;
    // int row_offset = ((n/BS) % 2 == 0)? : (n/BS/2+1) * BS;

    #pragma omp parallel num_threads(2)
    {
        int cpu_thread_id = omp_get_thread_num();
        cudaSetDevice(cpu_thread_id);

        //each gpu have one matrix
        cudaMalloc(&dist[cpu_thread_id], Size);
        cudaMemcpy(dist[cpu_thread_id], Dist, Size, cudaMemcpyHostToDevice);


        dim3 grid_phase3(round, row_offset/BS); // num of blocks for phase3, only have to due with half matrix_size
        dim3 grid_phase2(round, 2);
        dim3 blk(32,32); //1024 threads for phase1


        for (int r = 0; r < round; ++r) {

            /* Phase 1*/
            cal1<<<1, blk>>>(dist[cpu_thread_id], n, r, BS);

            /* Phase 2*/
            cal2<<<grid_phase2, blk>>>(dist[cpu_thread_id], n, r, BS);

            // /* Phase 3*/
            cal3<<<grid_phase3, blk>>>(dist[cpu_thread_id], n, r, BS, row_offset*cpu_thread_id);

            cudaDeviceSynchronize();
            #pragma omp barrier

            if (cpu_thread_id == 1 && (r+1) < row_offset/BS) {
                cudaMemcpy(dist[1]+(r+1)*BS*n, dist[0]+(r+1)*BS*n, BS*n*sizeof(int), cudaMemcpyDeviceToDevice);
            }
            else if(cpu_thread_id == 0 && (r+1) >= row_offset/BS) {
                cudaMemcpy(dist[0]+(r+1)*BS*n, dist[1]+(r+1)*BS*n, BS*n*sizeof(int), cudaMemcpyDeviceToDevice);
            }
        }

       if (cpu_thread_id == 0) {
            cudaMemcpy(Dist, dist[0], row_offset*n*sizeof(int), cudaMemcpyDeviceToHost);
        }
        else {
            cudaMemcpy(Dist+row_offset*n, dist[1]+row_offset*n, ((n-row_offset)*n)*sizeof(int), cudaMemcpyDeviceToHost);
        }

    }
    
	// cudaFree(dist[0]);
    // cudaFree(dist[1]);
}

__device__ void assign_to_share_mem(int share[BS][BS], int* dist, int offset, int n, int x, int y)
{
    share[y][x] = dist[offset + y*n + x];
    share[y+half_BS][x] = dist[offset + (y+half_BS)*n + x];
    share[y][x+half_BS] = dist[offset + y*n + x + half_BS];
    share[y+half_BS][x+half_BS] = dist[offset + (y+half_BS)*n + x + half_BS];
}

__device__ void assign_to_device(int share[BS][BS], int* dist, int offset, int n, int x, int y)
{
    dist[offset + y*n + x] = share[y][x];
    dist[offset + (y+half_BS)*n + x] = share[y+half_BS][x];
    dist[offset + y*n + x + half_BS] = share[y][x+half_BS];
    dist[offset + (y+half_BS)*n + x + half_BS] = share[y+half_BS][x+half_BS];
}

__global__ void cal1(int* dist, int n, int Round, int B)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int offset = BS*Round;

    
    //create shared memory
    __shared__ int Pivot[BS][BS];

    // every threads need to compute 4 points
    assign_to_share_mem(Pivot, dist, offset*(n+1), n, x, y);

    __syncthreads();

    for (int k=0; k<B; k++){
        Pivot[y][x] = min(Pivot[y][k] + Pivot[k][x], Pivot[y][x]);
        Pivot[y+half_BS][x] = min(Pivot[y+half_BS][k] + Pivot[k][x], Pivot[y+half_BS][x]);
        Pivot[y][x+half_BS] = min(Pivot[y][k] + Pivot[k][x+half_BS], Pivot[y][x+half_BS]);
        Pivot[y+half_BS][x+half_BS] = min(Pivot[y+half_BS][k] + Pivot[k][x+half_BS], Pivot[y+half_BS][x+half_BS]);
        __syncthreads(); 
    }
    
    assign_to_device(Pivot, dist, offset*(n+1), n, x, y);
}

__global__ void cal2(int* dist, int n, int Round, int B)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int offset = BS*Round;
    int blk_x = blockIdx.x;
    int blk_y = blockIdx.y;
    if(blk_x==Round) return;



    //create shared memory
    __shared__ int D[BS][BS];
    __shared__ int Pivot[BS][BS];

    // D depends on itself and Pivot
    if(blk_y==0){
        assign_to_share_mem(D, dist, offset*n + blk_x*BS, n, x, y);
    }
    else{
        assign_to_share_mem(D, dist, blk_x*BS*n + offset, n, x, y);
    }

    assign_to_share_mem(Pivot, dist, offset*(n+1), n, x, y);
    
    
    __syncthreads();

    
    if(blk_y==0){
        #pragma unroll 32
        for (int k=0; k<B; k++){
            D[y][x] = min(Pivot[y][k] + D[k][x], D[y][x]);
            D[y+half_BS][x] = min(Pivot[y+half_BS][k] + D[k][x], D[y+half_BS][x]);
            D[y][x+half_BS] = min(Pivot[y][k] + D[k][x+half_BS], D[y][x+half_BS]);
            D[y+half_BS][x+half_BS] = min(Pivot[y+half_BS][k] + D[k][x+half_BS], D[y+half_BS][x+half_BS]);
            __syncthreads();
        }
    }
    else{
        #pragma unroll 32
        for (int k=0; k<B; k++){
            D[y][x] = min(D[y][k] + Pivot[k][x], D[y][x]);
            D[y+half_BS][x] = min(D[y+half_BS][k] + Pivot[k][x], D[y+half_BS][x]);
            D[y][x+half_BS] = min(D[y][k] + Pivot[k][x+half_BS], D[y][x+half_BS]);
            D[y+half_BS][x+half_BS] = min(D[y+half_BS][k] + Pivot[k][x+half_BS], D[y+half_BS][x+half_BS]);
            __syncthreads();
        }
    }

    if(blk_y==0){
        assign_to_device(D, dist, offset*n + blk_x*BS, n, x, y);
    }
    else{
        assign_to_device(D, dist, blk_x*BS*n + offset, n, x, y);
    }
}

__global__ void cal3(int* dist, int n, int Round, int B, int row_offset)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int blk_x = blockIdx.x;
    int blk_y = blockIdx.y + row_offset;
    int new_blk_y = blockIdx.y*BS + row_offset;
    int offset = BS*Round;
    if(blk_x==Round || blk_y+row_offset/BS==Round) return;
    else if(new_blk_y>=n) return;

    //create shared memory
    // D depends on itself and Col&Row
    __shared__ int Col[BS][BS];
    __shared__ int Row[BS][BS];
    __shared__ int D[BS][BS];


    assign_to_share_mem(D, dist, new_blk_y*n + blk_x*BS, n, x, y);
    assign_to_share_mem(Col, dist, new_blk_y*n + offset, n, x, y);
    assign_to_share_mem(Row, dist, offset*n + blk_x*BS, n, x, y);

    __syncthreads();

    #pragma unroll 32
    for (int k=0; k<B; k++){
        D[y][x] = min(Col[y][k] + Row[k][x], D[y][x]);
        D[y+half_BS][x] = min(Col[y+half_BS][k] + Row[k][x], D[y+half_BS][x]);
        D[y][x+half_BS] = min(Col[y][k] + Row[k][x+half_BS], D[y][x+half_BS]);
        D[y+half_BS][x+half_BS] = min(Col[y+half_BS][k] + Row[k][x+half_BS], D[y+half_BS][x+half_BS]);
    }

    assign_to_device(D, dist, new_blk_y*n + blk_x*BS, n, x, y);
    
}

int main(int argc, char* argv[]) {
    input(argv[1]);
    block_FW(BS);
    output(argv[2]);
    return 0;
}

