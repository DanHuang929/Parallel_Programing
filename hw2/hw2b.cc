#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <emmintrin.h>
#include <mpi.h>
#include <omp.h>


void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}


double y0base, x0base;
int* local_image;
int iters ;
double left;
double right;
double lower;
double upper;
int width;
int height;

int finished_row=0, cur_row;

void cal_pixel(int cur_row){
    
    double y0 = cur_row * y0base + lower;
    __m128d y0_vec = _mm_set_pd(y0,y0);
    
    int end[2]={1,1};
    int repeats[2];
    int finished_col=-1, cur_col1, cur_col2;
    __m128d constant_2 = _mm_set_pd(2.0,2.0);
    __m128d x0_vec;
    __m128d x_vec, y_vec;
    __m128d length_squared_vec;
    
    while (finished_col<width-1){  //every col
        if(end[0]){
            finished_col++;
            cur_col1 = finished_col;
            x0_vec[0] = cur_col1 * x0base + left;
            x_vec[0] = 0;
            y_vec[0] = 0;
            end[0] = 0;
            repeats[0]=0;
        }
        if(end[1]){
            finished_col++;
            cur_col2 = finished_col;
            x0_vec[1] = cur_col2 * x0base + left;
            x_vec[1] = 0;
            y_vec[1] = 0;
            end[1] = 0;
            repeats[1]=0;
        }
        
            
        __m128d temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(x_vec, x_vec), _mm_mul_pd(y_vec, y_vec)), x0_vec);
        y_vec = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x_vec, constant_2), y_vec), y0_vec);
        x_vec = temp;

        length_squared_vec = _mm_add_pd(_mm_mul_pd(x_vec, x_vec),_mm_mul_pd(y_vec, y_vec));
        repeats[0]+=1;
        repeats[1]+=1;


        
        if (length_squared_vec[0] >= 4 || repeats[0] >= iters) {
            local_image[cur_row * width + cur_col1] = repeats[0];
            end[0] = 1;
        }
        if (length_squared_vec[1] >= 4 || repeats[1] >= iters) {
            local_image[cur_row * width + cur_col2] = repeats[1];
            end[1] = 1;
        }
    }

    //continue
    if(!end[0]){
        while (length_squared_vec[0] < 4 && repeats[0] < iters) {
            double temp = x_vec[0]  * x_vec[0] - y_vec[0] * y_vec[0] + x0_vec[0];
            y_vec[0] = 2 * x_vec[0] * y_vec[0] + y0;
            x_vec[0] = temp;
            length_squared_vec[0] = x_vec[0] * x_vec[0] + y_vec[0] * y_vec[0];
            ++repeats[0];
        }
        local_image[cur_row * width + cur_col1] = repeats[0];
    }
    if(!end[1]){
        while (length_squared_vec[1] < 4 && repeats[1] < iters) {
            double temp = x_vec[1] * x_vec[1] - y_vec[1] * y_vec[1] + x0_vec[1];
            y_vec[1] = 2 * x_vec[1] * y_vec[1] + y0;
            x_vec[1] = temp;
            length_squared_vec[1] = x_vec[1] * x_vec[1] + y_vec[1] * y_vec[1];
            ++repeats[1];
        }
        local_image[cur_row * width + cur_col2] = repeats[1];
    }

    return;
}


int main(int argc, char** argv) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    unsigned long long ncpus = CPU_COUNT(&cpu_set);
    /* argument parsing */
    const char* filename = argv[1];
    int rank, size;
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    y0base = ((upper - lower) / height);
    x0base = ((right - left) / width);

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* allocate memory for image */
    int image_size= width*height;
    int all_image_size = size*image_size;
    int* image = (int*)malloc(image_size * sizeof(int));
    int* all_image = (int*)malloc(all_image_size * sizeof(int));
    local_image = (int*)malloc(image_size * sizeof(int));
    

    /* mandelbrot set */
    #pragma omp parallel num_threads(ncpus)
    {
        #pragma omp for schedule(dynamic, 5)
        for(int row=rank; row<height; row+=size){
            cal_pixel(row);
        }
    }


    MPI_Reduce(local_image, image, image_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);


    
    if(rank==0){
        write_png(filename, iters, width, height, image);
        free(image);
    }

    MPI_Finalize();
    return 0;
}

