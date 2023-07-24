#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <cstring>
#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>
#define SEND 0
#define RECEIVE 1

int odd_even_sort(int move, int src, int dest, float** data, int local_size, int neighbor_size, int round){
    int swap=0;
    double comm_start, comm_end;
    float *new_data = (float*) malloc(sizeof(float)*local_size);
    float *neighbor_data = (float*) malloc(sizeof(float)*neighbor_size);
    if(move==SEND){ //left side
        // comm_start=MPI_Wtime();
        // MPI_Send(*data, local_size, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        // MPI_Recv(neighbor_data, neighbor_size, MPI_FLOAT, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(*data, local_size, MPI_FLOAT, dest, 1, neighbor_data, neighbor_size, MPI_FLOAT, dest, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);   
        // comm_end=MPI_Wtime();
        // comm_time+=(comm_end-comm_start);
        if((*data)[local_size-1]>neighbor_data[0]){
            int neighbor_id=0, self_id=0;
            for(int i=0; i<local_size; i++){ //找左半邊的data
                if(self_id<local_size && neighbor_id<neighbor_size){
                    if((*data)[self_id]<neighbor_data[neighbor_id]){
                        new_data[i]=(*data)[self_id];
                        self_id++;
                    }
                    else{
                        new_data[i]=neighbor_data[neighbor_id];
                        neighbor_id++;
                    }
                }
                else if (self_id<local_size){
                    new_data[i]=(*data)[self_id];
                    self_id++;
                }
                else{
                    new_data[i]=neighbor_data[neighbor_id];
                    neighbor_id++;
                }     
            }
            free(*data);
            *data=new_data;
        }        
    }
    else if(move==RECEIVE){ //right side
        // comm_start=MPI_Wtime();
        // MPI_Recv(neighbor_data, neighbor_size, MPI_FLOAT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // MPI_Send((*data), local_size, MPI_FLOAT, src, 0, MPI_COMM_WORLD); 
        MPI_Sendrecv(*data, local_size, MPI_FLOAT, src, 1, neighbor_data, neighbor_size, MPI_FLOAT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);      
        // comm_end=MPI_Wtime();
        // comm_time+=(comm_end-comm_start);
        if((*data)[0]<neighbor_data[neighbor_size-1]){
            swap=1;
            int neighbor_id=neighbor_size-1, self_id=local_size-1;
            for(int i=local_size-1; i>=0; i--){ //找右半邊的data
                if(self_id>=0 && neighbor_id>=0){
                    if((*data)[self_id]>neighbor_data[neighbor_id]){
                        new_data[i]=(*data)[self_id];
                        self_id--;
                    }
                    else{
                        new_data[i]=neighbor_data[neighbor_id];
                        neighbor_id--;
                    }
                }
                else if (self_id>=0){
                    new_data[i]=(*data)[self_id];
                    self_id--;
                }
                else{
                    new_data[i]=neighbor_data[neighbor_id];
                    neighbor_id--;
                }           
            }
            free(*data);
            *data=new_data;
        }       
    } 
    free(neighbor_data);
    return swap;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    double read_start, read_end, write_start, write_end, computing_start, computing_end;
    int rank, size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];
    MPI_File input_file, output_file;    
    
    // computing_start = MPI_Wtime();

    int offset=0;
    int remaining=n%size;
    int local_size=n/size;
    for(int i=0; i<rank; i++){
        if(i<remaining) offset+=(local_size+1);
        else offset+=local_size;
    }
    offset = offset*sizeof(float);

    int right_size, left_size;
    if(rank+1<remaining) right_size=local_size+1;
    else right_size=local_size;
    if(rank-1<remaining) left_size=local_size+1;
    else left_size=local_size;
    if(rank<remaining) local_size+=1;

    float *data = (float*) malloc(sizeof(float)*local_size);


    // read_start = MPI_Wtime();
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, offset, data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE); 
    MPI_File_close(&input_file);
    // read_end = MPI_Wtime();
    
    
    // computing_start = MPI_Wtime();
    boost::sort::spreadsort::spreadsort(data, data+local_size); 
    int swap=0, flag=1, sum=0;
    for(int p=0; p<=size; p++){
        if( (p%2==0 && rank%2==0) || (p%2==1 && rank%2==1)){  //send data           
            if(rank!=size-1)
                swap = odd_even_sort(SEND, rank, rank+1, &data, local_size, right_size, p);
            else swap=0;
        }
        else{ //receive data
            if(rank!=0)
                swap = odd_even_sort(RECEIVE, rank-1, rank, &data, local_size, left_size, p);
            else swap=0;
        }

        sum=0;
        MPI_Allreduce(&swap, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (sum==0 && !flag) break;
        flag=0;
    }
    // computing_end = MPI_Wtime();

    // write_start=MPI_Wtime();
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, offset, data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);
    // write_end=MPI_Wtime();
    // computing_end = MPI_Wtime();
    
    // printf("IO took %f seconds\n",(read_end-read_start)+(write_end-write_start));
    // printf("computing took %f seconds\n",computing_end-computing_start);
    // printf("communication took %f seconds\n",comm_time);

    MPI_Finalize();
    return 0;
}


//send recv
//merge sort
//early stop
//barrier
//pointer
//boost::sort::spreadsort::spreadsort