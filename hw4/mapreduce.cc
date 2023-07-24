#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <deque>
#include <queue>
#include <unistd.h>
#include <queue>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <pthread.h>
#include <mpi.h>
#include <omp.h>
#include "map_task.h"
#include "reduce_task.h"
#include <chrono>

using namespace std;

queue<pair<int, int>> tasks;    
pthread_mutex_t mutex;
pthread_cond_t cond;
queue<pair<int, pair<int, int>>> complete;  
pthread_mutex_t mutex_complete;
pthread_cond_t cond_complete;

int Size, rank_ID, num_threads;
int num_jobs, num_workers, master_node;  
int num_reducer, D, chunk_size, job_size;
string job_name, input_filename, locality_config_filename, output_dir; 

int calc_time(struct timespec start_time, struct timespec end_time)
{
    struct timespec temp;
    if ((end_time.tv_nsec - start_time.tv_nsec) < 0) {
        temp.tv_sec = end_time.tv_sec-start_time.tv_sec-1;
        temp.tv_nsec = 1000000000 + end_time.tv_nsec - start_time.tv_nsec;
    } else {
        temp.tv_sec = end_time.tv_sec - start_time.tv_sec;
        temp.tv_nsec = end_time.tv_nsec - start_time.tv_nsec;
    }
    double exe_time = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    return (int)(exe_time+0.5);
}

void* pool(void* args) 
{
    struct timespec start_time, end_time, temp;
    double exe_time;

    while (true) {
        pair<int, int> task;

        pthread_mutex_lock(&mutex);
        while (tasks.empty()) { 
            pthread_cond_wait(&cond, &mutex);
        }
        task = tasks.front();
        tasks.pop();
        pthread_mutex_unlock(&mutex);

        clock_gettime(CLOCK_MONOTONIC, &start_time);

        if (task.second != rank_ID) {
            sleep(D);
        }


        map<int, string> records = Input_split(task.first, input_filename, chunk_size);

        map<string, int> map_output;
        for (auto record: records) {
            map<string, int> map_result = Map(record);
            for (auto it: map_result) {
                if (map_output.count(it.first) == 0)
                    map_output[it.first] = it.second;
                else 
                    map_output[it.first] += it.second;
            }
        }


        ofstream out(output_dir + job_name + "-intermediate" + to_string(task.first) + ".out");
        for (auto it: map_output) {
            out << it.first << " " << it.second << endl;
        }

        clock_gettime(CLOCK_MONOTONIC, &end_time);

        pthread_mutex_lock(&mutex_complete);

        complete.push(make_pair(task.first, make_pair(calc_time(start_time, end_time), map_output.size())));
        num_jobs--;
        pthread_mutex_unlock(&mutex_complete);
        pthread_cond_signal(&cond_complete);
    }
}



void tasktracker()
{
    //map
    pthread_t threads[num_threads-1];
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
    pthread_mutex_init(&mutex_complete, NULL);
    pthread_cond_init(&cond_complete, NULL);
    for (int i = 0; i < num_threads-1; i++) { //create threads
        pthread_create(&threads[i], NULL, &pool, NULL);
    }

    num_jobs = 0;

    int job[2] = {0};
    //request jobs
    while (true) {
        pthread_mutex_lock(&mutex_complete);
        while (num_jobs == num_threads-1) { //wait if every threads have jobs
            pthread_cond_wait(&cond_complete, &mutex_complete);
        }
        pthread_mutex_unlock(&mutex_complete);

        MPI_Send(&rank_ID, 1, MPI_INT, master_node, 0, MPI_COMM_WORLD); //send request to jobtracker
        MPI_Recv(&job, 2, MPI_INT, master_node, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //get job from jobtracker
        if (job[0] == -1) { // if there is no job, break
            break;
        }

        pthread_mutex_lock(&mutex_complete);
        num_jobs++; 
        pthread_mutex_unlock(&mutex_complete);

        //put task into task queue
        pthread_mutex_lock(&mutex);
        tasks.push(make_pair(job[0], job[1]));
        pthread_mutex_unlock(&mutex);
        pthread_cond_signal(&cond);


        usleep(200);
    }


    int finish_message[3];
    while (true) {
        pthread_mutex_lock(&mutex_complete);
        if (complete.empty() && num_jobs == 0) {    // if all jobs are finished and all information is sent
            pthread_mutex_unlock(&mutex_complete);
            break;
        }
        else if (!complete.empty()) {   // send complete info to jobtracker
            pair<int, pair<int, int>> info = complete.front();
            complete.pop();
            pthread_mutex_unlock(&mutex_complete);
            finish_message[0] = info.first;
            finish_message[1] = info.second.first;
            finish_message[2] = info.second.second;
            MPI_Send(&finish_message, 3, MPI_INT, master_node, 2, MPI_COMM_WORLD);
        }
        else { 
            //there still have jobs
            pthread_mutex_unlock(&mutex_complete);
        }
    }

    //reduce
    struct timespec start_time, end_time, temp;
    double exe_time;
    queue<pair<int, int>> reduce_job_time;
    while (true) {
        // request for reduce task
        MPI_Send(&rank_ID, 1, MPI_INT, master_node, 3, MPI_COMM_WORLD);
        MPI_Recv(&job, 1, MPI_INT, master_node, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (job[0] == -1) { 
            break;
        }
        
        clock_gettime(CLOCK_MONOTONIC, &start_time);

        vector<Item> data;
        string intermediate_file = output_dir + job_name + "-intermediate_reducer_" + to_string(job[0]) + ".out";
        ifstream input_file(intermediate_file);
        string line;
        while (getline(input_file, line)) {
            size_t pos = line.find(" ");
            string key = line.substr(0, pos);
            int value = stoi(line.substr(pos+1));
            data.push_back(make_pair(key, value));
        }

        data = Sort(data);

        map<string, vector<int>, classcomp> grouped_data = Group(data);

        vector<Item> reduce_result;
        for (auto it: grouped_data) {
            reduce_result.push_back(Reduce(it.first, it.second));
        }

        Output(reduce_result, output_dir, job_name, job[0]);

        clock_gettime(CLOCK_MONOTONIC, &end_time);
        reduce_job_time.push(make_pair(job[0], calc_time(start_time, end_time)));
    }

    while (!reduce_job_time.empty()) {
        pair<int, int> info = reduce_job_time.front();
        reduce_job_time.pop();
        finish_message[0] = info.first;
        finish_message[1] = info.second;
        MPI_Send(&finish_message, 2, MPI_INT, master_node, 5, MPI_COMM_WORLD);
    }
}

void jobtracker()
{
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    ofstream out(output_dir + job_name + "-log.out");
    out << time(nullptr) << ",Start_Job," << Size << "," << num_threads << "," << job_name << "," << num_reducer << "," << D << "," << input_filename << "," << chunk_size << "," << locality_config_filename << "," << output_dir << endl;

    deque<pair<int, int>> jobs;   // task queue
    int task_node, num;
    int counting_pairs = 0;
    int job_message[2], finish_message[3];    // (job, time, pairs)   
    
    //map

    ifstream input_file(locality_config_filename);
    string line;
    while (getline(input_file, line)) {
        size_t pos = line.find(" ");
        int chunkID = stoi(line.substr(0, pos));
        int nodeID = stoi(line.substr(pos+1)) % num_workers;
        jobs.push_back(make_pair(chunkID, nodeID));    // Locality information: (chunkID, nodeID)
    }
    job_size = jobs.size();

    while (!jobs.empty()) {
        MPI_Recv(&task_node, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (auto job = jobs.begin(); job != jobs.end(); job++) {
            if (job == prev(jobs.end())) {  // last node => FIFO

                job_message[0] = jobs.begin()->first;
                job_message[1] = jobs.begin()->second;

                out << time(nullptr) << ",Dispatch_MapTask," << job_message[0] << "," << task_node << endl;
                MPI_Send(&job_message, 2, MPI_INT, task_node, 1, MPI_COMM_WORLD);
                jobs.erase(jobs.begin());
                break;
            }
            else if (job->second == task_node) {  // Locality-Aware Scheduling Algorithm

                job_message[0] = job->first;
                job_message[1] = job->second;

                out << time(nullptr) << ",Dispatch_MapTask," << job_message[0] << "," << task_node << endl;
                MPI_Send(&job_message, 2, MPI_INT, task_node, 1, MPI_COMM_WORLD);
                jobs.erase(job);
                break;
            }
        }
    }

    // notify workers that all jobs are dispatched
    num = 0;
    job_message[0] = -1;
    job_message[1] = -1;
    while (num != num_workers) {{ //until every Size finish its' task
        MPI_Recv(&task_node, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        num++;
        MPI_Send(&job_message, 2, MPI_INT, task_node, 1, MPI_COMM_WORLD); // notify the node that there is no job
    }}

    // receive complete information
    num = 0;
    // cout<<"job size: "<<jobs.size()<<endl;
    while (num != job_size) {{
        MPI_Recv(&finish_message, 3, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        num++;
        counting_pairs += finish_message[2];
        out << time(nullptr) << ",Complete_MapTask," << finish_message[0] << "," << finish_message[1] << endl;
    }}

    // cout<<counting_pairs<<endl;
    out << time(nullptr) << ",Start_Shuffle," << to_string(counting_pairs) << endl;

    struct timespec start_shuffle_time, end_shuffle_time;
    clock_gettime(CLOCK_MONOTONIC, &start_shuffle_time);

    ofstream *files = new ofstream[num_reducer];
    for (int i = 0; i < num_reducer; i++) {
        files[i] = ofstream(output_dir + job_name + "-intermediate_reducer_" + to_string(i+1) + ".out");
    }
    vector<Item> data;
    for (int i = 0; i < job_size; i++) {
        ifstream input_file(output_dir + job_name + "-intermediate" + to_string(i+1) + ".out");
        string line;
        data.clear();
        while (getline(input_file, line)) {
            size_t pos = line.find(" ");
            string key = line.substr(0, pos);
            int value = stoi(line.substr(pos+1));
            data.push_back(make_pair(key, value));
        }
        for (auto it: data) {
            int idx = Partition(it.first, num_reducer);
            files[idx] << it.first << " " << it.second << endl;
        }
    }
    delete [] files;
    
    clock_gettime(CLOCK_MONOTONIC, &end_shuffle_time);
    out << time(nullptr) << ",Finish_Shuffle," << to_string(calc_time(start_shuffle_time, end_shuffle_time)) << endl;

    //reduce
    num = num_reducer;
    while (num > 0) {
        MPI_Recv(&task_node, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //send reduce task to tasktracker
        job_message[0] = num;
        num--;
        out << time(nullptr) << ",Dispatch_ReduceTask," << job_message[0] << "," << task_node << endl;
        MPI_Send(&job_message, 1, MPI_INT, task_node, 4, MPI_COMM_WORLD);
        
    }

    // notify workers that all jobs are dispatched
    num = 0;
    job_message[0] = -1;
    while (num != num_workers) {{
        MPI_Recv(&task_node, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&job_message, 1, MPI_INT, task_node, 4, MPI_COMM_WORLD);
        num++;
    }}

    // receive complete information
    num = 0;
    while (num != num_reducer) {{
        MPI_Recv(&finish_message, 2, MPI_INT, MPI_ANY_SOURCE, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        out << time(nullptr) << ",Complete_ReduceTask," << finish_message[0] << "," << finish_message[1] << endl;
        num++;
    }}

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    out << time(nullptr) << ",Finish_Job," << to_string(calc_time(start_time, end_time)) << endl;
}

int main(int argc, char **argv)
{
    job_name = string(argv[1]);
    num_reducer = stoi(argv[2]);
    D = stoi(argv[3]);
    input_filename = string(argv[4]);
    chunk_size = stoi(argv[5]);
    locality_config_filename = string(argv[6]);
    output_dir = string(argv[7]);


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &Size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_ID);


    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_threads = CPU_COUNT(&cpu_set);

    num_workers = Size-1;
    master_node = 0;
    // start MapReduce
    // auto compute_start = std::chrono::steady_clock::now();
    
    if (rank_ID == master_node) {
        jobtracker();
    }
    else {
        tasktracker();
    }
    // auto compute_end = std::chrono::steady_clock::now();
    // std::cout << std::chrono::duration_cast<std::chrono::microseconds>(compute_end-compute_start).count() << " microseconds" << std::endl;


    MPI_Finalize();
    return 0;
}