#include <map>
#include <queue>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <string>

using namespace std;

typedef pair<int, string> Record;

int Partition(string K, int num) {
    int ret = K[0] - 'A';
    return ret % num;
}

map<int, string> Input_split(int id, string input_filename, int chunk_size)
{
    map<int, string> record;
    ifstream inputFile(input_filename);
    int count = 0;
    string L;
    while (count != (id-1)*chunk_size && getline(inputFile, L)) 
        count++;  
    while (count != (id)*chunk_size && getline(inputFile, L)) {
        record[count] = L;
        count++;
    }
    return record;
}

map<string, int> Map(Record record)
{
    size_t pos;
    string word;
    map<string, int> result; 
    while ((pos = record.second.find(" ")) != string::npos) { 
        word = record.second.substr(0, pos);

        if (result.count(word) != 0)
            result[word]++;
        else 
            result[word] = 1;
            

        record.second.erase(0, pos + 1);
    }
    if (result.count(record.second) == 0)
        result[record.second] = 1;
    else 
        result[record.second]++;
    return result;
}

