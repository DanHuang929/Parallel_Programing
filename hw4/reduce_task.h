#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <algorithm>

using namespace std;

typedef pair<string, int> Item;

bool ascend = true;

static bool comp(Item a, Item b) {
    if (ascend == true)
        return (a.first < b.first);
    else 
        return (a.first > b.first);
}

struct classcomp {
    bool operator() (const string& L, const string& R) const {
        
        if (ascend == true)
            return L < R;   
        else 
            return L > R;
    }
};

vector<Item> Sort(vector<Item> record)
{
    sort(record.begin(), record.end(), comp);
    return record;
}

map<string, vector<int>, classcomp> Group(vector<Item> data)
{
    map<string, vector<int>, classcomp> grouped;
    for (auto it: data) 
        grouped[it.first].push_back(it.second);
    return grouped;
}

Item Reduce(string key, vector<int> data)
{
    int sum = 0;
    for (auto it: data)
        sum += it;
    return make_pair(key, sum);
}

void Output(vector<Item> reduce_result, string output_dir, string job_name, int id)
{
    ofstream out(output_dir + job_name + "-" + to_string(id) + ".out");
    for (auto it: reduce_result) {
        out << it.first << " " << it.second << endl;
    }
}