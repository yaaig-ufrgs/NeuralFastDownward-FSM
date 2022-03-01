/**
 * Replace the h-values from a sample file with the h* value.
 * 
 * Usage: g++ -o create_hstar_sampling create_hstar_sampling.cpp
 *        ./create_hstar_sampling fs hstar_100pct_file sample_file
 *        ./create_hstar_sampling fs-nomutex hstar_fs_file sample_file
 *        ./create_hstar_sampling vs hstar_100pct_file sample_file
 */

#include <iostream>
#include <fstream>
#include <limits.h>
#include <string.h>

#include "../src/search/trie/trie.h"

#define SEP ';'

using namespace std;

vector<pair<int,string>> read_samples(string samples_file) {
    vector<pair<int,string>> samples;
    string h_sample;
    ifstream f(samples_file);
    while (getline(f, h_sample)) {
        if (h_sample[0] == '#')
            continue;
        int h = stoi(h_sample.substr(0, h_sample.find(SEP)));
        string s = h_sample.substr(h_sample.find(SEP) + 1, h_sample.size());
        samples.push_back(make_pair(h, s));
    }
    f.close();
    return samples;
}

vector<int> str2vec(string s) {
    vector<int> v;
    for (char& b : s)
        v.push_back((int)b - '0');
    return v;
}

void fullstate(char* hstar_file, char* samples_file) {
    // create h* trie
    trie::trie<int> trie;
    for (pair<int,string>& p : read_samples(hstar_file))
        trie.insert(str2vec(p.second), p.first);

    // for each sample... replace h^sampling by h*
    for (pair<int,string>& p : read_samples(samples_file)) {

        int hstar = INT_MAX;
        for (int& hs: trie.find_all_compatible(str2vec(p.second), true))
            hstar = min(hstar, hs);
        
        if (hstar == INT_MAX) {
            // Exceptions for blocks 7:
            // blocks 7 has some non-valid states (all goal states)
            assert(
                p.second == "0100000000000000000000100010000000000100001000000000010000001000" ||
                p.second == "0000000101000001000000100010000000000100001000000000010000001000" ||
                p.second == "0000000101000000000000100010000000000100001000000000010000001000" ||
                p.second == "0000000100000000000000100010000000000100001000000000010000001000" ||
                p.second == "0100000000000001000000100010000000000100001000000000010000001000" ||
                p.second == "0000000100000001000000100010000000000100001000000000010000001000"
            );
            hstar = 0;
            // TODO: to use in other domains, modify to accept sample_file with state_representation=values
        }
        assert(hstar != INT_MAX);

        cout << hstar << SEP << p.second << endl;
    }
}

void fullstate_nomutex(char* hstar_file, char* samples_file) {
    vector<pair<int,string>> hstar_samples = read_samples(hstar_file);
    vector<pair<int,string>> nomutex_samples = read_samples(samples_file);
    assert(hstar_samples.size() == nomutex_samples.size());
    for (int i = 0; i < hstar_samples.size(); i++)
        cout << hstar_samples[i].first << SEP << nomutex_samples[i].second << endl;
}

void validstate(char* hstar_file, char* samples_file) {
    // create h* trie
    trie::trie<int> trie;
    vector<pair<int,string>> hstar_samples = read_samples(hstar_file);
    for (int i = 0; i < hstar_samples.size(); i++)
        trie.insert(str2vec(hstar_samples[i].second), i);

    // for each sample... replace h^sampling by h*
    for (pair<int,string>& p : read_samples(samples_file)) {
        pair<int,string> vs = make_pair(INT_MAX, "");
        for (int& idx: trie.find_all_compatible(str2vec(p.second), true)) {
            if (vs.first > hstar_samples[idx].first)
                vs = hstar_samples[idx];
        }
        
        if (vs.first == INT_MAX) {
            // Exceptions for blocks 7:
            // blocks 7 has some non-valid states (all goal states)
            assert(
                p.second == "0100000000000000000000100010000000000100001000000000010000001000" ||
                p.second == "0000000101000001000000100010000000000100001000000000010000001000" ||
                p.second == "0000000101000000000000100010000000000100001000000000010000001000" ||
                p.second == "0000000100000000000000100010000000000100001000000000010000001000" ||
                p.second == "0100000000000001000000100010000000000100001000000000010000001000" ||
                p.second == "0000000100000001000000100010000000000100001000000000010000001000"
            );
            vs = make_pair(0, "0000000101000001000000100010000000000100001000000000010000001000");
            // TODO: to use in other domains, modify to accept sample_file with state_representation=values
        }
        assert(vs.first != INT_MAX);

        cout << vs.first << SEP << vs.second << endl;
    }
}

int main(int argc, char *argv[]) {
    if (strcmp(argv[1], "fs") == 0)
        fullstate(argv[2], argv[3]);
    else if (strcmp(argv[1], "fs-nomutex") == 0)
        fullstate_nomutex(argv[2], argv[3]);
    else if (strcmp(argv[1], "vs") == 0)
        validstate(argv[2], argv[3]);
    else
        cout << "Usage: ./create_hstar_sampling fs hstar_100pct_file sample_file"
             << "       ./create_hstar_sampling fs-nomutex hstar_fs_file sample_file"
             << "       ./create_hstar_sampling vs hstar_100pct_file sample_file" << endl;

    return 0;
}
