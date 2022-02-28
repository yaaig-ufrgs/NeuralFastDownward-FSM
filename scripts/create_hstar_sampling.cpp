/**
 * Replace the h-values from a sample file with the h* value.
 * 
 * Usage: g++ -o create_hstar_sampling create_hstar_sampling.cpp
 *        ./create_hstar_sampling hstar_file sample_file
 * 
 *   e.g. 
 */

#include <iostream>
#include <fstream>
#include <limits.h>

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

int main(int argc, char *argv[]) {
    // create h* trie
    trie::trie<int> trie;
    for (pair<int,string>& p : read_samples(argv[1]))
        trie.insert(str2vec(p.second), p.first);

    // for each sample... replace h^sampling by h*
    for (pair<int,string>& p : read_samples(argv[2])) {

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
        // assert(hstar != INT_MAX);

        cout << hstar << SEP << p.second << endl;
    }

    return 0;
}
