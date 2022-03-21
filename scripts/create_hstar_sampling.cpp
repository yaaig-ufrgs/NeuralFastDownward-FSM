/**
 * Create the h* samples.
 * 
 * Usage: g++ -o create_hstar_sampling create_hstar_sampling.cpp
 *        ./create_hstar_sampling sample-state_space sample-undefined_char sample-complete [sample-complete_no_mutex]
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
        v.push_back(b == '*' ? -1 : (int)b - '0');
    return v;
}

int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 5) {
        cout << "Usage: ./create_hstar_sampling sample-state_space sample-undefined_char sample-complete [sample-complete_no_mutex]" << endl;
        exit(1);
    }

    string filename_statespace = argv[1];
    string filename_undefinedchar = argv[2];
    string filename_complete = argv[3];
    string filename_completenomutex = argc == 5 ? argv[4] : "";

    string filename_hstar = filename_complete;
    size_t idx = filename_hstar.find("_fs_");
    assert(idx != std::string::npos);
    filename_hstar.replace(filename_hstar.begin()+idx, filename_hstar.begin()+idx+4, "_hstar_");
    string filename_vs = filename_complete;
    filename_vs.replace(filename_vs.begin()+idx, filename_vs.begin()+idx+4, "_vs_");
    string filename_hstarnomutex = filename_completenomutex;
    idx = filename_hstarnomutex.find("_fs-nomutex_");
    if (idx != std::string::npos)
        filename_hstarnomutex.replace(filename_hstarnomutex.begin()+idx, filename_hstarnomutex.begin()+idx+12, "_hstar-nomutex_");

    /* creating h* and h*-nomutex */

    // create h* trie
    trie::trie<int> trie;
    vector<pair<int,string>> samples_statespace = read_samples(filename_statespace);
    for (pair<int,string>& p : samples_statespace)
        trie.insert(str2vec(p.second), p.first);

    ofstream file_hstar, file_hstarnomutex, file_vs;
    vector<pair<int,string>> samples_complete, samples_completenomutex;
    file_hstar.open(filename_hstar);
    samples_complete = read_samples(filename_complete);
    int num_samples = samples_complete.size();
    if (filename_completenomutex != "") {
        file_hstarnomutex.open(filename_hstarnomutex);
        samples_completenomutex = read_samples(filename_completenomutex);
        assert(samples_completenomutex.size() == num_samples);
    }

    vector<pair<int,string>> samples_undefinedchar = read_samples(filename_undefinedchar);
    assert(samples_undefinedchar.size() == num_samples);
    for (int i = 0; i < num_samples; i++) {
        int hstar = INT_MAX;
        for (int& hs: trie.find_all_compatible(str2vec(samples_undefinedchar[i].second), "v_vu"))
            hstar = min(hstar, hs);
        assert(hstar != INT_MAX);
        file_hstar << hstar << SEP << samples_complete[i].second << endl;
        if (filename_completenomutex != "")
            file_hstarnomutex << hstar << SEP << samples_completenomutex[i].second << endl;
    }

    file_hstar.close();
    cout << filename_hstar << " DONE." << endl;
    if (filename_completenomutex != "") {
        file_hstarnomutex.close();
        cout << filename_hstarnomutex << " DONE." << endl;
    }

    /* creating vs */

    // create h* idx-based trie
    trie::trie<int> trie_idxbased;
    for (int i = 0; i < samples_statespace.size(); i++)
        trie.insert(str2vec(samples_statespace[i].second), i);
    file_vs.open(filename_vs);
    for (int i = 0; i < num_samples; i++) {
        pair<int,string> vs = make_pair(INT_MAX, "");
        for (int& idx: trie.find_all_compatible(str2vec(samples_undefinedchar[i].second), "v_vu")) {
            if (vs.first > samples_statespace[idx].first)
                vs = samples_statespace[idx];
        }
        assert(vs.first != INT_MAX);
        file_vs << vs.first << SEP << vs.second << endl;
    }
    file_vs.close();
    cout << filename_vs << " DONE." << endl;

    return 0;
}
