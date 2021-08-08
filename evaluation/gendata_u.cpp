#include <algorithm>
#include <iostream>
#include "Graph.h"
#include <gflags/gflags.h>

DEFINE_string(graph, "BlogCatalog", "Graph name.");
DEFINE_double(test_ratio, 0.3, "Testing ratio.");

int main(int argc,  char **argv){
    srand((unsigned)time(0));
    google::ParseCommandLineFlags(&argc, &argv, true);

    string dataset      = "data/" + FLAGS_graph + ".txt";
    string alphafile    = "alpha/" + FLAGS_graph + "_link.txt";
    string traindataset = "lp_data/train_graph/" + FLAGS_graph + ".txt";
    string ptestdataset = "lp_data/test/positive/" + FLAGS_graph + ".txt";
    string ntestdataset = "lp_data/test/negative/" + FLAGS_graph + ".txt";

    UGraph g;
    clock_t t0 = clock();
    g.inputGraph(dataset,alphafile);
    clock_t t1 = clock();
    cout << "reading in graph takes " << (t1 - t0)/(1.0 * CLOCKS_PER_SEC) << " s." << endl;
    clock_t t2 = clock();
    g.RandomSplitGraph(dataset, ptestdataset, traindataset, FLAGS_test_ratio);
    clock_t t3 = clock();
    cout << "splitting graph takes " << (t3 - t2)/(1.0 * CLOCKS_PER_SEC) << " s." << endl;
    g.NegativeSamples(ntestdataset, FLAGS_test_ratio);
    clock_t t4 = clock();
    cout << "sampling negative edges takes " << (t4 - t3)/(1.0 * CLOCKS_PER_SEC) << " s." << endl;
}