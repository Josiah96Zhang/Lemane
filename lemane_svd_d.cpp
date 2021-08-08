#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <deque>
#include <unordered_map>
#include "Graph.h"
#include "SVD.h"
#include <fstream>
#include <cstring>
#include <thread>
#include <mutex>
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <chrono>
#include <climits>
#include <random>
#include <gflags/gflags.h>

DEFINE_string(graph, "BlogCatalog", "Graph name.");
DEFINE_string(graph_path, "data/", "Graph filepath.");
DEFINE_string(emb_path, "embds/", "Embedding filepath.");
DEFINE_string(alpha_path, "alpha/", "Alpha filepath.");
DEFINE_string(task, "class", "Downstream task.");
DEFINE_int32(pass, 12, "Pass.");
DEFINE_int32(d, 128, "Embedding dimension.");
DEFINE_double(delta, 0.00001, "Push_error.");
DEFINE_int32(num_thread, 24, "Number of threads.");

using namespace Eigen;
using namespace std;

bool maxScoreCmpTriplet(const Triplet<double>& a, const Triplet<double>& b){
    return a.value() > b.value();
}

const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");

void Gpush(int* random_w, int start, int end, Graph* g, vector<Triplet<double>>* answer, double delta){
    unordered_map<int, double> mapR;
    queue<int> Q;
    int n_nodes = g->n;
    vector <double> mapSPPR(g->n);
  
    for (int it = start; it < end; it++) {
        int src = random_w[it];
        mapR[src] = 1.0;
        Q.push(src);
        double gamma = 0;
        int rsize = 0;
        //push
        while (!Q.empty()) {                     
            int uk = Q.front();
            int u = uk%n_nodes;
            int k = uk/n_nodes;
            Q.pop();
            double& residue = mapR[uk];
            double reserve = residue * g->alpha[k];
            mapSPPR[u] += reserve;
            residue -= reserve;
            double update = residue / (double)g->outdegree[u];
            mapR.erase(uk);
            for (int j = 0; j < g->outdegree[u]; j++) {
                int v = g->outAdjList[u][j];
                double threshold = g->outdegree[v] * delta;
                int vk = (k+1)  * n_nodes + v;
                double& rvk = mapR[vk];
                rvk += update;
                if ((rvk - update < threshold) && (rvk >= threshold)) {
                    Q.push(vk);
                }
            }
        }
        //reserve of last step
        for (auto it = mapR.begin(); it != mapR.end(); ++it) {
            int u = it->first%n_nodes;
            int k = it->first/n_nodes;
            double r = it->second;
            double rk = r * g->alpha[k];
            mapSPPR[u] += rk;
        }
    
        for (int i = 0; i < g->n; ++i) {
            if(mapSPPR[i] > delta){
                answer->push_back(Triplet<double>(src, i, mapSPPR[i]));
            }
            mapSPPR[i] = 0;
        }
        mapSPPR.clear();
        mapR.clear();
    }
    return;
}

void GpushT(int* random_w, int start, int end, Graph* g, vector<Triplet<double>>* answer, double delta){
    unordered_map<int, double> mapR;
    queue<int> Q;
    int n_nodes = g->n;
    vector <double> mapSPPR(g->n);
  
    for(int it = start; it < end; it++){
        int src = random_w[it];
        mapR[src] = 1.0;                           
        Q.push(src);
        //push
        while(!Q.empty()){                     
            int uk = Q.front();
            int u = uk%n_nodes;
            int k = uk/n_nodes;
            Q.pop();
            double& residue = mapR[uk];
            double reserve = residue * g->alpha[k];
            mapSPPR[u] += reserve;
            residue -= reserve;
            double update = residue / (double)g->indegree[u];
            mapR.erase(uk);
            for(int j = 0; j < g->indegree[u]; j++) {
              int v = g->inAdjList[u][j];
              double threshold = g->indegree[v] * delta;
              int vk = (k+1)  * n_nodes + v;
              double& rvk = mapR[vk];
              rvk += update;
              if ((rvk - update < threshold) && (rvk >= threshold)) {
                  Q.push(vk);
              }
            }
        }
        //reserve of last step
        for (auto it = mapR.begin(); it != mapR.end(); ++it) {
            int u = it->first%n_nodes;
            int k = it->first/n_nodes;
            double r = it->second;
            double rk = r * g->alpha[k];
            mapSPPR[u] += rk;
        }
    
        for (int i = 0; i < g->n; ++i){
            if (mapSPPR[i] > delta) {
                answer->push_back(Triplet<double>(i, src, mapSPPR[i]));
            }
            mapSPPR[i] = 0;
        }
        mapSPPR.clear();
        mapR.clear();
    }
    return;
}


int main(int argc,  char **argv){
    auto start_time = std::chrono::system_clock::now();
    srand((unsigned)time(0));
    google::ParseCommandLineFlags(&argc, &argv, true);
    
    string dataset = FLAGS_graph_path  + FLAGS_graph + ".txt";
    string alphafile = FLAGS_alpha_path + FLAGS_graph + "_" + FLAGS_task + ".txt";
    string outUfile = FLAGS_emb_path + FLAGS_graph + "/" + FLAGS_graph + "_lemane_svd_" + FLAGS_task + "_U.csv";
    string outVfile = FLAGS_emb_path + FLAGS_graph + "/" + FLAGS_graph + "_lemane_svd_" + FLAGS_task + "_V.csv";
    ofstream outU(outUfile.c_str());
    ofstream outV(outVfile.c_str());

    Graph g;
    g.inputGraph(dataset, alphafile);
    clock_t start = clock();
    
    cout << outUfile << endl;
    cout << outVfile << endl;

    int d = FLAGS_d;
    cout << "dimension: " << d << endl;
    cout << "reservemin: " << FLAGS_delta << " residuemax: " << FLAGS_delta << endl;
  
    int* random_w = new int[g.n];
    for (int i = 0; i < g.n; i++) {
        random_w[i] = i;
    }

    for (int i = 0; i < g.n; i++) {
      int r = rand()%(g.n-i);
      int temp = random_w[i + r];
      random_w[i + r] = random_w[i];
      random_w[i] = temp;
    }
  
    cout << "SPPR computation" << endl;
    auto sppr_start_time = std::chrono::system_clock::now();
    vector<thread> threads;
    vector<vector<Triplet<double>>> tripletList(FLAGS_num_thread);
    deque<Triplet<double>> TotalTripletList;
    for (int k = 1; k <= FLAGS_num_thread; k++) {
        int start = (k-1)*(g.n/FLAGS_num_thread);
        int end = 0;
        if (k == FLAGS_num_thread){
            end = g.n;
        } 
        else {
            end = k*(g.n/FLAGS_num_thread);
        }
        threads.push_back(thread(Gpush, random_w, start, end, &g, &tripletList[k-1], FLAGS_delta));
    }
    
    for (int k = 0; k < FLAGS_num_thread ; k++) {
        threads[k].join();
    }
    vector<thread>().swap(threads);
  
    for (int k = 1; k <= FLAGS_num_thread; k++) {
        int start = (k-1)*(g.n/FLAGS_num_thread);
        int end = 0;
        if (k == FLAGS_num_thread){
            end = g.n;
        } 
        else{
            end = k*(g.n/FLAGS_num_thread);
        }
        threads.push_back(thread(GpushT, random_w, start, end, &g, &tripletList[k-1], FLAGS_delta));
    }
    
    for (int k = 0; k < FLAGS_num_thread; k++){
        threads[k].join();
    }
    vector<thread>().swap(threads);
  
    delete[] random_w;

    auto start_sppr_matrix_time = chrono::system_clock::now();
    auto elapsed_sppr_time = chrono::duration_cast<std::chrono::seconds>(start_sppr_matrix_time - sppr_start_time);
    cout << "Computing sppr time: "<< elapsed_sppr_time.count() << endl;


    long total_size = 0;
    for (int t = 0; t < FLAGS_num_thread; t++){
        total_size += tripletList[t].size();
    }
    cout << "Total size: " << total_size << endl;
    cout << "Computing TotalTripletList" << endl;
    for (int t = 0; t < FLAGS_num_thread; t++) {
        TotalTripletList.insert(TotalTripletList.end(), tripletList[t].begin(), tripletList[t].end());
        vector<Triplet<double>>().swap(tripletList[t]);
    }
    vector<vector<Triplet<double>>>().swap(tripletList);

    long nnz = TotalTripletList.size();
    cout << "nnz1 + nnz2 = " << nnz << endl;
    auto merge_sppr_time = chrono::system_clock::now();
    auto elapsed_merge_sppr_time = chrono::duration_cast<std::chrono::seconds>(merge_sppr_time - start_sppr_matrix_time);
    cout << "merge sppr vec time: "<< elapsed_merge_sppr_time.count() << endl;


    //Combine bidirectional sppr vector into Eigen:Sparse
    long max_nnz = INT_MAX;
    if (nnz > max_nnz) {
        nth_element(TotalTripletList.begin(), TotalTripletList.begin()+max_nnz, TotalTripletList.end(), maxScoreCmpTriplet);
        TotalTripletList.erase(TotalTripletList.begin()+max_nnz+1, TotalTripletList.end());
        nnz = max_nnz;
    }

    cout << "Deque to sparse matrix" << endl;
    // Rewrite SparseMatrix::setFromTriplets to optimize memory usage
    SparseMatrix<double> sppr_matrix(g.n, g.n);
    SparseMatrix<double> trMat(g.n, g.n);
    deque<Triplet<double>>::iterator it;
    long max_step = nnz / 2 + 1;
    long step = 0;

    while (nnz){
        SparseMatrix<double>::IndexVector wi(trMat.outerSize());
        wi.setZero();
        if (nnz < max_step){
            step = nnz;
            nnz = 0;
        } 
        else {
            step = max_step;
            nnz -= max_step;
        }

        for (int j = 0; j < step; j++){
            wi(TotalTripletList[j].col())++;
        }

        trMat.reserve(wi);

        for (int j = 0; j < step; j++) {
            it = TotalTripletList.begin();
            double value1 = log(it->value() / FLAGS_delta);
            if (value1 > 0.0)
                trMat.insertBackUncompressed(it->row(), it->col()) = value1;
            else
                cout << value1 << endl;
            TotalTripletList.erase(TotalTripletList.begin());
        }

        trMat.collapseDuplicates(internal::scalar_sum_op<double, double>());
    }
    deque<Triplet<double>>().swap(TotalTripletList);
  
    sppr_matrix = trMat;
    trMat.resize(0, 0);
    trMat.data().squeeze();
  
    auto svd_start_time = chrono::system_clock::now();
    auto elapsed_deque_to_sparse_time = chrono::duration_cast<std::chrono::seconds>(svd_start_time - merge_sppr_time);
    cout << "Deque to sparse time: "<< elapsed_deque_to_sparse_time.count() << endl;
    cout << "Start SVD..." << endl;

    // Compute SVD
    SVD::SVD<SparseMatrix<double>> svd(sppr_matrix, d, FLAGS_pass);
    auto end_eb_time = chrono::system_clock::now();
    auto elapsed_svd_time = chrono::duration_cast<std::chrono::seconds>(end_eb_time - svd_start_time);
    cout << "SVD time: "<< elapsed_svd_time.count() << endl;

    VectorXd vS = svd.singularValues();
    double sppr_norm = sppr_matrix.norm()*sppr_matrix.norm();
    double S_norm = vS.norm()*vS.norm();
    cout << S_norm << " " << sppr_norm << " ratio: " << S_norm/sppr_norm << endl;

    MatrixXd S = vS.cwiseSqrt().asDiagonal();
    MatrixXd U = svd.matrixU()*S;
    MatrixXd V = svd.matrixV()*S;
    outU << U.format(CSVFormat);
    outV << V.format(CSVFormat);
    auto end_time = chrono::system_clock::now();
    auto elapsed_write_time = chrono::duration_cast<std::chrono::seconds>(end_time - end_eb_time);
    cout << "Writing out embedding time: "<< elapsed_write_time.count() << endl;
    auto elapsed_time = chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    cout << "Total embedding time: "<< elapsed_time.count() << endl;
}
