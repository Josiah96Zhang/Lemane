gcc -O3 -m64 frpca.c lemane_frpca_u.cpp matrix_vector_functions_intel_mkl_ext.c matrix_vector_functions_intel_mkl.c -liomp5 -lpthread -ldl -lm -fopenmp -w -o lemane_frpca_u -lstdc++ -lgflags
gcc -O3 -m64 frpca.c lemane_frpca_d.cpp matrix_vector_functions_intel_mkl_ext.c matrix_vector_functions_intel_mkl.c -liomp5 -lpthread -ldl -lm -fopenmp -w -o lemane_frpca_d -lstdc++ -lgflags
g++ -pthread -march=core2 -std=c++11 -O3 -o lemane_svd_u lemane_svd_u.cpp
g++ -pthread -march=core2 -std=c++11 -O3 -o lemane_svd_d lemane_svd_d.cpp

g++ -march=core2 -std=c++11 -O3 -o gendata_u gendata_u.cpp -lgflags
g++ -march=core2 -std=c++11 -O3 -o gendata_d gendata_d.cpp -lgflags

g++ -march=core2 -std=c++11 -O3 -o linkpred_u linkpred_u.cpp -lgflags
g++ -march=core2 -std=c++11 -O3 -o linkpred_d linkpred_d.cpp -lgflags
