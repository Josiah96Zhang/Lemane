./gen_data_u -graph BlogCatalog -test_ratio 0.3
./lemane_frpca_u -graph BlogCatalog -graph_path lp_data/train_graph/ -task link -delta 0.0000001
./linkpred_u -graph BlogCatalog -method lemane_frpca_link

./gen_data_d -graph wiki -test_ratio 0.3
./lemane_svd_d -graph wiki -graph_path lp_data/train_graph/ -task link -delta 0.00001
./linkpred_d -graph wiki -method lemane_svd_link

./gen_data_d -graph wikivote -test_ratio 0.3
./lemane_frpca_d -graph wikivote -graph_path lp_data/train_graph/ -task link -delta 0.00001
./linkpred_d -graph wiki -method lemane_frpca_link

./gen_data_d -graph slashdot -test_ratio 0.3
./lemane_frpca_d -graph slashdot -graph_path lp_data/train_graph/ -task link -delta 0.00001
./linkpred_d -graph wiki -method lemane_frpca_link

./gen_data_d -graph tweibo -test_ratio 0.3
./lemane_frpca_d -graph tweibo -graph_path lp_data/train_graph/ -task link -delta 0.00001
./linkpred_d -graph wiki -method lemane_frpca_link

./gen_data_d -graph orkut -test_ratio 0.3
./lemane_frpca_u -graph orkut -graph_path lp_data/train_graph/ -task link -delta 0.0001
./linkpred_d -graph wiki -method lemane_frpca_link
