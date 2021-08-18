./lemane_frpca_u -graph BlogCatalog -graph_path data/ -task class -delta 0.00001
./lemane_svd_d -graph wiki -graph_path data/ -task class -delta 0.00001
./lemane_frpca_d -graph tweibo -graph_path data/ -task class -delta 0.0001
./lemane_frpca_u -graph orkut -graph_path data/ -task class -delta 0.0001

python labelclassification.py --graph BlogCatalog --method lemane_frpca_class
python labelclassification.py --graph wiki --method lemane_svd_class
python labelclassification.py --graph tweibo --method lemane_frpca_class
python labelclassification.py --graph orkut --method lemane_frpca_class
