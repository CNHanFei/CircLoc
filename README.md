# PLSM

PLSM is designed to predict the subcellular localization of circRNAs. It employs multiple circRNA characteristics to extract informative circRNA features, including circRNA sequence similarity networks, circRNA-disease, circRNA-drug, and circRNA-miRNA association networks. Robust algorithms (node2vec and graph attention autoencoders) and a newly designed scheme are adopted to generate circRNA feature types. These features are fed into a self-attention layer and fully connected layers for prediction.

![image-20250909150329894](C:\Users\HJH\AppData\Roaming\Typora\typora-user-images\image-20250909150329894.png)

# Requirements 

Tensorflow = 1.14.0 

python = 3.7.16 

scikit-learn = 1.0.2

networkx = 2.6.3

