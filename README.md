# PLSM

PLSM设计用于预测 circRNA 亚细胞定位。采用多种circRNA 特性来提取信息丰富的circRNA 特征，包括circRNA 序列相似性网络、circRNA -疾病、circRNA -药物和circRNA -miRNA关联网络。采用强大的算法（node2vec和图注意力自动编码器）和一种新设计的方案来生成circRNA 特征类型。这些特征被输入到自注意力层和完全连接层中以进行预测

![image-20250909150329894](C:\Users\HJH\AppData\Roaming\Typora\typora-user-images\image-20250909150329894.png)

# Requirements 

Tensorflow = 1.14.0 

python = 3.7.16 

scikit-learn = 1.0.2

networkx = 2.6.3

