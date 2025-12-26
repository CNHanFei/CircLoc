import tensorflow._api.v2.compat.v1 as tf


class GATE():
    # 初始化函数：定义隐藏层维度和lambda参数
    def __init__(self, hidden_dims, lambda_):
        self.lambda_ = lambda_  # 控制结构重构损失的权重
        self.n_layers = len(hidden_dims) - 1  # 层数等于隐藏层维度个数减去1
        self.W, self.v = self.define_weights(hidden_dims)  # 定义权重
        self.C = {}  # 存储图注意力层的计算结果

#X节点的特征表示，A图的邻接矩阵
    # 主要的调用函数：接收邻接矩阵 A、节点特征矩阵 X、正负样本集合 R 和 S
    def __call__(self, A, X, R, S):
        # Encoder部分：逐层计算节点的隐藏表示
        H = X  # 初始的节点特征是输入的 X
        for layer in range(self.n_layers):
            H = self.__encoder(A, H, layer)  # 通过编码器进行计算

        # 最终的节点表示
        self.H = H  # H 是最终计算得到的节点嵌入表示

        # Decoder部分：逐层解码生成最终的节点表示
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder(H, layer)  # 解码器进行反向计算
        X_ = H  # X_ 是最终的重构结果（解码后的节点特征）

        # 计算节点特征的重构损失（L2范数）
        features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))  # 计算节点特征的损失

        # 计算图结构的重构损失
        self.S_emb = tf.nn.embedding_lookup(self.H, S)  # 获取正样本的节点嵌入
        self.R_emb = tf.nn.embedding_lookup(self.H, R)  # 获取负样本的节点嵌入
        structure_loss = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb * self.R_emb, axis=-1)))  # 计算图结构的损失
        structure_loss = tf.reduce_sum(structure_loss)  # 对所有样本的损失求和

        # 总损失：节点特征重构损失 + lambda * 图结构重构损失
        self.loss = features_loss + self.lambda_ * structure_loss

        return self.loss, self.H, self.C


    # 编码器：通过图注意力机制计算隐藏表示
    def __encoder(self, A, H, layer):
        # 通过矩阵乘法与权重W[layer]计算节点的隐藏表示
        H = tf.matmul(H, self.W[layer])
        # 使用图注意力层计算节点的权重矩阵
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        # 通过图注意力机制对隐藏表示进行加权平均
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i+1]))

        Ws_att = {}
        for i in range(self.n_layers):
            v = {}
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i+1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i+1], 1))
            Ws_att[i] = v

        return W, Ws_att

    def graph_attention_layer(self, A, M, v, layer):

        with tf.variable_scope("layer_%s"% layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            print(f2.shape, M.shape, v[1].shape)
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)

            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                         values=tf.nn.sigmoid(logits.values),
                                         dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions