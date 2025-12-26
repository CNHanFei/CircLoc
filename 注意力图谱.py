import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import argparse
import tensorflow._api.v2.compat.v1 as tf
import matplotlib.pyplot as plt
import seaborn as sns
from gate_trainer import GATETrainer


def sim_thresholding(matrix: np.ndarray, threshold):
    matrix_copy = matrix.copy()
    matrix_copy[matrix_copy >= threshold] = 1
    matrix_copy[matrix_copy < threshold] = 0
    print(f"rest links: {np.sum(np.sum(matrix_copy))}")
    return matrix_copy


def single_generate_graph_adj_and_feature(network, feature):
    features = sp.csr_matrix(feature).tolil().todense()

    graph = nx.from_numpy_matrix(network)
    adj = nx.adjacency_matrix(graph)
    adj = sp.coo_matrix(adj)

    return adj, features


def get_gate_feature_and_attention(adj, features, epochs, l):
    args = parse_args(epochs=epochs, l=l)
    feature_dim = features.shape[1]
    args.hidden_dims = [feature_dim] + args.hidden_dims

    G, S, R = prepare_graph_data(adj)
    gate_trainer = GATETrainer(args)
    gate_trainer(G, features, S, R)
    embeddings, attention_matrices = gate_trainer.infer(G, features, S, R)
    tf.reset_default_graph()
    return embeddings, attention_matrices


def prepare_graph_data(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    data = adj.tocoo().data
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col


def parse_args(epochs, l):
    parser = argparse.ArgumentParser(description="Run gate.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate. Default is 0.001.')
    parser.add_argument('--n-epochs', default=epochs, type=int,
                        help='Number of epochs')
    parser.add_argument('--hidden-dims', type=list, nargs='+', default=[256, 128],
                        help='Number of dimensions.')
    parser.add_argument('--lambda-', default=l, type=float,
                        help='Parameter controlling the contribution of graph structure reconstruction in the loss function.')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='Dropout.')
    parser.add_argument('--gradient_clipping', default=5.0, type=float,
                        help='gradient clipping')
    return parser.parse_args()


def plot_attention_matrix(attention_matrix, layer_idx, sample_indices=None, save_path=None):
    """
    绘制注意力系数矩阵
    Args:
        attention_matrix: 注意力系数矩阵 (numpy array)
        layer_idx: 层索引
        sample_indices: 要绘制的样本索引列表，如果为None则绘制所有样本
        save_path: 保存图片的路径
    """
    plt.figure(figsize=(12, 10))

    if sample_indices is not None:
        # 只绘制指定的样本
        attention_subset = attention_matrix[sample_indices][:, sample_indices]
        title = f"Attention Matrix (Layer {layer_idx}) - Subset of {len(sample_indices)} samples"
    else:
        # 绘制所有样本
        attention_subset = attention_matrix
        title = f"Attention Matrix (Layer {layer_idx}) - All {attention_matrix.shape[0]} samples"

    # 使用热力图显示注意力系数
    sns.heatmap(attention_subset, cmap='viridis',
                xticklabels=50, yticklabels=50,
                cbar_kws={'label': 'Attention Coefficient'})

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Node Index', fontsize=12)
    plt.ylabel('Node Index', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention matrix saved to {save_path}")

    plt.tight_layout()
    plt.show()


def plot_attention_distribution(attention_matrix, layer_idx, save_path=None):
    """
    绘制注意力系数的分布
    """
    # 提取非零的注意力系数
    non_zero_attentions = attention_matrix[attention_matrix > 0]

    plt.figure(figsize=(10, 6))

    # 绘制直方图
    plt.hist(non_zero_attentions.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Attention Coefficient Distribution (Layer {layer_idx})', fontsize=14, fontweight='bold')
    plt.xlabel('Attention Coefficient Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f'Non-zero attentions: {len(non_zero_attentions)}\n'
    stats_text += f'Mean: {non_zero_attentions.mean():.4f}\n'
    stats_text += f'Std: {non_zero_attentions.std():.4f}\n'
    stats_text += f'Min: {non_zero_attentions.min():.4f}\n'
    stats_text += f'Max: {non_zero_attentions.max():.4f}'

    plt.text(0.7, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention distribution saved to {save_path}")

    plt.tight_layout()
    plt.show()


def plot_node_attention_patterns(attention_matrix, layer_idx, node_indices, save_path=None):
    """
    绘制特定节点的注意力模式
    """
    num_nodes = len(node_indices)
    fig, axes = plt.subplots(num_nodes, 1, figsize=(12, 3 * num_nodes))

    if num_nodes == 1:
        axes = [axes]

    for idx, node_idx in enumerate(node_indices):
        ax = axes[idx]
        node_attention = attention_matrix[node_idx, :]

        # 找出注意力系数最高的几个邻居
        sorted_indices = np.argsort(node_attention)[::-1]
        top_k = min(10, len(sorted_indices))

        # 绘制条形图
        top_indices = sorted_indices[:top_k]
        top_values = node_attention[top_indices]

        bars = ax.bar(range(top_k), top_values, color='skyblue', edgecolor='black')

        # 为每个条形添加数值标签
        for bar, value in zip(bars, top_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_title(f'Node {node_idx} - Top {top_k} Attention Weights (Layer {layer_idx})', fontsize=12)
        ax.set_xlabel('Neighbor Node Index', fontsize=10)
        ax.set_ylabel('Attention Coefficient', fontsize=10)
        ax.set_xticks(range(top_k))
        ax.set_xticklabels([str(i) for i in top_indices], rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Node attention patterns saved to {save_path}")

    plt.show()


if __name__ == '__main__':
    # 读取数据
    df_disease = pd.read_excel('./../dataset/circRNA_protein_features_128.xlsx', header=None, index_col=0)
    df_func = pd.read_excel('./../dataset/7类cell_similarity.xlsx', index_col=0)

    feature = df_disease.values
    similarity = df_func.values

    # 二值化
    network = sim_thresholding(similarity, 0.45)
    adj, features = single_generate_graph_adj_and_feature(network, feature)

    # 获取GATE特征和注意力矩阵
    embeddings, attention_matrices = get_gate_feature_and_attention(adj, features, 500, 0.5)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of attention matrices (layers): {len(attention_matrices)}")

    # 保存embeddings
    embeddings_file_path = './注意力图谱/gate_feature_disease.csv'
    np.savetxt(embeddings_file_path, embeddings, delimiter=',')
    print(f"Embeddings saved to {embeddings_file_path}")

    # 对每一层的注意力矩阵进行可视化
    for layer_idx, attention_sparse in enumerate(attention_matrices):
        print(f"\n=== Processing Layer {layer_idx} ===")

        # 将稀疏矩阵转换为密集矩阵
        attention_dense = attention_sparse.toarray()
        print(f"Attention matrix shape: {attention_dense.shape}")
        print(f"Number of non-zero elements: {attention_sparse.nnz}")
        print(f"Sparsity: {1 - attention_sparse.nnz / (attention_dense.shape[0] * attention_dense.shape[1]):.4f}")

        # 保存注意力矩阵
        attention_file_path = f'./注意力图谱/gate_feature_protein{layer_idx}.csv'
        np.savetxt(attention_file_path, attention_dense, delimiter=',')
        print(f"Attention matrix saved to {attention_file_path}")

        # # 1. 绘制完整的注意力矩阵（使用子集以避免内存问题）
        # # 如果样本太多，我们可以先绘制前100个样本
        # if attention_dense.shape[0] > 100:
        #     sample_indices = list(range(0, 100, 2))  # 取前100个样本中的50个
        #     plot_attention_matrix(attention_dense, layer_idx,
        #                           sample_indices=sample_indices,
        #                           save_path=f'./注意力图谱/attention_matrix_layer{layer_idx}_subset.png')
        # else:
        #     plot_attention_matrix(attention_dense, layer_idx,
        #                           save_path=f'./注意力图谱/attention_matrix_layer{layer_idx}_full.png')
        #
        # # 2. 绘制注意力系数分布
        # plot_attention_distribution(attention_dense, layer_idx,
        #                             save_path=f'./注意力图谱/attention_distribution_layer{layer_idx}.png')
        #
        # # 3. 绘制特定节点的注意力模式（例如，前5个节点）
        # num_nodes_to_plot = min(5, attention_dense.shape[0])
        # plot_node_attention_patterns(attention_dense, layer_idx,
        #                              node_indices=list(range(num_nodes_to_plot)),
        #                              save_path=f'./注意力图谱/node_attention_patterns_layer{layer_idx}.png')

        # 4. 保存每个节点的平均注意力系数
        node_mean_attention = attention_dense.mean(axis=1)
        mean_attention_file = f'./注意力图谱/node_mean_attention_layer{layer_idx}.csv'
        np.savetxt(mean_attention_file, node_mean_attention, delimiter=',')
        print(f"Node mean attention saved to {mean_attention_file}")

        # # 5. 绘制平均注意力系数的分布
        # plt.figure(figsize=(10, 6))
        # plt.hist(node_mean_attention, bins=50, alpha=0.7, color='green', edgecolor='black')
        # plt.title(f'Node Mean Attention Distribution (Layer {layer_idx})', fontsize=14)
        # plt.xlabel('Mean Attention Coefficient', fontsize=12)
        # plt.ylabel('Frequency', fontsize=12)
        # plt.grid(True, alpha=0.3)
        # plt.savefig(f'./注意力图谱/node_mean_attention_distribution_layer{layer_idx}.png',
        #             dpi=300, bbox_inches='tight')
        # plt.close()

    print("\n=== Visualization Complete ===")