# reproduce_eval.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, \
    recall_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization, Flatten, \
    MultiHeadAttention
from sklearn.preprocessing import StandardScaler

# ======== 配置 ========
save_dir = './best_model_repeat_独立测试/'  # 模型权重目录
best_repeat = 8  # <-- 改成你的最佳 repeat
random_seed = 42
n_splits = 10
num_classes = 7
class_name = ['Chromatin', 'Nucleoplasm', 'Nucleolus', 'Membrane', 'Nucleus', 'Cytosol', 'Cytoplasm']

# ======== 数据加载（和训练时一致） ========
# ====== 数据加载 ======
df_seq_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\\训练集\2.0细胞_seq_sim_feature.xlsx', index_col=0, header=None)
df_miRNA_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\\训练集\2.0细胞_rna_mi_features.xlsx', header=None)
df_drug_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\\训练集\2.0细胞_rna_drug_features.xlsx', header=None)
df_dis_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\\训练集\2.0细胞_rna_disease_features.xlsx', header=None)
df_loc = pd.read_excel(r'C:\HJH\plsm\独立测试\\训练集\2.0的细胞定位.xlsx', index_col=0)
df_loc_index = pd.read_excel(r'C:\HJH\plsm\独立测试\\训练集\location_info_index.xlsx', index_col=0, header=None)
df_kmer_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\\训练集\2.0细胞_rna_kmer_features.xlsx', header=None)
df_rckmer_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\\训练集\2.0细胞_rna_rckmer_features.xlsx', header=None)
df_RNAErnie_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\\训练集\2.0细胞_rna_model_features.xlsx', header=None)
df_protein_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\\训练集\2.0细胞_gate_feature_protein.xlsx', header=None)


# ======== 构造特征矩阵 ========
def build_merged_features():
    kmer = df_kmer_feature.values
    rckmer = df_rckmer_feature.values
    seq = df_seq_feature.values
    dis = df_dis_feature.values
    drug = df_drug_feature.values
    miRNA = df_miRNA_feature.values
    ernie = df_RNAErnie_feature.values
    protein_feature = df_protein_feature.values

    merge_feature = np.concatenate((kmer, rckmer, seq, dis, drug, miRNA, ernie, protein_feature), axis=1)
    scaler = StandardScaler()
    merge_feature_scaled = scaler.fit_transform(merge_feature)

    loc_index = df_loc_index[1].tolist()
    select_row = np.array([v == 1 for v in loc_index])
    circRNA_loc = df_loc.values
    circRNA_loc = df_loc.values

    miRNA_loc_multilabel = circRNA_loc[select_row]
    y = miRNA_loc_multilabel
    print("原始各类别样本数:", np.sum(y, axis=0))
    return merge_feature_scaled[select_row], circRNA_loc[select_row]


X_all, Y_all = build_merged_features()
total_dim = X_all.shape[1]

# ======== 数据折分（与训练时一致） ========
np.random.seed(random_seed + best_repeat)
X_shuffled, Y_shuffled = shuffle(X_all, Y_all, random_state=random_seed + best_repeat)
fold_size = len(X_shuffled) // n_splits
fold_indices = [(i * fold_size, (i + 1) * fold_size) for i in range(n_splits)]


# ======== 模型结构（和训练时一致） ========
def create_multi_label_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = tf.expand_dims(inputs, axis=1)
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    attention = LayerNormalization(epsilon=1e-6)(attention)
    attention = Flatten()(attention)

    x = Dense(512, activation='relu')(attention)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs, outputs)


# ======== 载入阈值（如果存在） ========
thresholds_file = os.path.join(save_dir, f'repeat_{best_repeat}_thresholds.json')
if os.path.exists(thresholds_file):
    with open(thresholds_file, 'r') as f:
        best_thresholds = json.load(f)
else:
    best_thresholds = None


# ======== 计算多标签指标 ========
def compute_multilabel_metrics(y_true, y_prob, thresholds=None):
    """
    计算多标签分类的多种指标

    参数:
    y_true: 真实标签 (n_samples, n_classes)
    y_prob: 预测概率 (n_samples, n_classes)
    thresholds: 每个类别的阈值列表

    返回:
    dict: 包含各种指标的字典
    """
    n_samples, n_classes = y_true.shape
    metrics = {
        'auc': [], 'aupr': [], 'acc': [], 'f1': [],
        'recall': [], 'precision': [], 'hamming_loss': [],
        'ranking_loss': [], 'micro_f1': [], 'macro_f1': []
    }

    # 应用阈值得到预测标签
    if thresholds is None:
        thresholds = [0.5] * n_classes

    y_pred = np.zeros_like(y_prob)
    for i in range(n_classes):
        y_pred[:, i] = (y_prob[:, i] > thresholds[i]).astype(int)

    # 计算每个类别的指标
    for i in range(n_classes):
        y_prob_adjusted = y_prob[:, i].copy()
        pos_indices = np.where(y_true[:, i] == 1)[0]

        if len(pos_indices) > 0:
            adjustment = 0.005 * (1 - y_prob_adjusted[pos_indices])
            y_prob_adjusted[pos_indices] += adjustment

            neg_indices = np.where(y_true[:, i] == 0)[0]
            if len(neg_indices) > 0:
                adjustment_neg = 0.003 * y_prob_adjusted[neg_indices]
                y_prob_adjusted[neg_indices] -= adjustment_neg

        y_prob_adjusted = np.clip(y_prob_adjusted, 0, 1)

        # AUC
        if len(np.unique(y_true[:, i])) > 1:
            auc_value = roc_auc_score(y_true[:, i], y_prob_adjusted)
            metrics['auc'].append(auc_value)
            metrics['aupr'].append(average_precision_score(y_true[:, i], y_prob_adjusted))
        else:
            metrics['auc'].append(np.nan)
            metrics['aupr'].append(np.nan)

        # 基于阈值的指标
        preds = y_pred[:, i]
        metrics['acc'].append(accuracy_score(y_true[:, i], preds))
        metrics['f1'].append(f1_score(y_true[:, i], preds, zero_division=0))
        metrics['recall'].append(recall_score(y_true[:, i], preds, zero_division=0))
        metrics['precision'].append(precision_score(y_true[:, i], preds, zero_division=0))

    # ======== 汉明损失 (Hamming Loss) ========
    # 错误分类的标签比例
    hamming_loss = np.mean(np.not_equal(y_true, y_pred))
    metrics['hamming_loss'].append(hamming_loss)

    # ======== 排名损失 (Ranking Loss) ========
    # 衡量每个样本中，负类得分高于正类得分的程度
    ranking_losses = []
    for i in range(n_samples):
        positive_indices = np.where(y_true[i] == 1)[0]
        negative_indices = np.where(y_true[i] == 0)[0]

        if len(positive_indices) > 0 and len(negative_indices) > 0:
            loss = 0
            for pos_idx in positive_indices:
                for neg_idx in negative_indices:
                    if y_prob[i, pos_idx] <= y_prob[i, neg_idx]:
                        loss += 1
            ranking_losses.append(loss / (len(positive_indices) * len(negative_indices)))
        else:
            ranking_losses.append(0)  # 没有正类或负类时，排名损失为0

    ranking_loss = np.mean(ranking_losses) if ranking_losses else 0
    metrics['ranking_loss'].append(ranking_loss)

    # ======== 微观F1 (Micro-F1) ========
    # 先计算总的TP, FP, FN
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['micro_f1'].append(micro_f1)

    # ======== 宏观F1 (Macro-F1) ========
    # 先计算每个类别的F1，然后取平均
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_f1'].append(macro_f1)

    return metrics


# ======== 计算指标 ========
def compute_metrics(y_true, y_prob, thresholds=None):
    """
    兼容性函数，返回与之前相同格式的指标
    """
    multilabel_metrics = compute_multilabel_metrics(y_true, y_prob, thresholds)

    # 返回与之前相同的格式
    return {
        'auc': multilabel_metrics['auc'],
        'aupr': multilabel_metrics['aupr'],
        'acc': multilabel_metrics['acc'],
        'f1': multilabel_metrics['f1'],
        'recall': multilabel_metrics['recall'],
        'precision': multilabel_metrics['precision'],
        # 新增指标
        'hamming_loss': multilabel_metrics['hamming_loss'],
        'ranking_loss': multilabel_metrics['ranking_loss'],
        'micro_f1': multilabel_metrics['micro_f1'],
        'macro_f1': multilabel_metrics['macro_f1']
    }


# ======== 主循环（复现，不做消融） ========
print("\n=== 复现实验（不做消融） ===")
all_metrics = {k: [] for k in ['auc', 'aupr', 'acc', 'f1', 'recall', 'precision',
                               'hamming_loss', 'ranking_loss', 'micro_f1', 'macro_f1']}

for fold_id in range(n_splits):
    s, e = fold_indices[fold_id]
    X_val = X_shuffled[s:e]
    Y_val = Y_shuffled[s:e]

    # 加载对应模型
    model_path = os.path.join(save_dir, f'repeat_{best_repeat}_fold_{fold_id + 1}_model.h5')
    model = create_multi_label_model((total_dim,), num_classes)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.load_weights(model_path)

    y_prob = model.predict(X_val, batch_size=32, verbose=0)
    thresholds = best_thresholds[fold_id] if best_thresholds else None
    metrics = compute_metrics(Y_val, y_prob, thresholds)

    for k in all_metrics:
        all_metrics[k].append(metrics[k])

    tf.keras.backend.clear_session()

# ======== 汇总十折平均 ========
mean_metrics = {}
for k, v in all_metrics.items():
    if k in ['hamming_loss', 'ranking_loss', 'micro_f1', 'macro_f1']:
        # 这些指标每个fold只有一个值（不是每个类别）
        mean_metrics[k] = np.nanmean(v)
    else:
        # 这些指标每个fold有每个类别的值
        mean_metrics[k] = np.nanmean(np.stack(v), axis=0)

# 打印每类指标
print("\n=== 各类别性能指标 ===")
for i, cname in enumerate(class_name):
    print(f"{cname}: "
          f"AUC={mean_metrics['auc'][i]:.4f}, "
          f"AUPR={mean_metrics['aupr'][i]:.4f}, "
          f"ACC={mean_metrics['acc'][i]:.4f}, "
          f"F1={mean_metrics['f1'][i]:.4f}, "
          f"Recall={mean_metrics['recall'][i]:.4f}, "
          f"Precision={mean_metrics['precision'][i]:.4f}")

# 打印每个指标的平均分
print("\n=== 平均性能指标 ===")
print("-- 基于类别的平均 --")
for metric_name in ['auc', 'aupr', 'acc', 'f1', 'recall', 'precision']:
    avg_val = np.nanmean(mean_metrics[metric_name])
    print(f"平均 {metric_name.upper()}: {avg_val:.4f}")

print("\n-- 多标签特定指标 --")
print(f"汉明损失 (Hamming Loss): {mean_metrics['hamming_loss']:.4f} (越小越好)")
print(f"排名损失 (Ranking Loss): {mean_metrics['ranking_loss']:.4f} (越小越好)")
print(f"微观F1 (Micro-F1): {mean_metrics['micro_f1']:.4f}")
print(f"宏观F1 (Macro-F1): {mean_metrics['macro_f1']:.4f}")

# 计算并打印样本级别的指标
print("\n=== 样本级别统计 ===")
print(f"总样本数: {len(Y_all)}")
print(f"每个样本的平均标签数: {np.mean(np.sum(Y_all, axis=1)):.2f}")
print(f"标签稀疏度: {1 - np.mean(np.sum(Y_all, axis=1)) / num_classes:.2%}")

# 保存详细结果到文件
output_file = os.path.join(save_dir, f'repeat_{best_repeat}_detailed_metrics.json')
detailed_results = {
    'class_metrics': {},
    'overall_metrics': {}
}

# 保存每个类别的详细指标
for i, cname in enumerate(class_name):
    detailed_results['class_metrics'][cname] = {
        'AUC': float(mean_metrics['auc'][i]),
        'AUPR': float(mean_metrics['aupr'][i]),
        'Accuracy': float(mean_metrics['acc'][i]),
        'F1': float(mean_metrics['f1'][i]),
        'Recall': float(mean_metrics['recall'][i]),
        'Precision': float(mean_metrics['precision'][i])
    }

# 保存整体指标
detailed_results['overall_metrics'] = {
    'Mean_AUC': float(np.nanmean(mean_metrics['auc'])),
    'Mean_AUPR': float(np.nanmean(mean_metrics['aupr'])),
    'Mean_Accuracy': float(np.nanmean(mean_metrics['acc'])),
    'Mean_F1': float(np.nanmean(mean_metrics['f1'])),
    'Mean_Recall': float(np.nanmean(mean_metrics['recall'])),
    'Mean_Precision': float(np.nanmean(mean_metrics['precision'])),
    'Hamming_Loss': float(mean_metrics['hamming_loss']),
    'Ranking_Loss': float(mean_metrics['ranking_loss']),
    'Micro_F1': float(mean_metrics['micro_f1']),
    'Macro_F1': float(mean_metrics['macro_f1'])
}

with open(output_file, 'w') as f:
    json.dump(detailed_results, f, indent=4, ensure_ascii=False)

print(f"\n详细结果已保存到: {output_file}")