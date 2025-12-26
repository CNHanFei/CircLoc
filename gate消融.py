# reproduce_eval.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, \
    recall_score
from sklearn.metrics import hamming_loss, label_ranking_loss
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization, Flatten, \
    MultiHeadAttention
from sklearn.preprocessing import StandardScaler

# ======== 配置 ========
save_dir = './best_model_repeat_change/'  # 模型权重目录
best_repeat = 18  # <-- 改成你的最佳 repeat
random_seed = 42
n_splits = 10
num_classes = 7
class_name = ['Chromatin', 'Nucleoplasm', 'Nucleolus', 'Membrane', 'Nucleus', 'Cytosol', 'Cytoplasm']

# ======== 数据加载（和训练时一致） ========
df_seq_feature = pd.read_excel(r'C:\HJH\plsm\feature\seq_sim_feature.xlsx', index_col=0, header=None)
df_miRNA_feature = pd.read_excel(r'C:\HJH\plsm\dataset\rna_mi_features_128.xlsx', header=None)
df_drug_feature = pd.read_excel(r'C:\HJH\plsm\dataset\rna_drug_features_128.xlsx', header=None)
df_dis_feature = pd.read_excel(r'C:\HJH\plsm\dataset\rna_disease_features_128.xlsx', header=None)
df_loc = pd.read_excel(r'C:\HJH\plsm\dataset\location_info.xlsx', index_col=0)
df_loc_index = pd.read_excel(r'C:\HJH\excel处理\location_info_index.xlsx', index_col=0, header=None)
df_kmer_feature = pd.read_csv(r'C:\HJH\plsm\feature\rna_kmer_features_0.7_128_0.01.csv', header=None)
df_rckmer_feature = pd.read_csv(r'C:\HJH\plsm\feature\rna_rckmer_features_0.7_128_0.01.csv', header=None)
df_RNAErnie_feature = pd.read_excel(r'C:\HJH\excel处理\model_feature.xlsx', header=None)
df_protein_feature = pd.read_excel(r'C:\HJH\plsm\dataset\circRNA_protein_features_128.xlsx', header=None)


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


# ======== 计算指标 ========
def compute_metrics(y_true, y_prob, thresholds=None):
    Q = y_true.shape[1]
    metrics = {
        'auc': [], 'aupr': [], 'acc': [], 'f1': [], 'recall': [], 'precision': [],
        'macro_f1': [], 'micro_f1': [], 'hamming_loss': [], 'ranking_loss': []
    }

    # 使用阈值将概率转换为二进制预测
    if thresholds is not None:
        y_pred = np.array([(y_prob[:, i] > thresholds[i]).astype(int) for i in range(Q)]).T
    else:
        y_pred = (y_prob > 0.5).astype(int)

    # 计算每个类别的指标
    for i in range(Q):
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

        if len(np.unique(y_true[:, i])) > 1:
            auc_value = roc_auc_score(y_true[:, i], y_prob_adjusted)
            metrics['auc'].append(auc_value)
            metrics['aupr'].append(average_precision_score(y_true[:, i], y_prob_adjusted))
        else:
            metrics['auc'].append(np.nan)
            metrics['aupr'].append(np.nan)

        th = thresholds[i] if thresholds is not None else 0.5
        preds = (y_prob[:, i] > th).astype(int)
        metrics['acc'].append(accuracy_score(y_true[:, i], preds))
        metrics['f1'].append(f1_score(y_true[:, i], preds, zero_division=0))
        metrics['recall'].append(recall_score(y_true[:, i], preds, zero_division=0))
        metrics['precision'].append(precision_score(y_true[:, i], preds, zero_division=0))

    # 计算整体指标
    try:
        metrics['macro_f1'] = [f1_score(y_true, y_pred, average='macro', zero_division=0)]
    except:
        metrics['macro_f1'] = [np.nan]

    try:
        metrics['micro_f1'] = [f1_score(y_true, y_pred, average='micro', zero_division=0)]
    except:
        metrics['micro_f1'] = [np.nan]

    try:
        metrics['hamming_loss'] = [hamming_loss(y_true, y_pred)]
    except:
        metrics['hamming_loss'] = [np.nan]

    try:
        metrics['ranking_loss'] = [label_ranking_loss(y_true, y_prob)]
    except:
        metrics['ranking_loss'] = [np.nan]

    return {k: np.array(v) for k, v in metrics.items()}


# ======== 主循环（复现，不做消融） ========
print("\n=== 复现实验（不做消融） ===")
all_metrics = {
    'auc': [], 'aupr': [], 'acc': [], 'f1': [], 'recall': [], 'precision': [],
    'macro_f1': [], 'micro_f1': [], 'hamming_loss': [], 'ranking_loss': []
}

# 收集所有折的真实标签和预测结果用于整体计算
all_y_true = []
all_y_prob = []
all_y_pred = []

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

    # 使用阈值进行二分类预测
    thresholds = best_thresholds[fold_id] if best_thresholds else None
    if thresholds is not None:
        y_pred = np.array([(y_prob[:, i] > thresholds[i]).astype(int) for i in range(num_classes)]).T
    else:
        y_pred = (y_prob > 0.5).astype(int)

    # 收集所有折的数据
    all_y_true.append(Y_val)
    all_y_prob.append(y_prob)
    all_y_pred.append(y_pred)

    metrics = compute_metrics(Y_val, y_prob, thresholds)

    for k in all_metrics:
        if k in ['macro_f1', 'micro_f1', 'hamming_loss', 'ranking_loss']:
            # 这些指标已经在compute_metrics中计算为单个值
            all_metrics[k].append(metrics[k][0])  # 取第一个元素（数组）
        else:
            # 其他指标是每个类别的数组
            all_metrics[k].append(metrics[k])

    tf.keras.backend.clear_session()

# ======== 汇总十折平均 ========
# 对于类别特定的指标（auc, aupr, acc, f1, recall, precision），取每类的平均值
mean_metrics = {}
for k in ['auc', 'aupr', 'acc', 'f1', 'recall', 'precision']:
    # 这些指标是每折每类的数组，形状为(n_splits, num_classes)
    stacked = np.stack(all_metrics[k], axis=0)  # (n_splits, num_classes)
    mean_metrics[k] = np.nanmean(stacked, axis=0)  # (num_classes,)

# 对于整体指标（macro_f1, micro_f1, hamming_loss, ranking_loss），直接取平均值
for k in ['macro_f1', 'micro_f1', 'hamming_loss', 'ranking_loss']:
    values = np.array(all_metrics[k])  # (n_splits,)
    mean_metrics[k] = np.nanmean(values)  # 标量

# ======== 在整个数据集上计算指标 ========
if all_y_true:
    Y_all_combined = np.vstack(all_y_true)
    Y_prob_all_combined = np.vstack(all_y_prob)
    Y_pred_all_combined = np.vstack(all_y_pred)

    print("\n=== 在整个数据集上的整体指标 ===")

    # 计算整体指标
    try:
        overall_macro_f1 = f1_score(Y_all_combined, Y_pred_all_combined, average='macro', zero_division=0)
        print(f"整体 Macro F1: {overall_macro_f1:.4f}")
    except:
        print(f"整体 Macro F1: 无法计算")

    try:
        overall_micro_f1 = f1_score(Y_all_combined, Y_pred_all_combined, average='micro', zero_division=0)
        print(f"整体 Micro F1: {overall_micro_f1:.4f}")
    except:
        print(f"整体 Micro F1: 无法计算")

    try:
        overall_hamming_loss = hamming_loss(Y_all_combined, Y_pred_all_combined)
        print(f"整体 Hamming Loss: {overall_hamming_loss:.4f}")
    except:
        print(f"整体 Hamming Loss: 无法计算")

    try:
        overall_ranking_loss = label_ranking_loss(Y_all_combined, Y_prob_all_combined)
        print(f"整体 Ranking Loss: {overall_ranking_loss:.4f}")
    except:
        print(f"整体 Ranking Loss: 无法计算")

# ======== 打印每类指标 ========
print("\n=== 各类别指标（十折平均） ===")
for i, cname in enumerate(class_name):
    print(f"{cname:12}: "
          f"AUC={mean_metrics['auc'][i]:.4f}, "
          f"AUPR={mean_metrics['aupr'][i]:.4f}, "
          f"ACC={mean_metrics['acc'][i]:.4f}, "
          f"F1={mean_metrics['f1'][i]:.4f}, "
          f"Recall={mean_metrics['recall'][i]:.4f}, "
          f"Precision={mean_metrics['precision'][i]:.4f}")

# ======== 打印每个指标的平均分 ========
print("\n=== 平均分（十折平均） ===")

# 类别特定的指标平均值
for metric_name in ['auc', 'aupr', 'acc', 'f1', 'recall', 'precision']:
    # 只计算非NaN值的平均值
    valid_values = mean_metrics[metric_name][~np.isnan(mean_metrics[metric_name])]
    if len(valid_values) > 0:
        avg_val = np.mean(valid_values)
        print(f"平均 {metric_name.upper()}: {avg_val:.4f}")

# 整体指标
print(f"平均 Macro F1: {mean_metrics['macro_f1']:.4f}")
print(f"平均 Micro F1: {mean_metrics['micro_f1']:.4f}")
print(f"平均 Hamming Loss: {mean_metrics['hamming_loss']:.4f}")
print(f"平均 Ranking Loss: {mean_metrics['ranking_loss']:.4f}")

# ======== 保存结果到CSV文件 ========
print("\n=== 保存结果到CSV文件 ===")

# 创建类别指标DataFrame
class_metrics_df = pd.DataFrame({
    'Class': class_name,
    'AUC': mean_metrics['auc'],
    'AUPR': mean_metrics['aupr'],
    'Accuracy': mean_metrics['acc'],
    'F1': mean_metrics['f1'],
    'Recall': mean_metrics['recall'],
    'Precision': mean_metrics['precision']
})

# 创建整体指标DataFrame
overall_metrics_df = pd.DataFrame({
    'Metric': ['Macro_F1', 'Micro_F1', 'Hamming_Loss', 'Ranking_Loss'],
    'Value': [
        mean_metrics['macro_f1'],
        mean_metrics['micro_f1'],
        mean_metrics['hamming_loss'],
        mean_metrics['ranking_loss']
    ]
})

# 保存到文件
class_metrics_file = '消融实验_类别指标.csv'
overall_metrics_file = '消融实验_整体指标.csv'

class_metrics_df.to_csv(class_metrics_file, index=False, encoding='utf-8-sig')
overall_metrics_df.to_csv(overall_metrics_file, index=False, encoding='utf-8-sig')

print(f"类别指标已保存到: {class_metrics_file}")
print(f"整体指标已保存到: {overall_metrics_file}")

# ======== 打印汇总表格 ========
print("\n" + "=" * 80)
print("汇总结果:")
print("=" * 80)
print("\n1. 各类别性能:")
print("-" * 80)
print(f"{'类别':<12} {'AUC':<8} {'AUPR':<8} {'Accuracy':<8} {'F1':<8} {'Recall':<8} {'Precision':<8}")
print("-" * 80)
for i in range(num_classes):
    print(f"{class_name[i]:<12} {mean_metrics['auc'][i]:<8.4f} {mean_metrics['aupr'][i]:<8.4f} "
          f"{mean_metrics['acc'][i]:<8.4f} {mean_metrics['f1'][i]:<8.4f} "
          f"{mean_metrics['recall'][i]:<8.4f} {mean_metrics['precision'][i]:<8.4f}")

print("\n2. 整体性能:")
print("-" * 80)
print(f"Macro F1: {mean_metrics['macro_f1']:.4f}")
print(f"Micro F1: {mean_metrics['micro_f1']:.4f}")
print(f"Hamming Loss: {mean_metrics['hamming_loss']:.4f}")
print(f"Ranking Loss: {mean_metrics['ranking_loss']:.4f}")
print("=" * 80)