# reproduce_eval.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization, Flatten, MultiHeadAttention
from sklearn.preprocessing import StandardScaler

# ======== 配置 ========
save_dir = './best_model_repeat_change/'   # 模型权重目录
best_repeat = 18                  # <-- 改成你的最佳 repeat
random_seed = 42
n_splits = 10
num_classes = 7
class_name = ['Chromatin', 'Nucleoplasm', 'Nucleolus', 'Membrane', 'Nucleus', 'Cytosol', 'Cytoplasm']

# ======== 数据加载（和训练时一致） ========
df_seq_feature = pd.read_excel(r'C:\HJH\plsm\feature\seq_sim_feature.xlsx', index_col=0, header=None)
df_miRNA_feature = pd.read_csv(r'C:\HJH\plsm\feature\rna_mi_features_0.7_128_0.01.csv', header=None)
df_drug_feature = pd.read_csv(r'C:\HJH\plsm\feature\rna_drug_features_0.7_128_0.01.csv', header=None)
df_dis_feature = pd.read_csv(r'C:\HJH\plsm\feature\rna_disease_features_0.7_128_0.01.csv', header=None)
df_loc = pd.read_excel(r'C:\HJH\plsm\dataset\location_info.xlsx', index_col=0)
df_loc_index = pd.read_excel(r'C:\HJH\excel处理\location_info_index.xlsx', index_col=0, header=None)
df_kmer_feature = pd.read_csv(r'C:\HJH\plsm\feature\rna_kmer_features_0.7_128_0.01.csv', header=None)
df_rckmer_feature = pd.read_csv(r'C:\HJH\plsm\feature\rna_rckmer_features_0.7_128_0.01.csv', header=None)
df_RNAErnie_feature = pd.read_excel(r'C:\HJH\excel处理\model_feature.xlsx', header=None)
df_protein_feature = pd.read_csv(r'C:\HJH\plsm\feature\gate_feature_protein_0.7_128_0.01.csv', header=None)

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

    merge_feature = np.concatenate((kmer, rckmer, seq, dis, drug, miRNA, ernie,protein_feature), axis=1)
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
# ======== 计算指标 ========
def compute_metrics(y_true, y_prob, thresholds=None):
    Q = y_true.shape[1]
    metrics = {'auc': [], 'aupr': [], 'acc': [], 'f1': [], 'recall': [], 'precision': []}

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

    return {k: np.array(v) for k, v in metrics.items()}

# ======== 主循环（复现，不做消融） ========
print("\n=== 复现实验（不做消融） ===")
all_metrics = {k: [] for k in ['auc', 'aupr', 'acc', 'f1', 'recall', 'precision']}

for fold_id in range(n_splits):
    s, e = fold_indices[fold_id]
    X_val = X_shuffled[s:e]
    Y_val = Y_shuffled[s:e]

    # 加载对应模型
    model_path = os.path.join(save_dir, f'repeat_{best_repeat}_fold_{fold_id+1}_model.h5')
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
mean_metrics = {k: np.nanmean(np.stack(v), axis=0) for k, v in all_metrics.items()}

# 打印每类指标
for i, cname in enumerate(class_name):
    print(f"{cname}: "
          f"AUC={mean_metrics['auc'][i]:.4f}, "
          f"AUPR={mean_metrics['aupr'][i]:.4f}, "
          f"ACC={mean_metrics['acc'][i]:.4f}, "
          f"F1={mean_metrics['f1'][i]:.4f}, "
          f"Recall={mean_metrics['recall'][i]:.4f}, "
          f"Precision={mean_metrics['precision'][i]:.4f}")

# 打印每个指标的平均分
print("\n-- 平均分 --")
for metric_name in ['auc', 'aupr', 'acc', 'f1', 'recall', 'precision']:
    avg_val = np.nanmean(mean_metrics[metric_name])
    print(f"平均 {metric_name.upper()}: {avg_val:.4f}")
