# ablation_eval.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, \
    recall_score, hamming_loss, label_ranking_loss
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization, Flatten, \
    MultiHeadAttention
from sklearn.preprocessing import StandardScaler
import itertools

# ======== 配置 ========
save_dir = './best_model_repeat_change/'  # 模型权重目录
best_repeat = 18  # <-- 修改为你的最佳 repeat
random_seed = 42
n_splits = 10
num_classes = 7
class_name = ['Chromatin', 'Nucleoplasm', 'Nucleolus', 'Membrane', 'Nucleus', 'Cytosol', 'Cytoplasm']

# ======== 数据加载（和 gate消融.py 一致） ========
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
    protein = df_protein_feature.values
    merge_feature = np.concatenate((
        kmer, rckmer,
        seq,
        dis, drug, miRNA, protein,
        ernie
    ), axis=1)
    scaler = StandardScaler()
    merge_feature_scaled = scaler.fit_transform(merge_feature)

    loc_index = df_loc_index[1].tolist()
    select_row = np.array([v == 1 for v in loc_index])
    circRNA_loc = df_loc.values

    return merge_feature_scaled[select_row], circRNA_loc[select_row], {
        'kmer': kmer.shape[1],
        'rckmer': rckmer.shape[1],
        'seq': seq.shape[1],
        'dis': dis.shape[1],
        'drug': drug.shape[1],
        'miRNA': miRNA.shape[1],
        'protein': protein.shape[1],
        'ernie': ernie.shape[1]
    }


X_all, Y_all, feat_dims = build_merged_features()
total_dim = X_all.shape[1]

# ======== 分组索引（α、β、γ、η） ========
indices = {}
start = 0
order = ['kmer', 'rckmer', 'seq', 'dis', 'drug', 'miRNA', 'protein', 'ernie']
for k in order:
    indices[k] = (start, start + feat_dims[k])
    start += feat_dims[k]

group_indices = {
    'alpha': [indices['kmer'], indices['rckmer']],
    'beta': [indices['ernie']],
    'gamma': [indices['seq']],
    'eta': [indices['dis'], indices['drug'], indices['miRNA'], indices['protein']]
}


# ======== 生成所有消融组合（共15种） ========
def generate_ablation_schemes():
    """生成所有可能的消融组合"""
    groups = ['alpha', 'beta', 'gamma', 'eta']
    schemes = {}

    # 生成所有非空子集（从1个组到所有组）
    for r in range(1, len(groups) + 1):
        for combo in itertools.combinations(groups, r):
            scheme_name = f'no_{"_".join(combo)}' if r < len(groups) else 'no_all'
            schemes[scheme_name] = list(combo)

    return schemes


# 使用所有组合
ablation_schemes = generate_ablation_schemes()
print(f"总共有 {len(ablation_schemes)} 种消融方案:")
for name, groups in ablation_schemes.items():
    print(f"  {name}: 消融 {groups}")

# ======== 数据折分（与训练时一致） ========
np.random.seed(random_seed + best_repeat)
X_shuffled, Y_shuffled = shuffle(X_all, Y_all, random_state=random_seed + best_repeat)
fold_size = len(X_shuffled) // n_splits
fold_indices = [(i * fold_size, (i + 1) * fold_size) for i in range(n_splits)]


# ======== 模型结构 ========
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


# ======== 载入阈值 ========
thresholds_file = os.path.join(save_dir, f'repeat_{best_repeat}_thresholds.json')
if os.path.exists(thresholds_file):
    with open(thresholds_file, 'r') as f:
        best_thresholds = json.load(f)
else:
    best_thresholds = None


# ======== 计算指标 ========
def compute_metrics(y_true, y_prob, thresholds=None):
    Q = y_true.shape[1]

    per_class_metrics = {'auc': [], 'aupr': [], 'acc': [], 'f1': [], 'recall': [], 'precision': []}

    for i in range(Q):
        if len(np.unique(y_true[:, i])) > 1:
            per_class_metrics['auc'].append(roc_auc_score(y_true[:, i], y_prob[:, i]))
            per_class_metrics['aupr'].append(average_precision_score(y_true[:, i], y_prob[:, i]))
        else:
            per_class_metrics['auc'].append(np.nan)
            per_class_metrics['aupr'].append(np.nan)

        th = thresholds[i] if thresholds is not None else 0.5
        preds = (y_prob[:, i] > th).astype(int)

        per_class_metrics['acc'].append(accuracy_score(y_true[:, i], preds))
        per_class_metrics['f1'].append(f1_score(y_true[:, i], preds, zero_division=0))
        per_class_metrics['recall'].append(recall_score(y_true[:, i], preds, zero_division=0))
        per_class_metrics['precision'].append(precision_score(y_true[:, i], preds, zero_division=0))

    per_class_metrics = {k: np.array(v) for k, v in per_class_metrics.items()}

    if thresholds is not None:
        y_pred = np.zeros_like(y_prob)
        for i in range(Q):
            th = thresholds[i] if thresholds is not None else 0.5
            y_pred[:, i] = (y_prob[:, i] > th).astype(int)
    else:
        y_pred = (y_prob > 0.5).astype(int)

    hamming = hamming_loss(y_true, y_pred)
    ranking = label_ranking_loss(y_true, y_prob)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    overall_metrics = {
        'hamming_loss': hamming,
        'ranking_loss': ranking,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro
    }

    return per_class_metrics, overall_metrics


# ======== 用于存储所有结果的字典 ========
all_results = {}

# ======== 主循环 ========
for scheme_name, groups_to_ablate in ablation_schemes.items():
    print(f"\n{'=' * 60}")
    print(f"消融方案: {scheme_name} (消融组: {groups_to_ablate})")
    print(f"{'=' * 60}")

    all_per_class_metrics = {k: [] for k in ['auc', 'aupr', 'acc', 'f1', 'recall', 'precision']}
    all_overall_metrics = {k: [] for k in ['hamming_loss', 'ranking_loss', 'f1_micro', 'f1_macro']}

    for fold_id in range(n_splits):
        s, e = fold_indices[fold_id]
        X_val = X_shuffled[s:e].copy()
        Y_val = Y_shuffled[s:e]

        # 屏蔽所有指定的组特征
        for group in groups_to_ablate:
            for (si, ei) in group_indices[group]:
                X_val[:, si:ei] = 0.0

        # 加载模型
        model_path = os.path.join(save_dir, f'repeat_{best_repeat}_fold_{fold_id + 1}_model.h5')
        model = create_multi_label_model((total_dim,), num_classes)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.load_weights(model_path)

        y_prob = model.predict(X_val, batch_size=32, verbose=0)
        thresholds = best_thresholds[fold_id] if best_thresholds else None

        per_class_metrics, overall_metrics = compute_metrics(Y_val, y_prob, thresholds)

        for k in all_per_class_metrics:
            all_per_class_metrics[k].append(per_class_metrics[k])

        for k in all_overall_metrics:
            all_overall_metrics[k].append(overall_metrics[k])

        tf.keras.backend.clear_session()

    # 计算十折平均
    mean_per_class_metrics = {k: np.nanmean(np.stack(v), axis=0) for k, v in all_per_class_metrics.items()}
    mean_overall_metrics = {k: np.mean(v) for k, v in all_overall_metrics.items()}

    # 存储结果
    all_results[scheme_name] = {
        'per_class': mean_per_class_metrics,
        'overall': mean_overall_metrics
    }

    # 打印每类指标
    print("\n每类指标:")
    for i, cname in enumerate(class_name):
        print(f"  {cname:12s}: "
              f"AUC={mean_per_class_metrics['auc'][i]:.4f}, "
              f"AUPR={mean_per_class_metrics['aupr'][i]:.4f}, "
              f"ACC={mean_per_class_metrics['acc'][i]:.4f}, "
              f"F1={mean_per_class_metrics['f1'][i]:.4f}")

    # 打印平均指标
    print("\n平均指标:")
    for metric_name in ['auc', 'aupr', 'acc', 'f1', 'recall', 'precision']:
        avg_val = np.nanmean(mean_per_class_metrics[metric_name])
        print(f"  平均 {metric_name.upper():9s}: {avg_val:.4f}")

    print("\n整体指标:")
    print(f"  汉明损失: {mean_overall_metrics['hamming_loss']:.4f}")
    print(f"  排名损失: {mean_overall_metrics['ranking_loss']:.4f}")
    print(f"  微观F1: {mean_overall_metrics['f1_micro']:.4f}")
    print(f"  宏观F1: {mean_overall_metrics['f1_macro']:.4f}")

# ======== 保存所有结果到文件 ========
results_file = os.path.join(save_dir, f'repeat_{best_repeat}_ablation_results.json')

# 将numpy数组转换为Python列表以便JSON序列化
results_for_save = {}
for scheme_name, result in all_results.items():
    results_for_save[scheme_name] = {
        'per_class': {k: v.tolist() if isinstance(v, np.ndarray) else v
                      for k, v in result['per_class'].items()},
        'overall': result['overall']
    }

with open(results_file, 'w') as f:
    json.dump(results_for_save, f, indent=2)

print(f"\n所有消融结果已保存到: {results_file}")

# ======== 生成总结表格 ========
print(f"\n{'=' * 80}")
print("消融实验总结表格")
print(f"{'=' * 80}")

# 创建汇总表格
summary_data = []
for scheme_name, result in all_results.items():
    per_class = result['per_class']
    overall = result['overall']

    # 计算平均AUC和平均F1
    avg_auc = np.nanmean(per_class['auc'])
    avg_f1 = np.nanmean(per_class['f1'])

    summary_data.append({
        '消融方案': scheme_name,
        '消融组数': len(scheme_name.split('_')) - 1,  # 计算消融的组数
        '平均AUC': f"{avg_auc:.4f}",
        '平均F1': f"{avg_f1:.4f}",
        '宏观F1': f"{overall['f1_macro']:.4f}",
        '汉明损失': f"{overall['hamming_loss']:.4f}"
    })

# 转换为DataFrame并按消融组数排序
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('消融组数')
print("\n", summary_df.to_string(index=False))

# 保存为Excel文件
excel_file = os.path.join(save_dir, f'repeat_{best_repeat}_ablation_summary.xlsx')
summary_df.to_excel(excel_file, index=False)
print(f"\n总结表格已保存到: {excel_file}")