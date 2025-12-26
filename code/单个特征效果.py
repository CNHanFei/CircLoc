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

# ======== 数据加载 ========
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

# ======== 每个特征的索引 ========
indices = {}
start = 0
order = ['kmer', 'rckmer', 'seq', 'dis', 'drug', 'miRNA', 'protein', 'ernie']
for k in order:
    indices[k] = (start, start + feat_dims[k])
    start += feat_dims[k]


# ======== 生成所有消融组合 ========
def generate_ablation_schemes():
    """生成所有可能的消融组合，包括单个特征和特征组合"""
    features = list(indices.keys())
    schemes = {}

    # 1. 基线：所有特征 (no ablation)
    schemes['all_features'] = features.copy()

    # 2. 消融单个特征（保留其他7个）
    for feature in features:
        scheme_name = f'only_without_{feature}'
        schemes[scheme_name] = [f for f in features if f != feature]

    # 3. 只使用单个特征
    for feature in features:
        scheme_name = f'only_{feature}'
        schemes[scheme_name] = [feature]

    # 4. 生成所有可能的特征组合（可选，这里限制最多3个特征组合）
    # 如果您想要评估所有特征组合，可以取消注释以下代码
    """
    for r in range(1, len(features) + 1):
        for combo in itertools.combinations(features, r):
            if 1 < len(combo) <= 3:  # 只考虑2-3个特征的组合
                scheme_name = f'combo_{"_".join(combo)}'
                schemes[scheme_name] = list(combo)
    """

    return schemes


# 使用所有组合
ablation_schemes = generate_ablation_schemes()
print(f"总共有 {len(ablation_schemes)} 种消融方案:")

# 按类别打印方案
categories = {
    '基线': ['all_features'],
    '消融单个特征': [s for s in ablation_schemes.keys() if s.startswith('only_without_')],
    '仅单个特征': [s for s in ablation_schemes.keys() if s.startswith('only_') and not s.startswith('only_without_')]
}

for category, schemes in categories.items():
    print(f"\n{category} ({len(schemes)}种):")
    for scheme in schemes:
        features = ablation_schemes[scheme]
        print(f"  {scheme:30s}: 使用 {len(features)} 个特征: {features}")

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
for scheme_name, features_to_keep in ablation_schemes.items():
    print(f"\n{'=' * 60}")
    print(f"消融方案: {scheme_name}")
    print(f"保留的特征: {features_to_keep}")
    print(f"{'=' * 60}")

    all_per_class_metrics = {k: [] for k in ['auc', 'aupr', 'acc', 'f1', 'recall', 'precision']}
    all_overall_metrics = {k: [] for k in ['hamming_loss', 'ranking_loss', 'f1_micro', 'f1_macro']}

    for fold_id in range(n_splits):
        s, e = fold_indices[fold_id]
        X_val = X_shuffled[s:e].copy()
        Y_val = Y_shuffled[s:e]

        # 创建掩码：只保留指定的特征，其他特征置零
        mask = np.zeros(total_dim)
        for feature in features_to_keep:
            si, ei = indices[feature]
            mask[si:ei] = 1.0

        # 应用掩码
        X_val = X_val * mask

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
        'overall': mean_overall_metrics,
        'features_used': features_to_keep,
        'num_features': len(features_to_keep)
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
        'features_used': result['features_used'],
        'num_features': result['num_features'],
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
    num_features = result['num_features']

    # 计算平均AUC和平均F1
    avg_auc = np.nanmean(per_class['auc'])
    avg_f1 = np.nanmean(per_class['f1'])

    summary_data.append({
        '消融方案': scheme_name,
        '特征数量': num_features,
        '平均AUC': f"{avg_auc:.4f}",
        '平均F1': f"{avg_f1:.4f}",
        '宏观F1': f"{overall['f1_macro']:.4f}",
        '汉明损失': f"{overall['hamming_loss']:.4f}",
        '排名损失': f"{overall['ranking_loss']:.4f}"
    })

# 转换为DataFrame并按特征数量排序
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('特征数量', ascending=False)
print("\n", summary_df.to_string(index=False))

# 保存为Excel文件
excel_file = os.path.join(save_dir, f'repeat_{best_repeat}_ablation_summary.xlsx')
summary_df.to_excel(excel_file, index=False)
print(f"\n总结表格已保存到: {excel_file}")
# ======== 在保存所有结果到文件之后，添加以下代码 ========

# 创建汇总Excel文件
summary_excel_file = os.path.join(save_dir, f'repeat_{best_repeat}_all_ablation_detailed_results.xlsx')

with pd.ExcelWriter(summary_excel_file, engine='openpyxl') as writer:
    # 为每个消融方案创建一个sheet
    for scheme_name, result in all_results.items():
        print(f"正在写入方案 '{scheme_name}' 到Excel...")

        per_class = result['per_class']
        overall = result['overall']

        # 创建每类指标DataFrame
        df_per_class = pd.DataFrame()
        df_per_class['类别'] = class_name
        df_per_class['AUC'] = per_class['auc']
        df_per_class['AUPR'] = per_class['aupr']
        df_per_class['ACC'] = per_class['acc']
        df_per_class['F1'] = per_class['f1']
        df_per_class['RECALL'] = per_class['recall']
        df_per_class['PRECISION'] = per_class['precision']

        # 创建方案信息DataFrame
        scheme_info_data = [
            ['消融方案', scheme_name],
            ['使用特征数', result['num_features']],
            ['特征列表', ', '.join(result['features_used'])],
            ['', ''],
            ['每类指标:', '']
        ]
        df_scheme_info = pd.DataFrame(scheme_info_data, columns=['项目', '值'])

        # 写入方案信息
        start_row = 0
        df_scheme_info.to_excel(writer, sheet_name=scheme_name[:31],  # sheet名称最多31个字符
                                startrow=start_row, index=False)
        start_row += len(df_scheme_info) + 2

        # 写入每类指标
        df_per_class.to_excel(writer, sheet_name=scheme_name[:31],
                              startrow=start_row, index=False)
        start_row += len(df_per_class) + 3

        # 创建平均指标DataFrame
        avg_metrics_data = [
            ['平均AUC', np.nanmean(per_class['auc'])],
            ['平均AUPR', np.nanmean(per_class['aupr'])],
            ['平均ACC', np.nanmean(per_class['acc'])],
            ['平均F1', np.nanmean(per_class['f1'])],
            ['平均RECALL', np.nanmean(per_class['recall'])],
            ['平均PRECISION', np.nanmean(per_class['precision'])]
        ]
        df_avg_metrics = pd.DataFrame(avg_metrics_data, columns=['平均指标', '值'])

        # 写入平均指标
        df_avg_metrics.to_excel(writer, sheet_name=scheme_name[:31],
                                startrow=start_row, index=False)
        start_row += len(df_avg_metrics) + 3

        # 创建整体指标DataFrame
        overall_metrics_data = [
            ['汉明损失', overall['hamming_loss']],
            ['排名损失', overall['ranking_loss']],
            ['微观F1', overall['f1_micro']],
            ['宏观F1', overall['f1_macro']]
        ]
        df_overall_metrics = pd.DataFrame(overall_metrics_data, columns=['整体指标', '值'])

        # 写入整体指标
        df_overall_metrics.to_excel(writer, sheet_name=scheme_name[:31],
                                    startrow=start_row, index=False)

print(f"\n所有方案的详细结果已保存到: {summary_excel_file}")

# 创建索引sheet
with pd.ExcelWriter(summary_excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    # 创建索引表格
    index_data = []
    for scheme_name, result in all_results.items():
        index_data.append({
            '消融方案': scheme_name,
            '特征数量': result['num_features'],
            '使用特征': ', '.join(result['features_used']),
            '平均AUC': f"{np.nanmean(result['per_class']['auc']):.4f}",
            '平均F1': f"{np.nanmean(result['per_class']['f1']):.4f}",
            '宏观F1': f"{result['overall']['f1_macro']:.4f}",
            '对应Sheet': scheme_name[:31]
        })

    df_index = pd.DataFrame(index_data)
    df_index.to_excel(writer, sheet_name='方案索引', index=False)


# ======== 生成特征重要性分析 ========
print(f"\n{'=' * 80}")
print("特征重要性分析")
print(f"{'=' * 80}")

# 提取基线性能
baseline_perf = all_results['all_features']
baseline_auc = np.nanmean(baseline_perf['per_class']['auc'])
baseline_f1 = np.nanmean(baseline_perf['per_class']['f1'])

# 分析单个特征的重要性
print("\n单个特征重要性（与基线比较）:")
print(f"{'特征':10s} {'特征数':8s} {'平均AUC':10s} {'AUC变化':10s} {'平均F1':10s} {'F1变化':10s}")
print("-" * 60)

for scheme_name, result in all_results.items():
    if scheme_name.startswith('only_without_'):
        feature_name = scheme_name.replace('only_without_', '')
        avg_auc = np.nanmean(result['per_class']['auc'])
        avg_f1 = np.nanmean(result['per_class']['f1'])

        auc_change = avg_auc - baseline_auc
        f1_change = avg_f1 - baseline_f1

        print(f"{feature_name:10s} {result['num_features']:8d} "
              f"{avg_auc:.4f}      {auc_change:+.4f}     "
              f"{avg_f1:.4f}      {f1_change:+.4f}")

# 分析单独使用每个特征的性能
print("\n\n单独使用每个特征的性能:")
print(f"{'特征':10s} {'平均AUC':10s} {'平均F1':10s} {'宏观F1':10s}")
print("-" * 40)

for feature in order:
    scheme_name = f'only_{feature}'
    if scheme_name in all_results:
        result = all_results[scheme_name]
        avg_auc = np.nanmean(result['per_class']['auc'])
        avg_f1 = np.nanmean(result['per_class']['f1'])
        f1_macro = result['overall']['f1_macro']

        print(f"{feature:10s} {avg_auc:.4f}      {avg_f1:.4f}      {f1_macro:.4f}")
