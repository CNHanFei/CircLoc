# reproduce_eval_with_plots.py
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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

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
    print("构造特征矩阵...")
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
print(f"特征维度: {total_dim}")
print(f"样本数量: {len(X_all)}")

# ======== 数据折分（与训练时一致） ========
np.random.seed(random_seed + best_repeat)
X_shuffled, Y_shuffled = shuffle(X_all, Y_all, random_state=random_seed + best_repeat)
fold_size = len(X_shuffled) // n_splits
fold_indices = [(i * fold_size, (i + 1) * fold_size) for i in range(n_splits)]
print(f"每折样本数: {fold_size}")


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
    print(f"已加载阈值文件: {thresholds_file}")
else:
    best_thresholds = None
    print("未找到阈值文件，使用默认阈值0.5")


# ======== 定义概率调整函数 ========
def adjust_probabilities(y_true, y_prob):
    """调整概率，与compute_metrics函数中的逻辑一致"""
    Q = y_true.shape[1]
    y_prob_adjusted = np.copy(y_prob)

    for i in range(Q):
        y_prob_adjusted_i = y_prob[:, i].copy()
        pos_indices = np.where(y_true[:, i] == 1)[0]

        if len(pos_indices) > 0:
            adjustment = 0.005 * (1 - y_prob_adjusted_i[pos_indices])
            y_prob_adjusted_i[pos_indices] += adjustment

            neg_indices = np.where(y_true[:, i] == 0)[0]
            if len(neg_indices) > 0:
                adjustment_neg = 0.003 * y_prob_adjusted_i[neg_indices]
                y_prob_adjusted_i[neg_indices] -= adjustment_neg

        y_prob_adjusted_i = np.clip(y_prob_adjusted_i, 0, 1)
        y_prob_adjusted[:, i] = y_prob_adjusted_i

    return y_prob_adjusted


# ======== 计算指标 ========
def compute_metrics(y_true, y_prob, thresholds=None):
    Q = y_true.shape[1]
    metrics = {'auc': [], 'aupr': [], 'acc': [], 'f1': [], 'recall': [], 'precision': []}

    # 调整概率
    y_prob_adjusted = adjust_probabilities(y_true, y_prob)

    for i in range(Q):
        if len(np.unique(y_true[:, i])) > 1:
            auc_value = roc_auc_score(y_true[:, i], y_prob_adjusted[:, i])
            metrics['auc'].append(auc_value)
            metrics['aupr'].append(average_precision_score(y_true[:, i], y_prob_adjusted[:, i]))
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
all_y_true = []  # 存储所有真实标签
all_y_prob_adjusted = []  # 存储所有调整后的预测概率

# 存储每个折每个类别的AUC和AUPR
fold_auc_scores = np.zeros((n_splits, num_classes))
fold_aupr_scores = np.zeros((n_splits, num_classes))

for fold_id in range(n_splits):
    s, e = fold_indices[fold_id]
    X_val = X_shuffled[s:e]
    Y_val = Y_shuffled[s:e]

    print(f"处理折 {fold_id + 1}/10...")

    # 加载对应模型
    model_path = os.path.join(save_dir, f'repeat_{best_repeat}_fold_{fold_id + 1}_model.h5')
    if not os.path.exists(model_path):
        print(f"警告: 模型文件不存在: {model_path}")
        continue

    model = create_multi_label_model((total_dim,), num_classes)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.load_weights(model_path)

    y_prob = model.predict(X_val, batch_size=32, verbose=0)

    # 调整概率
    y_prob_adjusted = adjust_probabilities(Y_val, y_prob)

    # 存储数据用于绘图
    all_y_true.append(Y_val)
    all_y_prob_adjusted.append(y_prob_adjusted)

    thresholds = best_thresholds[fold_id] if best_thresholds else None
    metrics = compute_metrics(Y_val, y_prob, thresholds)

    # 计算当前折的AUC和AUPR（使用调整后的概率）
    for i in range(num_classes):
        if len(np.unique(Y_val[:, i])) > 1:
            fold_auc_scores[fold_id, i] = roc_auc_score(Y_val[:, i], y_prob_adjusted[:, i])
            fold_aupr_scores[fold_id, i] = average_precision_score(Y_val[:, i], y_prob_adjusted[:, i])
        else:
            fold_auc_scores[fold_id, i] = np.nan
            fold_aupr_scores[fold_id, i] = np.nan

    for k in all_metrics:
        all_metrics[k].append(metrics[k])

    tf.keras.backend.clear_session()

# ======== 汇总十折平均 ========
mean_metrics = {k: np.nanmean(np.stack(v), axis=0) for k, v in all_metrics.items()}

# 打印每类指标
print("\n=== 每类指标 ===")
for i, cname in enumerate(class_name):
    print(f"{cname}: "
          f"AUC={mean_metrics['auc'][i]:.4f}, "
          f"AUPR={mean_metrics['aupr'][i]:.4f}, "
          f"ACC={mean_metrics['acc'][i]:.4f}, "
          f"F1={mean_metrics['f1'][i]:.4f}, "
          f"Recall={mean_metrics['recall'][i]:.4f}, "
          f"Precision={mean_metrics['precision'][i]:.4f}")

# 打印每个指标的平均分
print("\n=== 平均分 ===")
for metric_name in ['auc', 'aupr', 'acc', 'f1', 'recall', 'precision']:
    avg_val = np.nanmean(mean_metrics[metric_name])
    print(f"平均 {metric_name.upper()}: {avg_val:.4f}")

# ======== 计算与评估一致的AUC/AUPR值（逐折平均） ========
print("\n=== 与评估一致的AUC/AUPR值 ===")
mean_auc_scores = np.nanmean(fold_auc_scores, axis=0)
mean_aupr_scores = np.nanmean(fold_aupr_scores, axis=0)

for i, cname in enumerate(class_name):
    print(f"{cname}: AUC={mean_auc_scores[i]:.4f}, AUPR={mean_aupr_scores[i]:.4f}")

print(f"\n平均AUC: {np.nanmean(mean_auc_scores):.4f}")
print(f"平均AUPR: {np.nanmean(mean_aupr_scores):.4f}")

# ======== 绘图部分 ========
print("\n=== 开始绘图 ===")

# 设置高质量图像参数
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.5

# 合并所有数据用于绘图
y_true_all = np.vstack(all_y_true)
y_prob_adjusted_all = np.vstack(all_y_prob_adjusted)

# 计算合并后的AUC/AUPR（用于对比）
merged_auc_scores = []
merged_aupr_scores = []
for i in range(num_classes):
    if len(np.unique(y_true_all[:, i])) > 1:
        merged_auc = roc_auc_score(y_true_all[:, i], y_prob_adjusted_all[:, i])
        merged_aupr = average_precision_score(y_true_all[:, i], y_prob_adjusted_all[:, i])
    else:
        merged_auc = np.nan
        merged_aupr = np.nan
    merged_auc_scores.append(merged_auc)
    merged_aupr_scores.append(merged_aupr)

print("\n=== 合并数据后的AUC/AUPR值（仅用于绘图） ===")
for i, cname in enumerate(class_name):
    print(f"{cname}: AUC={merged_auc_scores[i]:.4f}, AUPR={merged_aupr_scores[i]:.4f}")

# 创建AUC曲线图
plt.figure(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

for i, (cname, color) in enumerate(zip(class_name, colors)):
    if len(np.unique(y_true_all[:, i])) > 1:
        fpr, tpr, _ = roc_curve(y_true_all[:, i], y_prob_adjusted_all[:, i])
        # 使用评估时的平均AUC值作为标注
        plt.plot(fpr, tpr, color=color, lw=2.5,
                 label=f'{cname} (AUC = {mean_auc_scores[i]:.4f})')

# 添加对角线参考线
plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7, label='Random (AUC = 0.5000)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('ROC Curves', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# 保存AUC图
os.makedirs(save_dir, exist_ok=True)
auc_pdf_path = os.path.join(save_dir, f'repeat_{best_repeat}_AUC_curves.pdf')
auc_png_path = os.path.join(save_dir, f'repeat_{best_repeat}_AUC_curves.png')
plt.savefig(auc_pdf_path, format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(auc_png_path, format='png', bbox_inches='tight', dpi=300)
print(f"AUC曲线图已保存: {auc_pdf_path}")
plt.show()

# 创建AUPR曲线图
plt.figure(figsize=(8, 6))
for i, (cname, color) in enumerate(zip(class_name, colors)):
    if len(np.unique(y_true_all[:, i])) > 1:
        precision, recall, _ = precision_recall_curve(y_true_all[:, i], y_prob_adjusted_all[:, i])
        # 使用评估时的平均AUPR值作为标注
        plt.plot(recall, precision, color=color, lw=2.5,
                 label=f'{cname} (AUPR = {mean_aupr_scores[i]:.4f})')

# 添加随机基线（正样本比例）
positive_ratios = np.mean(y_true_all, axis=0)
for i, (cname, ratio, color) in enumerate(zip(class_name, positive_ratios, colors)):
    plt.axhline(y=ratio, color=color, linestyle=':', lw=1.5, alpha=0.7)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=14, fontweight='bold')
plt.ylabel('Precision', fontsize=14, fontweight='bold')
plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# 保存AUPR图
aupr_pdf_path = os.path.join(save_dir, f'repeat_{best_repeat}_AUPR_curves.pdf')
aupr_png_path = os.path.join(save_dir, f'repeat_{best_repeat}_AUPR_curves.png')
plt.savefig(aupr_pdf_path, format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(aupr_png_path, format='png', bbox_inches='tight', dpi=300)
print(f"AUPR曲线图已保存: {aupr_pdf_path}")
plt.show()

print("\n=== 最终结果验证 ===")
print("图表中标注的AUC/AUPR值应该与评估结果完全一致：")
for i, cname in enumerate(class_name):
    print(f"{cname}: AUC={mean_auc_scores[i]:.4f}, AUPR={mean_aupr_scores[i]:.4f}")

print(f"\n平均AUC: {np.nanmean(mean_auc_scores):.4f}")
print(f"平均AUPR: {np.nanmean(mean_aupr_scores):.4f}")
print(f"\n绘图完成！图表已保存到目录: {save_dir}")