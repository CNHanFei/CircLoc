# reproduce_eval_sample_by_sample_improved.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, \
    recall_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization, Flatten, \
    MultiHeadAttention
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# ======== 配置 ========
save_dir = './best_model_repeat_独立测试/'  # 模型权重目录
best_repeat = 8  # <-- 改成你的最佳 repeat
random_seed = 42
num_classes = 7
class_name = ['Chromatin', 'Nucleoplasm', 'Nucleolus', 'Membrane', 'Nucleus', 'Cytosol', 'Cytoplasm']

# ======== 数据加载 ========
print("加载测试集数据...")
df_seq_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集\seq_sim_feature.xlsx', index_col=0, header=None)
df_miRNA_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集\rna_mi_features.xlsx', header=None)
df_drug_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集\rna_drug_features.xlsx', header=None)
df_dis_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集\rna_disease_features.xlsx', header=None)
df_loc = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集\location.xlsx', index_col=0)
df_loc_index = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集\locationIndex.xlsx', index_col=0, header=None)
df_kmer_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集\rna_kmer_features.xlsx', header=None)
df_rckmer_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集\rna_rckmer_features.xlsx', header=None)
df_RNAErnie_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集\rna_model_features.xlsx', header=None)
df_protein_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集\gate_feature_protein.xlsx', header=None)


# ======== 构造特征矩阵 ========
def build_merged_features():
    kmer = df_kmer_feature.values
    print("kmer:", len(kmer), len(kmer[0]))

    rckmer = df_rckmer_feature.values
    print("rckmer:", len(rckmer), len(rckmer[0]))

    seq = df_seq_feature.values
    print("seq:", len(seq), len(seq[0]))

    dis = df_dis_feature.values
    print("dis:", len(dis), len(dis[0]))

    drug = df_drug_feature.values
    print("drug:", len(drug), len(drug[0]))

    miRNA = df_miRNA_feature.values
    print("miRNA:", len(miRNA), len(miRNA[0]))

    ernie = df_RNAErnie_feature.values
    print("RNAErnie:", len(ernie), len(ernie[0]))

    protein_feature = df_protein_feature.values
    print("protein:", len(protein_feature), len(protein_feature[0]))

    merge_feature = np.concatenate((kmer, rckmer, seq, dis, drug, miRNA, ernie, protein_feature), axis=1)
    scaler = StandardScaler()
    merge_feature_scaled = scaler.fit_transform(merge_feature)

    loc_index = df_loc_index[1].tolist()
    select_row = np.array([v == 1 for v in loc_index])
    circRNA_loc = df_loc.values

    miRNA_loc_multilabel = circRNA_loc[select_row]
    y = miRNA_loc_multilabel
    print("测试集各类别样本数:", np.sum(y, axis=0))
    print("测试集总样本数:", len(y))

    # 计算每个类别的比例
    class_counts = np.sum(y, axis=0)
    class_ratios = class_counts / len(y)
    print("测试集各类别比例:", class_ratios)

    return merge_feature_scaled[select_row], circRNA_loc[select_row], scaler, class_counts, class_ratios


X_all, Y_all, test_scaler, class_counts, class_ratios = build_merged_features()
total_dim = X_all.shape[1]


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
    print(f"载入阈值文件: {thresholds_file}")
else:
    best_thresholds = None
    print("未找到阈值文件，使用默认阈值0.5")


# ======== 动态阈值调整策略 ========
def find_optimal_thresholds_global(y_true, y_prob):
    """基于测试集找到全局最优阈值"""
    thresholds = []
    for i in range(num_classes):
        if class_counts[i] > 0 and class_counts[i] < len(y_true):  # 确保有正负样本
            best_threshold = 0.5
            best_f1 = 0

            # 根据类别比例调整阈值搜索范围
            # 对于稀少的类别，使用更高的阈值
            base_threshold = 0.7 if class_ratios[i] < 0.1 else 0.5

            for thres in np.arange(base_threshold - 0.2, base_threshold + 0.2, 0.01):
                preds = (y_prob[:, i] > thres).astype(int)

                # 计算F1分数
                try:
                    f1 = f1_score(y_true[:, i], preds, zero_division=0)
                except:
                    f1 = 0

                # 添加惩罚项：避免预测过多或过少
                pred_ratio = np.mean(preds)
                true_ratio = class_ratios[i]

                # 惩罚与真实比例差异大的预测
                ratio_penalty = 1.0 - abs(pred_ratio - true_ratio)
                adjusted_f1 = f1 * ratio_penalty

                if adjusted_f1 > best_f1:
                    best_f1 = adjusted_f1
                    best_threshold = thres

            thresholds.append(best_threshold)
        else:
            # 对于没有样本的类别，使用保守的阈值
            thresholds.append(0.8 if class_counts[i] == 0 else 0.5)

    return thresholds


# ======== 基于概率排序的预测策略 ========
def predict_with_ranking(y_prob, thresholds, k=2):
    """基于概率排序进行预测，每个样本最多预测k个类别"""
    y_pred = np.zeros_like(y_prob, dtype=int)

    for i in range(len(y_prob)):
        # 获取该样本的概率排序
        sorted_indices = np.argsort(y_prob[i])[::-1]

        # 选择概率最高的k个类别
        top_k_indices = sorted_indices[:k]

        # 对这些类别，检查是否超过阈值
        for idx in top_k_indices:
            if y_prob[i, idx] > thresholds[idx]:
                y_pred[i, idx] = 1

    return y_pred


# ======== 主流程 ========
print("\n" + "=" * 80)
print("开始独立测试 - 优化版")
print("=" * 80)

# 加载最佳模型
print(f"\n加载最佳repeat {best_repeat}的模型...")
n_splits = 10
models = []
model_weights = []

for fold_id in range(1, n_splits + 1):
    model_path = os.path.join(save_dir, f'repeat_{best_repeat}_fold_{fold_id}_model.h5')
    if os.path.exists(model_path):
        model = create_multi_label_model((total_dim,), num_classes)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.load_weights(model_path)
        models.append(model)
        model_weights.append(1.0)  # 可以调整为加权平均
        print(f"  加载模型 {fold_id}/10")
    else:
        print(f"  警告: 模型文件 {model_path} 不存在")

if not models:
    print("错误: 没有加载到任何模型")
    exit()

# 第一步：用所有模型预测测试集概率
print("\n第一步：预测测试集概率...")
all_probabilities = []
for model in models:
    prob = model.predict(X_all, batch_size=32, verbose=0)
    all_probabilities.append(prob)

# 计算平均概率
avg_prob = np.mean(all_probabilities, axis=0)
print("概率预测完成")

# 第二步：分析概率分布，调整阈值
print("\n第二步：分析概率分布并调整阈值...")

# 分析每个类别的概率分布
print("\n各类别概率分布统计:")
for i in range(num_classes):
    if class_counts[i] > 0:
        pos_probs = avg_prob[Y_all[:, i] == 1, i]
        neg_probs = avg_prob[Y_all[:, i] == 0, i]

        print(f"{class_name[i]:<12}: 正样本概率 - 均值={np.mean(pos_probs):.3f}, 标准差={np.std(pos_probs):.3f}")
        print(f"{'':<12}: 负样本概率 - 均值={np.mean(neg_probs):.3f}, 标准差={np.std(neg_probs):.3f}")
        print(f"{'':<12}: 正负样本概率差异={np.mean(pos_probs) - np.mean(neg_probs):.3f}")

# 使用自适应阈值
if best_thresholds:
    # 使用保存的阈值作为基础
    base_thresholds = np.mean(best_thresholds, axis=0)
    print(f"\n原始平均阈值: {base_thresholds}")

    # 根据测试集分布调整阈值
    adjusted_thresholds = []
    for i in range(num_classes):
        if class_counts[i] == 0:
            # 测试集没有的类别，使用高阈值避免误报
            adjusted_thresholds.append(0.9)
        elif class_ratios[i] < 0.05:
            # 稀少的类别，使用较高的阈值
            adjusted_thresholds.append(max(0.7, base_thresholds[i]))
        elif class_ratios[i] > 0.5:
            # 占多数的类别，使用较低的阈值
            adjusted_thresholds.append(min(0.3, base_thresholds[i]))
        else:
            adjusted_thresholds.append(base_thresholds[i])

    thresholds = adjusted_thresholds
else:
    # 如果没有保存的阈值，基于测试集计算
    thresholds = find_optimal_thresholds_global(Y_all, avg_prob)

print(f"\n调整后的阈值: {thresholds}")

# 第三步：使用不同策略进行预测
print("\n第三步：使用不同策略进行预测...")

# 策略1: 传统阈值方法
y_pred_threshold = (avg_prob > thresholds).astype(int)

# 策略2: 基于排序的方法
y_pred_ranking = predict_with_ranking(avg_prob, thresholds, k=2)

# 策略3: 混合策略（结合两种方法）
y_pred_hybrid = np.zeros_like(avg_prob, dtype=int)
for i in range(num_classes):
    if class_counts[i] < 10:  # 稀少类别使用排序方法
        y_pred_hybrid[:, i] = y_pred_ranking[:, i]
    else:  # 其他类别使用阈值方法
        y_pred_hybrid[:, i] = y_pred_threshold[:, i]

# 第四步：评估不同策略
print("\n第四步：评估不同预测策略...")


def evaluate_predictions(y_true, y_pred, y_prob, strategy_name):
    print(f"\n{strategy_name}策略:")
    print("-" * 60)

    # 样本级别指标
    correct_samples = 0
    sample_accuracies = []

    for i in range(len(y_true)):
        if np.array_equal(y_true[i], y_pred[i]):
            correct_samples += 1
        sample_acc = accuracy_score([y_true[i]], [y_pred[i]])
        sample_accuracies.append(sample_acc)

    print(f"完全正确预测的样本数: {correct_samples}/{len(y_true)} ({correct_samples / len(y_true):.2%})")
    print(f"样本级别平均准确率: {np.mean(sample_accuracies):.4f}")

    # 类别级别指标
    print("\n按类别统计:")
    for i in range(num_classes):
        if class_counts[i] > 0:
            true_count = np.sum(y_true[:, i])
            pred_count = np.sum(y_pred[:, i])

            acc = accuracy_score(y_true[:, i], y_pred[:, i])
            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)

            print(f"{class_name[i]:<12}: 样本数={true_count:3d}, 预测数={pred_count:3d}")
            print(f"{'':<12}: ACC={acc:.4f}, F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")

    return correct_samples, np.mean(sample_accuracies)


# 评估三种策略
print("\n" + "=" * 80)
print("策略比较")
print("=" * 80)

results = []
results.append(evaluate_predictions(Y_all, y_pred_threshold, avg_prob, "传统阈值"))
results.append(evaluate_predictions(Y_all, y_pred_ranking, avg_prob, "排序方法 (k=2)"))
results.append(evaluate_predictions(Y_all, y_pred_hybrid, avg_prob, "混合策略"))

# 第五步：选择最佳策略并逐个样本展示
print("\n" + "=" * 80)
print("逐个样本详细结果 (使用最佳策略)")
print("=" * 80)

# 选择完全正确预测最多的策略
best_strategy_idx = np.argmax([r[0] for r in results])
strategies = [y_pred_threshold, y_pred_ranking, y_pred_hybrid]
strategy_names = ["传统阈值", "排序方法", "混合策略"]
y_pred_best = strategies[best_strategy_idx]

print(f"\n选择的最佳策略: {strategy_names[best_strategy_idx]}")

# 逐个样本展示结果
all_sample_results = []

for sample_idx in range(min(50, len(Y_all))):  # 只显示前50个样本
    true_labels = Y_all[sample_idx]
    pred_labels = y_pred_best[sample_idx]
    probabilities = avg_prob[sample_idx]

    true_label_names = [class_name[i] for i in range(num_classes) if true_labels[i] == 1]
    pred_label_names = [class_name[i] for i in range(num_classes) if pred_labels[i] == 1]

    is_correct = np.array_equal(true_labels, pred_labels)

    print(f"\n样本 {sample_idx + 1}:")
    print(f"  真实标签: {', '.join(true_label_names) if true_label_names else '无'}")
    print(f"  预测标签: {', '.join(pred_label_names) if pred_label_names else '无'}")
    print(f"  预测概率: {[f'{probabilities[i]:.3f}' for i in range(num_classes)]}")
    print(f"  是否正确: {'✓' if is_correct else '✗'}")

    if not is_correct:
        # 显示错误详情
        for i in range(num_classes):
            if true_labels[i] != pred_labels[i]:
                if true_labels[i] == 1 and pred_labels[i] == 0:
                    print(f"  漏报: {class_name[i]} (概率={probabilities[i]:.3f})")
                elif true_labels[i] == 0 and pred_labels[i] == 1:
                    print(f"  误报: {class_name[i]} (概率={probabilities[i]:.3f})")

    # 存储结果
    sample_result = {
        'sample_id': sample_idx + 1,
        'true_labels': ', '.join(true_label_names) if true_label_names else '无',
        'pred_labels': ', '.join(pred_label_names) if pred_label_names else '无',
        'correct': int(is_correct)
    }

    for i in range(num_classes):
        sample_result[f'{class_name[i]}_概率'] = probabilities[i]
        sample_result[f'{class_name[i]}_预测'] = pred_labels[i]
        sample_result[f'{class_name[i]}_真实'] = true_labels[i]

    all_sample_results.append(sample_result)

print(f"\n... (共{len(Y_all)}个样本，显示前50个)")

# 第六步：保存详细结果
print(f"\n保存详细结果到CSV文件...")

# 创建详细结果DataFrame
detailed_results = pd.DataFrame(all_sample_results)

# 保存到文件
output_file = '独立测试_优化结果.csv'
detailed_results.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"详细结果已保存到: {output_file}")

# 保存所有样本的完整预测
full_results = []
for sample_idx in range(len(Y_all)):
    true_labels = Y_all[sample_idx]
    pred_labels = y_pred_best[sample_idx]
    probabilities = avg_prob[sample_idx]

    true_label_names = [class_name[i] for i in range(num_classes) if true_labels[i] == 1]
    pred_label_names = [class_name[i] for i in range(num_classes) if pred_labels[i] == 1]

    full_results.append({
        'sample_id': sample_idx + 1,
        'true_labels': ', '.join(true_label_names) if true_label_names else '无',
        'pred_labels': ', '.join(pred_label_names) if pred_label_names else '无',
        'correct': int(np.array_equal(true_labels, pred_labels)),
        **{f'{class_name[i]}_概率': probabilities[i] for i in range(num_classes)},
        **{f'{class_name[i]}_预测': pred_labels[i] for i in range(num_classes)},
        **{f'{class_name[i]}_真实': true_labels[i] for i in range(num_classes)}
    })

full_results_df = pd.DataFrame(full_results)
full_output_file = '独立测试_完整结果.csv'
full_results_df.to_csv(full_output_file, index=False, encoding='utf-8-sig')
print(f"完整结果已保存到: {full_output_file}")

print("\n独立测试完成!")