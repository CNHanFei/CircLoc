# reproduce_eval_sample_by_sample.py
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
from collections import defaultdict

# ======== 配置 ========
save_dir = './best_model_repeat_独立测试/'  # 模型权重目录
best_repeat = 8  # <-- 改成你的最佳 repeat
random_seed = 42
num_classes = 7
class_name = ['Chromatin', 'Nucleoplasm', 'Nucleolus', 'Membrane', 'Nucleus', 'Cytosol', 'Cytoplasm']

# ======== 数据加载（和训练时一致） ========
print("加载测试集数据...")
df_seq_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集少\seq_sim_feature.xlsx', index_col=0, header=None)
df_miRNA_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集少\rna_mi_features.xlsx', header=None)
df_drug_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集少\rna_drug_features.xlsx', header=None)
df_dis_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集少\rna_disease_features.xlsx', header=None)
df_loc = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集少\location.xlsx', index_col=0)
df_loc_index = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集少\locationIndex.xlsx', index_col=0, header=None)
df_kmer_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集少\rna_kmer_features.xlsx', header=None)
df_rckmer_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集少\rna_rckmer_features.xlsx', header=None)
df_RNAErnie_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集少\rna_model_features.xlsx', header=None)
df_protein_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\测试集少\gate_feature_protein.xlsx', header=None)


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
    return merge_feature_scaled[select_row], circRNA_loc[select_row], scaler


X_all, Y_all, test_scaler = build_merged_features()
total_dim = X_all.shape[1]


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
    print(f"载入阈值文件: {thresholds_file}")
    print("各折叠阈值:", best_thresholds)
else:
    best_thresholds = None
    print("未找到阈值文件，使用默认阈值0.5")


# ======== 计算单个样本的指标 ========
def compute_sample_metrics(y_true_sample, y_pred_sample):
    """计算单个样本的指标"""
    # 样本级别的指标（将单个样本视为一个batch）
    y_true_2d = y_true_sample.reshape(1, -1)
    y_pred_2d = y_pred_sample.reshape(1, -1)

    acc = accuracy_score(y_true_2d, y_pred_2d)
    precision = precision_score(y_true_2d, y_pred_2d, average='samples', zero_division=0)
    recall = recall_score(y_true_2d, y_pred_2d, average='samples', zero_division=0)
    f1 = f1_score(y_true_2d, y_pred_2d, average='samples', zero_division=0)

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ======== 逐个样本进行预测 ========
print("\n" + "=" * 80)
print("开始逐个样本独立测试")
print("=" * 80)

# 准备存储所有结果
all_sample_results = []
all_true_labels = []
all_pred_labels = []
all_probabilities = []

# 统计每个折叠模型对每个样本的预测
n_splits = 10
fold_predictions = [[] for _ in range(n_splits)]
fold_thresholds = []

# 先加载所有模型
print(f"\n加载最佳repeat {best_repeat}的{10}个模型...")
models = []
for fold_id in range(1, n_splits + 1):
    model_path = os.path.join(save_dir, f'repeat_{best_repeat}_fold_{fold_id}_model.h5')
    if os.path.exists(model_path):
        model = create_multi_label_model((total_dim,), num_classes)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.load_weights(model_path)
        models.append(model)
        print(f"  加载模型 {fold_id}/10")

        # 获取该折叠的阈值
        if best_thresholds and fold_id - 1 < len(best_thresholds):
            fold_thresholds.append(best_thresholds[fold_id - 1])
        else:
            fold_thresholds.append([0.5] * num_classes)
    else:
        print(f"  警告: 模型文件 {model_path} 不存在")
        models.append(None)

print(f"\n成功加载 {len([m for m in models if m is not None])} 个模型")

# 计算平均阈值
avg_thresholds = np.mean(fold_thresholds, axis=0) if fold_thresholds else [0.5] * num_classes
print(f"平均阈值: {avg_thresholds}")

# 对每个样本进行预测
for sample_idx in range(len(X_all)):
    print(f"\n{'=' * 60}")
    print(f"样本 {sample_idx + 1}/{len(X_all)}")

    # 获取当前样本
    X_sample = X_all[sample_idx:sample_idx + 1]
    Y_sample = Y_all[sample_idx]

    # 存储每个模型的预测结果
    fold_probs = []
    fold_preds = []

    # 使用每个模型进行预测
    for fold_id, model in enumerate(models):
        if model is not None:
            # 预测概率
            prob = model.predict(X_sample, batch_size=1, verbose=0)[0]
            fold_probs.append(prob)

            # 使用该折叠的阈值进行二分类
            thresholds = fold_thresholds[fold_id]
            pred = (prob > thresholds).astype(int)
            fold_preds.append(pred)

    if fold_probs:
        # 计算平均概率和投票结果
        avg_prob = np.mean(fold_probs, axis=0)

        # 方法1: 使用平均阈值
        final_pred_avg = (avg_prob > avg_thresholds).astype(int)

        # 方法2: 投票机制（多数表决）
        vote_pred = np.mean(fold_preds, axis=0)
        final_pred_vote = (vote_pred > 0.5).astype(int)

        # 使用投票结果作为最终预测
        final_pred = final_pred_vote
        final_prob = avg_prob

        # 获取标签名称
        true_label_indices = np.where(Y_sample == 1)[0]
        pred_label_indices = np.where(final_pred == 1)[0]

        true_labels = [class_name[i] for i in true_label_indices] if len(true_label_indices) > 0 else ['无']
        pred_labels = [class_name[i] for i in pred_label_indices] if len(pred_label_indices) > 0 else ['无']

        # 计算样本指标
        sample_metrics = compute_sample_metrics(Y_sample, final_pred)

        # 显示结果
        print(f"真实标签: {', '.join(true_labels)}")
        print(f"预测标签: {', '.join(pred_labels)}")
        print(f"预测概率: {[f'{final_prob[i]:.3f}' for i in range(num_classes)]}")
        print(f"阈值: {[f'{avg_thresholds[i]:.3f}' for i in range(num_classes)]}")
        print(f"样本准确率: {sample_metrics['acc']:.3f}")
        print(f"样本精确率: {sample_metrics['precision']:.3f}")
        print(f"样本召回率: {sample_metrics['recall']:.3f}")
        print(f"样本F1分数: {sample_metrics['f1']:.3f}")

        # 检查是否正确预测
        if np.array_equal(Y_sample, final_pred):
            print("✓ 预测正确!")
        else:
            print("✗ 预测错误")

            # 显示错误详情
            for i in range(num_classes):
                if Y_sample[i] != final_pred[i]:
                    if Y_sample[i] == 1 and final_pred[i] == 0:
                        print(f"  漏报: {class_name[i]}")
                    elif Y_sample[i] == 0 and final_pred[i] == 1:
                        print(f"  误报: {class_name[i]}")

        # 存储结果
        all_sample_results.append({
            'sample_id': sample_idx + 1,
            'true_labels': ', '.join(true_labels),
            'pred_labels': ', '.join(pred_labels),
            'accuracy': sample_metrics['acc'],
            'precision': sample_metrics['precision'],
            'recall': sample_metrics['recall'],
            'f1': sample_metrics['f1'],
            'correct': int(np.array_equal(Y_sample, final_pred))
        })

        all_true_labels.append(Y_sample)
        all_pred_labels.append(final_pred)
        all_probabilities.append(final_prob)

    else:
        print("错误: 没有可用的模型进行预测")

# ======== 总体评估 ========
print("\n" + "=" * 80)
print("独立测试总体结果")
print("=" * 80)

if all_true_labels and all_pred_labels:
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    all_probabilities = np.array(all_probabilities)

    # 按类别计算指标
    print("\n按类别统计:")
    print("-" * 60)

    for i in range(num_classes):
        true_count = np.sum(all_true_labels[:, i])
        pred_count = np.sum(all_pred_labels[:, i])

        if true_count > 0:  # 只在有真实标签的类别上计算
            try:
                auc = roc_auc_score(all_true_labels[:, i], all_probabilities[:, i])
            except:
                auc = np.nan

            try:
                aupr = average_precision_score(all_true_labels[:, i], all_probabilities[:, i])
            except:
                aupr = np.nan

            acc = accuracy_score(all_true_labels[:, i], all_pred_labels[:, i])
            f1 = f1_score(all_true_labels[:, i], all_pred_labels[:, i], zero_division=0)
            recall = recall_score(all_true_labels[:, i], all_pred_labels[:, i], zero_division=0)
            precision = precision_score(all_true_labels[:, i], all_pred_labels[:, i], zero_division=0)

            print(f"{class_name[i]:<12}: 样本数={true_count:3d}, 预测数={pred_count:3d}")
            print(f"              ACC={acc:.4f}, AUC={auc:.4f}, AUPR={aupr:.4f}")
            print(f"              F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")
            print()
        else:
            print(f"{class_name[i]:<12}: 测试集中无样本")
            print()

    # 总体指标
    print("\n总体指标:")
    print("-" * 60)

    # 样本级别的平均指标
    if all_sample_results:
        avg_acc = np.mean([r['accuracy'] for r in all_sample_results])
        avg_precision = np.mean([r['precision'] for r in all_sample_results])
        avg_recall = np.mean([r['recall'] for r in all_sample_results])
        avg_f1 = np.mean([r['f1'] for r in all_sample_results])
        correct_samples = sum([r['correct'] for r in all_sample_results])
        total_samples = len(all_sample_results)

        print(f"样本级别平均准确率: {avg_acc:.4f}")
        print(f"样本级别平均精确率: {avg_precision:.4f}")
        print(f"样本级别平均召回率: {avg_recall:.4f}")
        print(f"样本级别平均F1分数: {avg_f1:.4f}")
        print(f"完全正确预测的样本数: {correct_samples}/{total_samples} ({correct_samples / total_samples:.2%})")

    # 宏平均AUC和AUPR
    valid_classes = [i for i in range(num_classes) if np.sum(all_true_labels[:, i]) > 0]
    auc_values = []
    aupr_values = []

    for i in valid_classes:
        if len(np.unique(all_true_labels[:, i])) > 1:
            try:
                auc = roc_auc_score(all_true_labels[:, i], all_probabilities[:, i])
                auc_values.append(auc)
            except:
                pass
            try:
                aupr = average_precision_score(all_true_labels[:, i], all_probabilities[:, i])
                aupr_values.append(aupr)
            except:
                pass

    if auc_values:
        print(f"宏平均AUC: {np.mean(auc_values):.4f}")
    if aupr_values:
        print(f"宏平均AUPR: {np.mean(aupr_values):.4f}")

    # 保存详细结果到CSV
    print(f"\n保存详细结果到CSV文件...")

    # 创建详细结果DataFrame
    detailed_results = pd.DataFrame(all_sample_results)

    # 添加每个类别的概率
    for i in range(num_classes):
        detailed_results[f'{class_name[i]}_概率'] = all_probabilities[:, i]
        detailed_results[f'{class_name[i]}_预测'] = all_pred_labels[:, i]
        detailed_results[f'{class_name[i]}_真实'] = all_true_labels[:, i]

    # 保存到文件
    output_file = '独立测试_逐个样本结果.csv'
    detailed_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"详细结果已保存到: {output_file}")

    # 创建汇总统计文件
    summary_stats = []
    for i in range(num_classes):
        if np.sum(all_true_labels[:, i]) > 0:
            true_count = np.sum(all_true_labels[:, i])
            pred_count = np.sum(all_pred_labels[:, i])
            acc = accuracy_score(all_true_labels[:, i], all_pred_labels[:, i])
            f1 = f1_score(all_true_labels[:, i], all_pred_labels[:, i], zero_division=0)

            try:
                auc = roc_auc_score(all_true_labels[:, i], all_probabilities[:, i])
            except:
                auc = np.nan

            summary_stats.append({
                '类别': class_name[i],
                '测试集样本数': true_count,
                '预测为正样本数': pred_count,
                '准确率': acc,
                'F1分数': f1,
                'AUC': auc
            })

    summary_df = pd.DataFrame(summary_stats)
    summary_file = '独立测试_类别汇总.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"类别汇总已保存到: {summary_file}")

else:
    print("错误: 没有预测结果")

print("\n独立测试完成!")