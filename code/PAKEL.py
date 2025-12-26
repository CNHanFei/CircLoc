import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, hamming_loss, label_ranking_loss
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from skmultilearn.ensemble import RakelD
from skmultilearn.model_selection import iterative_train_test_split
import joblib
import json
import os
from collections import Counter


def create_rakel_model(num_classes, base_estimator=None, n_estimators=20, labelset_size=3):
    """创建RAkEL多标签分类模型"""
    if base_estimator is None:
        base_estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )

    # 创建RAkEL模型
    model = RakelD(
        base_classifier=base_estimator,
        base_classifier_require_dense=[True, True],
        labelset_size=labelset_size,

    )
    return model


if __name__ == '__main__':
    # ====== 数据加载 ======
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

    loc_index = df_loc_index[1].tolist()
    select_row = np.array([value == 1 for value in loc_index])

    dis_feature = df_dis_feature.values
    seq_feature = df_seq_feature.values
    miRNA_feature = df_miRNA_feature.values
    RNAErnie_feature = df_RNAErnie_feature.values
    drug_feature = df_drug_feature.values
    circRNA_loc = df_loc.values
    kmer_feature = df_kmer_feature.values
    rckmer_feature = df_rckmer_feature.values
    protein_feature = df_protein_feature.values

    # 合并特征
    merge_feature = np.concatenate((
        kmer_feature, rckmer_feature,
        seq_feature,
        dis_feature, drug_feature, miRNA_feature,
        RNAErnie_feature,
        protein_feature
    ), axis=1)

    # 标准化特征
    scaler = StandardScaler()
    merge_feature_scaled = scaler.fit_transform(merge_feature)
    miRNA_loc_multilabel = circRNA_loc[select_row]
    merge_feature_scaled_multilabel = merge_feature_scaled[select_row]

    x = merge_feature_scaled_multilabel
    y = miRNA_loc_multilabel

    n_splits = 10
    num_classes = 7
    random_seed = 42
    np.random.seed(random_seed)
    class_name = ['Chromatin', 'Nucleoplasm', 'Nucleolus', 'Membrane', 'Nucleus', 'Cytosol', 'Cytoplasm']

    print("原始数据集大小:", len(x))
    print("原始各类别样本数:", np.sum(y, axis=0))
    print("各类别正样本比例:", np.mean(y, axis=0))

    # 初始化存储结果的列表
    auc_ls = [0] * num_classes
    aupr_ls = [0] * num_classes
    acc_ls = [0] * num_classes
    f1_ls = [0] * num_classes
    recall_ls = [0] * num_classes
    precision_ls = [0] * num_classes
    hamming_loss_ls = 0
    macro_f1_ls = 0
    micro_f1_ls = 0
    ranking_loss_ls = 0

    # 用于存储每类的预测结果（用于计算AUC和AUPR）
    all_true = {i: [] for i in range(num_classes)}
    all_pred_proba = {i: [] for i in range(num_classes)}

    print("\n=== 开始训练RAkEL模型（随机森林基分类器）===")

    # 10折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold = 1

    for train_idx, val_idx in kf.split(x):
        print(f"\n  Fold {fold}/{n_splits}")
        fold += 1

        X_train, X_val = x[train_idx], x[val_idx]
        Y_train, Y_val = y[train_idx], y[val_idx]

        # 创建并训练RAkEL模型
        print(f"    训练样本数: {len(X_train)}, 验证样本数: {len(X_val)}")

        # 创建RAkEL模型（使用随机森林作为基分类器）
        model = create_rakel_model(
            num_classes=num_classes,
            base_estimator=RandomForestClassifier(
                n_estimators=100,
                max_depth=20,  # 限制树深度防止过拟合
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42 + fold,
                n_jobs=-1
            ),
            n_estimators=20,  # RAkEL中的分类器数量
            labelset_size=3  # 每个分类器处理的标签数
        )

        # 训练模型
        model.fit(X_train, Y_train)

        # 预测
        y_pred_bin = model.predict(X_val).toarray()  # 转换为密集数组
        y_pred_proba = model.predict_proba(X_val).toarray()  # 预测概率

        # 为每个类别计算指标
        for i in range(num_classes):
            # 收集所有预测概率用于计算AUC和AUPR
            all_true[i].extend(Y_val[:, i])
            all_pred_proba[i].extend(y_pred_proba[:, i])

            # 计算当前fold的指标并累加
            if len(np.unique(Y_val[:, i])) > 1:
                auc_ls[i] += roc_auc_score(Y_val[:, i], y_pred_proba[:, i])
                aupr_ls[i] += average_precision_score(Y_val[:, i], y_pred_proba[:, i])

            acc_ls[i] += accuracy_score(Y_val[:, i], y_pred_bin[:, i])
            f1_ls[i] += f1_score(Y_val[:, i], y_pred_bin[:, i], zero_division=0)
            recall_ls[i] += recall_score(Y_val[:, i], y_pred_bin[:, i], zero_division=0)
            precision_ls[i] += precision_score(Y_val[:, i], y_pred_bin[:, i], zero_division=0)

        # 计算多标签指标
        hamming_loss_ls += hamming_loss(Y_val, y_pred_bin)
        macro_f1_ls += f1_score(Y_val, y_pred_bin, average='macro', zero_division=0)
        micro_f1_ls += f1_score(Y_val, y_pred_bin, average='micro', zero_division=0)
        ranking_loss_ls += label_ranking_loss(Y_val, y_pred_proba)

    # 计算平均指标
    print("\n=== 交叉验证结果 ===")
    print(f"10折交叉验证平均指标:")

    # 计算每个类别的平均AUC和AUPR（基于所有预测）
    final_auc_scores = []
    final_aupr_scores = []

    for i in range(num_classes):
        # 使用所有预测计算AUC和AUPR
        true_values = np.array(all_true[i])
        pred_values = np.array(all_pred_proba[i])

        if len(np.unique(true_values)) > 1:
            class_auc = roc_auc_score(true_values, pred_values)
            class_aupr = average_precision_score(true_values, pred_values)
        else:
            class_auc = 0.0
            class_aupr = 0.0

        final_auc_scores.append(class_auc)
        final_aupr_scores.append(class_aupr)

        print(f"\n{class_name[i]}:")
        print(f"  AUC:  {class_auc:.4f}")
        print(f"  AUPR: {class_aupr:.4f}")
        print(f"  ACC:  {acc_ls[i] / n_splits:.4f}")
        print(f"  F1:   {f1_ls[i] / n_splits:.4f}")
        print(f"  Recall: {recall_ls[i] / n_splits:.4f}")
        print(f"  Precision: {precision_ls[i] / n_splits:.4f}")

    # 打印整体指标
    print("\n=== 整体多标签指标 ===")
    print(f"平均Hamming Loss: {hamming_loss_ls / n_splits:.4f}")
    print(f"平均Macro F1: {macro_f1_ls / n_splits:.4f}")
    print(f"平均Micro F1: {micro_f1_ls / n_splits:.4f}")
    print(f"平均Ranking Loss: {ranking_loss_ls / n_splits:.4f}")

    # 计算并打印AUC和AUPR的宏平均
    macro_auc = np.mean([auc for auc in final_auc_scores if auc > 0])
    macro_aupr = np.mean([aupr for aupr in final_aupr_scores if aupr > 0])

    print(f"\n宏平均AUC: {macro_auc:.4f}")
    print(f"宏平均AUPR: {macro_aupr:.4f}")

    # 保存结果到文件
    results = {
        'class_names': class_name,
        'auc_scores': final_auc_scores,
        'aupr_scores': final_aupr_scores,
        'macro_auc': float(macro_auc),
        'macro_aupr': float(macro_aupr),
        'fold_metrics': {
            'acc': [float(acc / n_splits) for acc in acc_ls],
            'f1': [float(f1 / n_splits) for f1 in f1_ls],
            'recall': [float(recall / n_splits) for recall in recall_ls],
            'precision': [float(precision / n_splits) for precision in precision_ls]
        },
        'overall_metrics': {
            'hamming_loss': float(hamming_loss_ls / n_splits),
            'macro_f1': float(macro_f1_ls / n_splits),
            'micro_f1': float(micro_f1_ls / n_splits),
            'ranking_loss': float(ranking_loss_ls / n_splits)
        }
    }

    # 创建结果目录
    os.makedirs('./ml_results/', exist_ok=True)

    # 保存结果
    with open('./ml_results/rakel_rf_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n结果已保存到: ./ml_results/rakel_rf_results.json")

    # 可视化结果
    plt.figure(figsize=(12, 5))

    # AUC和AUPR柱状图
    plt.subplot(1, 2, 1)
    x_pos = np.arange(len(class_name))
    width = 0.35

    plt.bar(x_pos - width / 2, final_auc_scores, width, label='AUC', alpha=0.8)
    plt.bar(x_pos + width / 2, final_aupr_scores, width, label='AUPR', alpha=0.8)
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('AUC and AUPR for Each Class')
    plt.xticks(x_pos, class_name, rotation=45, ha='right')
    plt.legend()
    plt.ylim([0, 1.0])
    plt.grid(True, alpha=0.3)

    # 各类别F1分数
    plt.subplot(1, 2, 2)
    f1_scores = [f1 / n_splits for f1 in f1_ls]
    plt.bar(class_name, f1_scores, alpha=0.7, color='green')
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score for Each Class')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1.0])
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./ml_results/rakel_rf_performance.png', dpi=300, bbox_inches='tight')
    plt.show()