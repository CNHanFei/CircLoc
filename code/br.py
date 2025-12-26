import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, hamming_loss, label_ranking_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings('ignore')


def calculate_metrics(y_true, y_pred, y_pred_proba, num_classes):
    """计算多标签分类的各类指标"""
    metrics_per_class = {
        'acc': [],
        'auc': [],
        'aupr': [],
        'f1': [],
        'recall': [],
        'precision': []
    }

    # 计算每个类别的指标
    for i in range(num_classes):
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            except:
                auc = np.nan
            metrics_per_class['auc'].append(auc)

            try:
                aupr = average_precision_score(y_true[:, i], y_pred_proba[:, i])
            except:
                aupr = np.nan
            metrics_per_class['aupr'].append(aupr)
        else:
            metrics_per_class['auc'].append(np.nan)
            metrics_per_class['aupr'].append(np.nan)

        # 计算其他基于阈值的指标
        metrics_per_class['acc'].append(accuracy_score(y_true[:, i], y_pred[:, i]))
        metrics_per_class['f1'].append(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        metrics_per_class['recall'].append(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
        metrics_per_class['precision'].append(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))

    return metrics_per_class


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

    X = merge_feature_scaled_multilabel
    y = miRNA_loc_multilabel

    # 设置参数
    n_splits = 10
    num_classes = 7
    random_seed = 42
    np.random.seed(random_seed)
    class_name = ['Chromatin', 'Nucleoplasm', 'Nucleolus', 'Membrane', 'Nucleus', 'Cytosol', 'Cytoplasm']

    print("原始数据集大小:", len(X))
    print("原始各类别样本数:", np.sum(y, axis=0))

    # 初始化存储指标
    auc_ls = [0.0] * num_classes
    aupr_ls = [0.0] * num_classes
    acc_ls = [0.0] * num_classes
    f1_ls = [0.0] * num_classes
    recall_ls = [0.0] * num_classes
    precision_ls = [0.0] * num_classes

    # 初始化整体指标
    hamming_loss_total = 0.0
    macro_f1_total = 0.0
    micro_f1_total = 0.0
    ranking_loss_total = 0.0

    # 创建交叉验证分割器
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # 使用一个标签进行分层（选择样本最多的类别）
    y_single = y.argmax(axis=1) if y.shape[1] > 0 else np.zeros(len(y))

    fold_count = 0
    for train_idx, val_idx in kf.split(X, y_single):
        fold_count += 1
        print(f"Fold {fold_count}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = y[train_idx], y[val_idx]

        # 创建BR模型（Binary Relevance），使用随机森林作为基分类器
        # MultiOutputClassifier会自动为每个类别训练一个独立的随机森林
        rf_classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_seed,
            n_jobs=-1
        )

        br_model = MultiOutputClassifier(rf_classifier, n_jobs=-1)

        # 训练模型
        br_model.fit(X_train, Y_train)

        # 预测
        y_pred = br_model.predict(X_val)
        y_pred_proba = br_model.predict_proba(X_val)

        # 转换预测概率格式
        # MultiOutputClassifier的predict_proba返回每个分类器的概率数组
        # 我们需要将其转换为 (n_samples, n_classes) 的格式
        y_pred_proba_reshaped = np.zeros((len(X_val), num_classes))
        for i in range(num_classes):
            y_pred_proba_reshaped[:, i] = y_pred_proba[i][:, 1]  # 正类的概率

        # 计算每个类别的指标
        fold_metrics = calculate_metrics(Y_val, y_pred, y_pred_proba_reshaped, num_classes)

        # 累加指标
        for i in range(num_classes):
            if not np.isnan(fold_metrics['auc'][i]):
                auc_ls[i] += fold_metrics['auc'][i]
            if not np.isnan(fold_metrics['aupr'][i]):
                aupr_ls[i] += fold_metrics['aupr'][i]
            acc_ls[i] += fold_metrics['acc'][i]
            f1_ls[i] += fold_metrics['f1'][i]
            recall_ls[i] += fold_metrics['recall'][i]
            precision_ls[i] += fold_metrics['precision'][i]

        # 计算整体指标
        hamming_loss_total += hamming_loss(Y_val, y_pred)
        macro_f1_total += f1_score(Y_val, y_pred, average='macro', zero_division=0)
        micro_f1_total += f1_score(Y_val, y_pred, average='micro', zero_division=0)
        ranking_loss_total += label_ranking_loss(Y_val, y_pred_proba_reshaped)

    # 计算平均指标
    auc_ls = [auc / n_splits for auc in auc_ls]
    aupr_ls = [aupr / n_splits for aupr in aupr_ls]
    acc_ls = [acc / n_splits for acc in acc_ls]
    f1_ls = [f1 / n_splits for f1 in f1_ls]
    recall_ls = [recall / n_splits for recall in recall_ls]
    precision_ls = [precision / n_splits for precision in precision_ls]

    hamming_loss_avg = hamming_loss_total / n_splits
    macro_f1_avg = macro_f1_total / n_splits
    micro_f1_avg = micro_f1_total / n_splits
    ranking_loss_avg = ranking_loss_total / n_splits

    # 打印结果
    print("\n" + "=" * 80)
    print("10折交叉验证结果 - 随机森林(BR算法)")
    print("=" * 80)

    # 打印每个类别的指标
    print("\n每个类别的指标:")
    print("-" * 80)
    for i in range(num_classes):
        print(f"{class_name[i]:<12} - ACC: {acc_ls[i]:.4f}, AUC: {auc_ls[i]:.4f}, "
              f"AUPR: {aupr_ls[i]:.4f}, F1: {f1_ls[i]:.4f}, "
              f"Recall: {recall_ls[i]:.4f}, Precision: {precision_ls[i]:.4f}")

    # 打印整体指标
    print("\n整体指标:")
    print("-" * 80)
    print(f"Hamming Loss: {hamming_loss_avg:.4f}")
    print(f"Macro F1: {macro_f1_avg:.4f}")
    print(f"Micro F1: {micro_f1_avg:.4f}")
    print(f"Ranking Loss: {ranking_loss_avg:.4f}")

    # 计算并打印宏平均AUC和AUPR
    valid_auc = [auc for auc in auc_ls if not np.isnan(auc)]
    valid_aupr = [aupr for aupr in aupr_ls if not np.isnan(aupr)]

    if valid_auc:
        macro_auc = np.mean(valid_auc)
        print(f"Macro AUC: {macro_auc:.4f}")
    else:
        print("Macro AUC: N/A (所有类别的AUC计算失败)")

    if valid_aupr:
        macro_aupr = np.mean(valid_aupr)
        print(f"Macro AUPR: {macro_aupr:.4f}")
    else:
        print("Macro AUPR: N/A (所有类别的AUPR计算失败)")

    print("\n" + "=" * 80)