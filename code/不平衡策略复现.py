from random import random

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, hamming_loss, label_ranking_loss
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization, Flatten, \
    MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.metrics import AUC
import gc
import os
import json
import joblib

'''监控AUC早停，加入focal_loss,去除weight'''


def data_augmentation(x):
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.05)
    return x + noise


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

    model = Model(inputs, outputs)
    return model


def focal_loss_with_tau(gamma=2.0, alpha=0.25, tau=None):
    import tensorflow as tf
    import tensorflow.keras.backend as K
    import numpy as np

    def loss(y_true, y_pred):
        # cast to float32 to avoid dtype mismatch
        y_true_f = tf.cast(y_true, tf.float32)

        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)

        # pt: probability of the true class (works for both y=1 and y=0)
        pt = tf.where(tf.equal(y_true_f, 1.0), y_pred, 1.0 - y_pred)

        # alpha: can be scalar or per-class list/array
        if isinstance(alpha, (list, tuple, np.ndarray)):
            alpha_t = tf.constant(np.array(alpha, dtype=np.float32))[tf.newaxis, :]
        else:
            alpha_t = tf.constant(float(alpha), dtype=tf.float32)

        # tau: per-class weight (None or length-Q array)
        if tau is None:
            tau_t = tf.constant(1.0, dtype=tf.float32)
        else:
            tau_arr = np.array(tau, dtype=np.float32)
            tau_t = tf.constant(tau_arr)[tf.newaxis, :]  # shape (1, Q) -> will broadcast over batch

        modulating = K.pow(1.0 - pt, gamma)  # (batch, Q)
        # base focal term (same shape)
        loss_raw = - alpha_t * modulating * K.log(pt)

        # apply per-class tau
        loss_weighted = loss_raw * tau_t

        # average over batch and classes
        return K.mean(loss_weighted)

    return loss


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

    merge_feature = np.concatenate((
        kmer_feature, rckmer_feature,
        seq_feature,
        dis_feature, drug_feature, miRNA_feature,
        RNAErnie_feature,
        protein_feature
    ), axis=1)

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

    save_dir = './best_model_repeat_focalloss/'
    best_repeat = 11  # 指定最佳轮次为第11轮

    # ====== 加载第11轮保存的结果 ======
    print(f"=== 加载第{best_repeat}轮保存的结果 ===")

    # 加载阈值
    thresholds_path = os.path.join(save_dir, f'repeat_{best_repeat}_thresholds.json')
    with open(thresholds_path, 'r') as f:
        best_thresholds = json.load(f)
    print(f"已加载阈值文件: {thresholds_path}")

    # 加载最佳指标
    metrics_path = os.path.join(save_dir, f'repeat_{best_repeat}_best_metrics.json')
    with open(metrics_path, 'r') as f:
        best_metrics = json.load(f)
    print(f"已加载指标文件: {metrics_path}")

    # 加载fold结果
    fold_results_path = os.path.join(save_dir, f'repeat_{best_repeat}_fold_results.pkl')
    best_fold_results = joblib.load(fold_results_path)
    print(f"已加载fold结果文件: {fold_results_path}")

    # ====== 重新构建模型并进行预测 ======
    print(f"\n=== 重新构建第{best_repeat}轮模型并进行评估 ===")

    # 设置随机种子以确保可复现性
    np.random.seed(random_seed + best_repeat)
    tf.random.set_seed(random_seed + best_repeat)

    # 初始化指标列表
    auc_ls = [0] * num_classes
    aupr_ls = [0] * num_classes
    acc_ls = [0] * num_classes
    f1_ls = [0] * num_classes
    recall_ls = [0] * num_classes
    precision_ls = [0] * num_classes

    # 初始化多标签评估指标
    micro_f1_per_fold = []
    macro_f1_per_fold = []
    hamming_loss_per_fold = []
    ranking_loss_per_fold = []

    # 用于存储所有fold的结果
    all_fold_true = []
    all_fold_pred_bin = []
    all_fold_pred_prob = []

    X, Y = shuffle(x, y, random_state=random_seed + best_repeat)

    for fold in range(n_splits):
        print(f"  Fold {fold + 1}")
        fold_size = len(X) // n_splits
        val_idx = np.arange(fold * fold_size, (fold + 1) * fold_size)
        train_idx = np.setdiff1d(np.arange(len(X)), val_idx)
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        # 构建模型
        model = create_multi_label_model((len(merge_feature[0]),), num_classes)

        # 加载保存的模型权重
        model_path = os.path.join(save_dir, f'repeat_{best_repeat}_fold_{fold + 1}_model.h5')
        if os.path.exists(model_path):
            model.load_weights(model_path)
            print(f"    已加载模型权重: {model_path}")
        else:
            print(f"    警告: 模型权重文件不存在: {model_path}")
            continue

        tau = [1.6259, 1.6085, 1.3893, 0.8611, 0.5407, 0.5324, 0.4422]

        model.compile(optimizer='adam',
                      loss=focal_loss_with_tau(2.0, 0.25, tau),
                      metrics=[AUC(name='auc')])

        # 进行预测
        y_pred = model.predict(X_val)

        # 使用保存的阈值进行二值化
        thresholds = best_thresholds[fold]
        y_pred_bin = np.array([(y_pred[:, i] > thresholds[i]).astype(int) for i in range(num_classes)]).T

        # 计算每个类别的指标
        for i in range(num_classes):
            if len(np.unique(Y_val[:, i])) > 1:
                auc_ls[i] += roc_auc_score(Y_val[:, i], y_pred[:, i])
                aupr_ls[i] += average_precision_score(Y_val[:, i], y_pred[:, i])
            acc_ls[i] += accuracy_score(Y_val[:, i], y_pred_bin[:, i])
            f1_ls[i] += f1_score(Y_val[:, i], y_pred_bin[:, i], zero_division=0)
            recall_ls[i] += recall_score(Y_val[:, i], y_pred_bin[:, i], zero_division=0)
            precision_ls[i] += precision_score(Y_val[:, i], y_pred_bin[:, i], zero_division=0)

        # 计算多标签评估指标
        # 微观和宏观F1
        micro_f1 = f1_score(Y_val, y_pred_bin, average='micro', zero_division=0)
        macro_f1 = f1_score(Y_val, y_pred_bin, average='macro', zero_division=0)
        micro_f1_per_fold.append(micro_f1)
        macro_f1_per_fold.append(macro_f1)

        # 汉明损失
        hamming = hamming_loss(Y_val, y_pred_bin)
        hamming_loss_per_fold.append(hamming)

        # 排名损失
        ranking = label_ranking_loss(Y_val, y_pred)
        ranking_loss_per_fold.append(ranking)

        # 保存fold结果
        all_fold_true.append(Y_val)
        all_fold_pred_bin.append(y_pred_bin)
        all_fold_pred_prob.append(y_pred)

        del model
        tf.keras.backend.clear_session()
        gc.collect()

        print(f"    Fold {fold + 1} 结果: Micro-F1={micro_f1:.4f}, Macro-F1={macro_f1:.4f}, "
              f"Hamming Loss={hamming:.4f}, Ranking Loss={ranking:.4f}")

    # ====== 计算平均指标 ======
    print(f"\n=== 第{best_repeat}轮平均指标 ===")

    # 计算每个类别的平均指标
    avg_auc = [auc_ls[i] / n_splits for i in range(num_classes)]
    avg_aupr = [aupr_ls[i] / n_splits for i in range(num_classes)]
    avg_acc = [acc_ls[i] / n_splits for i in range(num_classes)]
    avg_f1 = [f1_ls[i] / n_splits for i in range(num_classes)]
    avg_recall = [recall_ls[i] / n_splits for i in range(num_classes)]
    avg_precision = [precision_ls[i] / n_splits for i in range(num_classes)]

    # 计算多标签评估指标的平均值
    avg_micro_f1 = np.mean(micro_f1_per_fold)
    avg_macro_f1 = np.mean(macro_f1_per_fold)
    avg_hamming_loss = np.mean(hamming_loss_per_fold)
    avg_ranking_loss = np.mean(ranking_loss_per_fold)

    # 打印每个类别的指标
    for i in range(num_classes):
        print(f"Class {class_name[i]}: "
              f"ACC: {avg_acc[i]:.3f}, AUC: {avg_auc[i]:.3f}, AUPR: {avg_aupr[i]:.3f}, "
              f"F1: {avg_f1[i]:.3f}, Recall: {avg_recall[i]:.3f}, Precision: {avg_precision[i]:.3f}")

    print(f"\n=== 多标签评估指标 ===")
    print(f"平均 Micro-F1: {avg_micro_f1:.4f}")
    print(f"平均 Macro-F1: {avg_macro_f1:.4f}")
    print(f"平均 Hamming Loss: {avg_hamming_loss:.4f}")
    print(f"平均 Ranking Loss: {avg_ranking_loss:.4f}")

    # ====== 合并所有fold结果进行整体评估 ======
    print(f"\n=== 整体评估（合并所有fold）===")

    # 合并所有fold的结果
    y_true_all = np.vstack(all_fold_true)
    y_pred_bin_all = np.vstack(all_fold_pred_bin)
    y_pred_prob_all = np.vstack(all_fold_pred_prob)

    # 计算整体指标
    overall_metrics = {}

    # 计算每个类别的整体指标
    for i in range(num_classes):
        if len(np.unique(y_true_all[:, i])) > 1:
            auc_i = roc_auc_score(y_true_all[:, i], y_pred_prob_all[:, i])
            aupr_i = average_precision_score(y_true_all[:, i], y_pred_prob_all[:, i])
        else:
            auc_i = 0.0
            aupr_i = 0.0

        acc_i = accuracy_score(y_true_all[:, i], y_pred_bin_all[:, i])
        f1_i = f1_score(y_true_all[:, i], y_pred_bin_all[:, i], zero_division=0)
        recall_i = recall_score(y_true_all[:, i], y_pred_bin_all[:, i], zero_division=0)
        precision_i = precision_score(y_true_all[:, i], y_pred_bin_all[:, i], zero_division=0)

        overall_metrics[class_name[i]] = {
            'ACC': acc_i,
            'AUC': auc_i,
            'AUPR': aupr_i,
            'F1': f1_i,
            'Recall': recall_i,
            'Precision': precision_i
        }

    # 计算整体多标签指标
    overall_micro_f1 = f1_score(y_true_all, y_pred_bin_all, average='micro', zero_division=0)
    overall_macro_f1 = f1_score(y_true_all, y_pred_bin_all, average='macro', zero_division=0)
    overall_hamming_loss = hamming_loss(y_true_all, y_pred_bin_all)
    overall_ranking_loss = label_ranking_loss(y_true_all, y_pred_prob_all)

    # 计算微观和宏观的精确率与召回率
    micro_precision = precision_score(y_true_all, y_pred_bin_all, average='micro', zero_division=0)
    micro_recall = recall_score(y_true_all, y_pred_bin_all, average='micro', zero_division=0)
    macro_precision = precision_score(y_true_all, y_pred_bin_all, average='macro', zero_division=0)
    macro_recall = recall_score(y_true_all, y_pred_bin_all, average='macro', zero_division=0)

    # 打印整体指标
    print("\n=== 每个类别的整体指标 ===")
    for i in range(num_classes):
        metrics = overall_metrics[class_name[i]]
        print(f"Class {class_name[i]}: "
              f"ACC: {metrics['ACC']:.3f}, AUC: {metrics['AUC']:.3f}, AUPR: {metrics['AUPR']:.3f}, "
              f"F1: {metrics['F1']:.3f}, Recall: {metrics['Recall']:.3f}, Precision: {metrics['Precision']:.3f}")

    print(f"\n=== 整体多标签评估指标 ===")
    print(f"整体 Micro-Precision: {micro_precision:.4f}")
    print(f"整体 Micro-Recall: {micro_recall:.4f}")
    print(f"整体 Micro-F1: {overall_micro_f1:.4f}")
    print(f"整体 Macro-Precision: {macro_precision:.4f}")
    print(f"整体 Macro-Recall: {macro_recall:.4f}")
    print(f"整体 Macro-F1: {overall_macro_f1:.4f}")
    print(f"整体 Hamming Loss: {overall_hamming_loss:.4f}")
    print(f"整体 Ranking Loss: {overall_ranking_loss:.4f}")

    # 计算平均AUC和平均F1
    mean_auc = np.mean([overall_metrics[cn]['AUC'] for cn in class_name])
    mean_f1 = np.mean([overall_metrics[cn]['F1'] for cn in class_name])

    print(f"\n=== 汇总指标 ===")
    print(f"平均AUC: {mean_auc:.4f}")
    print(f"平均F1: {mean_f1:.4f}")

    # ====== 保存最终评估结果 ======
    final_results = {
        'best_repeat': best_repeat,
        'class_metrics': overall_metrics,
        'overall_metrics': {
            'mean_auc': mean_auc,
            'mean_f1': mean_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': overall_micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': overall_macro_f1,
            'hamming_loss': overall_hamming_loss,
            'ranking_loss': overall_ranking_loss
        },
        'per_fold_metrics': {
            'micro_f1_per_fold': micro_f1_per_fold,
            'macro_f1_per_fold': macro_f1_per_fold,
            'hamming_loss_per_fold': hamming_loss_per_fold,
            'ranking_loss_per_fold': ranking_loss_per_fold
        }
    }

    # 保存结果
    results_dir = './final_results_repeat_11/'
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, 'repeat_11_final_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"\n=== 结果已保存 ===")
    print(f"最终评估结果已保存到: {results_path}")

    # 也保存为CSV格式便于查看
    class_metrics_df = pd.DataFrame([
        {**{'Class': class_name[i]}, **overall_metrics[class_name[i]]}
        for i in range(num_classes)
    ])

    csv_path = os.path.join(results_dir, 'repeat_11_class_metrics.csv')
    class_metrics_df.to_csv(csv_path, index=False)
    print(f"类别指标已保存到: {csv_path}")

    # 创建汇总表格
    summary_df = pd.DataFrame({
        'Metric': ['Average AUC', 'Average F1', 'Micro-F1', 'Macro-F1',
                   'Hamming Loss', 'Ranking Loss'],
        'Value': [mean_auc, mean_f1, overall_micro_f1, overall_macro_f1,
                  overall_hamming_loss, overall_ranking_loss]
    })

    summary_csv_path = os.path.join(results_dir, 'repeat_11_summary_metrics.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"汇总指标已保存到: {summary_csv_path}")