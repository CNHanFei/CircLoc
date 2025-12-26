from random import random

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, hamming_loss, label_ranking_loss
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.metrics import AUC
import gc

'''监控AUC早停，加入focal_loss,去除weight'''


def data_augmentation(x):
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.05)
    return x + noise


def create_multi_label_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    # 移除了MultiHeadAttention层
    x = Flatten()(inputs)

    x = Dense(512, activation='relu')(x)
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

    print("原始数据集大小:", len(x))
    print("原始各类别样本数:", np.sum(y, axis=0))

    # 初始化指标列表
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

    # 准备存储所有折的预测结果
    all_y_true = []
    all_y_pred = []
    all_y_pred_bin = []

    print("=== 单次10折交叉验证开始 ===")

    # 只进行一次shuffle，使用固定随机种子
    X, Y = shuffle(x, y, random_state=random_seed)

    for fold in range(n_splits):
        print(f"  Fold {fold + 1}")
        fold_size = len(X) // n_splits
        val_idx = np.arange(fold * fold_size, (fold + 1) * fold_size)
        train_idx = np.setdiff1d(np.arange(len(X)), val_idx)
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        # 创建模型（已移除注意力层）
        model = create_multi_label_model((len(merge_feature[0]),), num_classes)

        # 编译模型
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=[AUC(name='auc')])

        # 回调函数
        early_stop = EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

        # 训练模型
        model.fit(data_augmentation(X_train), Y_train,
                  epochs=100, batch_size=16, validation_split=0.2,
                  callbacks=[early_stop, reduce_lr], verbose=0)

        # 预测
        y_pred = model.predict(X_val)

        # 为每个类别寻找最佳阈值
        thresholds = []
        for i in range(num_classes):
            best_thres = 0.5
            best_f1 = 0
            for thres in np.arange(0.1, 0.9, 0.01):
                preds = (y_pred[:, i] > thres).astype(int)
                score = f1_score(Y_val[:, i], preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_thres = thres
            thresholds.append(best_thres)

        # 使用最佳阈值进行二值化
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

        # 计算整体指标
        hamming_loss_ls += hamming_loss(Y_val, y_pred_bin)
        macro_f1_ls += f1_score(Y_val, y_pred_bin, average='macro', zero_division=0)
        micro_f1_ls += f1_score(Y_val, y_pred_bin, average='micro', zero_division=0)
        ranking_loss_ls += label_ranking_loss(Y_val, y_pred)

        # 收集预测结果
        all_y_true.append(Y_val)
        all_y_pred.append(y_pred)
        all_y_pred_bin.append(y_pred_bin)

        # 清理内存
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # 计算平均指标
    auc_ls = [auc / n_splits for auc in auc_ls]
    aupr_ls = [aupr / n_splits for aupr in aupr_ls]
    acc_ls = [acc / n_splits for acc in acc_ls]
    f1_ls = [f1 / n_splits for f1 in f1_ls]
    recall_ls = [recall / n_splits for recall in recall_ls]
    precision_ls = [precision / n_splits for precision in precision_ls]

    hamming_loss_avg = hamming_loss_ls / n_splits
    macro_f1_avg = macro_f1_ls / n_splits
    micro_f1_avg = micro_f1_ls / n_splits
    ranking_loss_avg = ranking_loss_ls / n_splits

    # 打印每个类别的指标
    print("\n=== 各类别指标 ===")
    for i in range(num_classes):
        print(f"Class {class_name[i]}: ACC: {acc_ls[i]:.3f}, AUC: {auc_ls[i]:.3f}, "
              f"AUPR: {aupr_ls[i]:.3f}, F1: {f1_ls[i]:.3f}, "
              f"Recall: {recall_ls[i]:.3f}, Precision: {precision_ls[i]:.3f}")

    # 打印整体指标
    print(f"\n=== 整体指标 ===")
    print(f"Hamming Loss: {hamming_loss_avg:.4f}")
    print(f"Macro F1: {macro_f1_avg:.4f}")
    print(f"Micro F1: {micro_f1_avg:.4f}")
    print(f"Ranking Loss: {ranking_loss_avg:.4f}")

    # 计算并打印平均AUC
    mean_auc = np.mean(auc_ls)
    print(f"\n平均AUC: {mean_auc:.4f}")
    mean_aupr = np.mean(aupr_ls)
    print(f"平均AUPR: {mean_aupr:.4f}")

    # 可以保存预测结果供后续分析
    print("\n训练完成！")