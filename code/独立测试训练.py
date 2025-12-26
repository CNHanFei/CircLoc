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
    df_seq_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\训练集\2.0细胞_seq_sim_feature.xlsx', index_col=0, header=None)
    df_miRNA_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\训练集\2.0细胞_rna_mi_features.xlsx', header=None)
    df_drug_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\训练集\2.0细胞_rna_drug_features.xlsx', header=None)
    df_dis_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\训练集\2.0细胞_rna_disease_features.xlsx', header=None)
    df_loc = pd.read_excel(r'C:\HJH\plsm\独立测试\训练集\2.0的细胞定位.xlsx', index_col=0)
    df_loc_index = pd.read_excel(r'C:\HJH\plsm\独立测试\训练集\location_info_index.xlsx', index_col=0, header=None)
    df_kmer_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\训练集\2.0细胞_rna_kmer_features.xlsx', header=None)
    df_rckmer_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\训练集\2.0细胞_rna_rckmer_features.xlsx', header=None)
    df_RNAErnie_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\训练集\2.0细胞_rna_model_features.xlsx', header=None)
    df_protein_feature = pd.read_excel(r'C:\HJH\plsm\独立测试\训练集\2.0细胞_gate_feature_protein.xlsx', header=None)

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
    # miloc_feature = df_miloc_feature.values
    protein_feature = df_protein_feature.values

    merge_feature = np.concatenate((
        kmer_feature, rckmer_feature,
        seq_feature,
        dis_feature, drug_feature, miRNA_feature,
        RNAErnie_feature,
        # miloc_feature
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

    # # 收集需要过采样的样本
    # x_augmented = []
    # y_augmented = []
    #
    # # 遍历每个类别
    # for class_idx in range(y.shape[1]):
    #     # 获取该类别的样本索引
    #     class_indices = np.where(y[:, class_idx] == 1)[0]
    #     class_count = len(class_indices)
    #
    #     print(f"类别 {class_idx} ({class_name[class_idx]}) 有 {class_count} 个样本")
    #
    #     # 如果该类别样本数小于50
    #     if class_count < 47:
    #         # 复制该类别所有样本
    #         x_augmented.append(x[class_indices])
    #         y_augmented.append(y[class_indices])
    #
    # # 如果有需要过采样的样本
    # if x_augmented:
    #     # 将所有复制的样本合并
    #     x_augmented = np.vstack(x_augmented)
    #     y_augmented = np.vstack(y_augmented)
    #
    #     # 将原始数据和复制的数据合并
    #     x = np.vstack([x, x_augmented])
    #     y = np.vstack([y, y_augmented])
    #
    #     # 打乱数据
    #     x, y = shuffle(x, y, random_state=random_seed)
    # else:
    #     print("没有需要过采样的类别")




    auc_ls = [0] * 7
    aupr_ls = [0] * 7
    acc_ls = [0] * 7
    f1_ls = [0] * 7
    recall_ls = [0] * 7
    precision_ls = [0] * 7
    # 添加新的指标
    hamming_loss_ls = 0
    macro_f1_ls = 0
    micro_f1_ls = 0
    ranking_loss_ls = 0  # 排名损失


    from sklearn.metrics import roc_curve, precision_recall_curve

    # 初始化用于存储每类的 fpr/tpr 和 precision/recall
    fpr_dict = {i: [] for i in range(num_classes)}
    tpr_dict = {i: [] for i in range(num_classes)}
    precision_dict = {i: [] for i in range(num_classes)}
    recall_dict = {i: [] for i in range(num_classes)}

    import os
    import json
    import joblib

    save_dir = './best_model_repeat_独立测试111/'
    os.makedirs(save_dir, exist_ok=True)

    best_repeat_auc = -1
    best_repeat = -1
    best_metrics = None
    best_thresholds = None
    best_fold_results = None

    for repeat in range(1, 10):
        print(f"\n=== Repeat {repeat} / 100 ===")

        auc_ls = [0] * num_classes
        aupr_ls = [0] * num_classes
        acc_ls = [0] * num_classes
        f1_ls = [0] * num_classes
        recall_ls = [0] * num_classes
        precision_ls = [0] * num_classes
        # 为每个repeat初始化新的指标
        hamming_loss_ls = 0
        macro_f1_ls = 0
        micro_f1_ls = 0
        ranking_loss_ls = 0  # 排名损失

        thresholds_per_fold = []
        fold_pred_bin = []
        fold_pred_prob = []
        fold_true = []
        temp_model_paths = []

        X, Y = shuffle(x, y, random_state=random_seed + repeat)

        for fold in range(n_splits):
            print(f"  Fold {fold + 1}")
            fold_size = len(X) // n_splits
            val_idx = np.arange(fold * fold_size, (fold + 1) * fold_size)
            train_idx = np.setdiff1d(np.arange(len(X)), val_idx)
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]

            model = create_multi_label_model((len(merge_feature[0]),), num_classes)

            tau = [1.6259, 1.6085, 1.3893, 0.8611, 0.5407, 0.5324, 0.4422]

            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=[AUC(name='auc')])

            early_stop = EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True, mode='max')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

            model.fit(data_augmentation(X_train), Y_train,
                      epochs=100, batch_size=16, validation_split=0.2,
                      callbacks=[early_stop, reduce_lr], verbose=0)

            y_pred = model.predict(X_val)
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

            y_pred_bin = np.array([(y_pred[:, i] > thresholds[i]).astype(int) for i in range(num_classes)]).T

            for i in range(num_classes):
                if len(np.unique(Y_val[:, i])) > 1:
                    auc_ls[i] += roc_auc_score(Y_val[:, i], y_pred[:, i])
                    aupr_ls[i] += average_precision_score(Y_val[:, i], y_pred[:, i])
                acc_ls[i] += accuracy_score(Y_val[:, i], y_pred_bin[:, i])
                f1_ls[i] += f1_score(Y_val[:, i], y_pred_bin[:, i], zero_division=0)
                recall_ls[i] += recall_score(Y_val[:, i], y_pred_bin[:, i], zero_division=0)
                precision_ls[i] += precision_score(Y_val[:, i], y_pred_bin[:, i], zero_division=0)

            # 计算新的指标
            hamming_loss_ls += hamming_loss(Y_val, y_pred_bin)
            macro_f1_ls += f1_score(Y_val, y_pred_bin, average='macro', zero_division=0)
            micro_f1_ls += f1_score(Y_val, y_pred_bin, average='micro', zero_division=0)
            ranking_loss_ls += label_ranking_loss(Y_val, y_pred)  # 排名损失

            thresholds_per_fold.append(thresholds)
            fold_pred_bin.append(y_pred_bin)
            fold_pred_prob.append(y_pred)
            fold_true.append(Y_val)

            # 保存模型（暂时保留，稍后判断是否保留）
            model_path = os.path.join(save_dir, f'repeat_{repeat}_fold_{fold + 1}_model.h5')
            model.save_weights(model_path)
            temp_model_paths.append(model_path)

            del model
            tf.keras.backend.clear_session()
            gc.collect()

        mean_auc = np.mean(auc_ls) / n_splits
        print(f"Repeat {repeat} Mean AUC: {mean_auc:.4f}")

        if mean_auc > best_repeat_auc:
            # 删除旧模型文件
            if best_fold_results and 'models' in best_fold_results:
                for old_model_path in best_fold_results['models']:
                    if os.path.exists(old_model_path):
                        os.remove(old_model_path)

            best_repeat_auc = mean_auc
            best_repeat = repeat
            best_metrics = {
                'acc': np.array(acc_ls) / n_splits,
                'auc': np.array(auc_ls) / n_splits,
                'aupr': np.array(aupr_ls) / n_splits,
                'f1': np.array(f1_ls) / n_splits,
                'recall': np.array(recall_ls) / n_splits,
                'precision': np.array(precision_ls) / n_splits,
                # 添加新的指标
                'hamming_loss': hamming_loss_ls / n_splits,
                'macro_f1': macro_f1_ls / n_splits,
                'micro_f1': micro_f1_ls / n_splits,
                'ranking_loss': ranking_loss_ls / n_splits,  # 排名损失
            }
            best_thresholds = thresholds_per_fold
            best_fold_results = {
                'pred_bin': fold_pred_bin,
                'pred_prob': fold_pred_prob,
                'true': fold_true,
                'models': temp_model_paths,
            }
        else:
            # 不是最优 repeat，删除当前模型文件
            for tmp_path in temp_model_paths:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # 最后保存最优 repeat 的结果
    best_metrics_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in best_metrics.items()}
    json.dump(best_metrics_serializable, open(os.path.join(save_dir, f'repeat_{best_repeat}_best_metrics.json'), 'w'),
              indent=4)
    json.dump(best_thresholds, open(os.path.join(save_dir, f'repeat_{best_repeat}_thresholds.json'), 'w'), indent=4)
    joblib.dump(best_fold_results, os.path.join(save_dir, f'repeat_{best_repeat}_fold_results.pkl'))

    print(f"\n>>> Best Repeat: {best_repeat} with Average AUC: {best_repeat_auc:.4f}")

    # 打印每个类别的指标
    for i in range(num_classes):
        print(f"Class {class_name[i]}: ACC: {best_metrics['acc'][i]:.3f}, AUC: {best_metrics['auc'][i]:.3f}, "
              f"AUPR: {best_metrics['aupr'][i]:.3f}, F1: {best_metrics['f1'][i]:.3f}, "
              f"Recall: {best_metrics['recall'][i]:.3f}, Precision: {best_metrics['precision'][i]:.3f}")

    # 打印整体指标
    print(f"\nOverall Metrics:")
    print(f"Hamming Loss: {best_metrics['hamming_loss']:.4f}")
    print(f"Macro F1: {best_metrics['macro_f1']:.4f}")
    print(f"Micro F1: {best_metrics['micro_f1']:.4f}")
    print(f"Ranking Loss: {best_metrics['ranking_loss']:.4f}")  # 排名损失

    print(f"\nModel weights saved to: {save_dir}")