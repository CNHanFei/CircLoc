import pandas as pd

if __name__ == '__main__':
    # 读取 circRNA_miRNA_Matrix.xlsx 文件，header=0 表示第一行作为列名，index_col=0 表示第一列作为行名
    df_circRNA_miRNA = pd.read_excel(r'../dataset/circRNA_miRNA_Matrix.xlsx',
                                     header=0, index_col=0)
    # 获取矩阵数据（去除行和列名）
    circRNA_miRNA_matrix = df_circRNA_miRNA.values

    # 读取 miRNA_loc.xlsx 文件，header=0 表示第一行作为列名，index_col=0 表示第一列作为行名
    df_miRNA_loc = pd.read_excel(r'../dataset/miRNA_loc.xlsx', header=0,
                                 index_col=0)
    # 获取 miRNA 的位置信息数据
    miRNA_loc = df_miRNA_loc.values

    # 打印读取的数据的形状
    print("circRNA_miRNA_matrix shape:", circRNA_miRNA_matrix.shape)
    print("mRNA_loc shape:", miRNA_loc.shape)

    # 用来保存每个 circRNA 对应的 miRNA 位置信息比例的列表
    circRNA_miRNA_ratio_vector = []

    # 遍历 circRNA_miRNA_matrix 的每一行（每个 circRNA）
    for i in range(circRNA_miRNA_matrix.shape[0]):
        # 初始化 temp 列表，7 个位置的计数值，初始为0
        # 这7个位置对应miRNA的位置信息，每个值代表该位置上的 miRNA 的比例
        temp = [0, 0, 0, 0, 0, 0, 0]

        # 遍历 miRNA_miRNA_matrix 的每一列（每个 miRNA）
        for j in range(circRNA_miRNA_matrix.shape[1]):
            # 如果该位置的值为 1，说明当前的 circRNA 和该 miRNA 有关系
            if circRNA_miRNA_matrix[i][j] == 1:
                # 获取对应 j 行的 miRNA 定位信息
                for k in range(7):  # 遍历该 miRNA 的 7 个位置
                    # 如果该位置的值为 1，说明该 miRNA 在这个位置上有标记
                    if miRNA_loc[j][k] == 1:
                        temp[k] += 1  # 该位置计数增加

        # 处理完一个 circRNA 对应的所有 miRNA 关系后，进行归一化
        sum_number = sum(temp)  # 计算所有位置的总计数
        if sum_number != 0:
            # 归一化每个位置的计数，使其比例之和为 1
            for k in range(len(temp)):
                temp[k] = temp[k] / sum_number

        # 将每个 circRNA 对应的 miRNA 位置比例添加到列表中
        circRNA_miRNA_ratio_vector.append(temp)

    # 打印最终的结果，检查计算的 miRNA 位置信息比例
    print(circRNA_miRNA_ratio_vector)

    # 将结果保存为 Excel 文件
    output_file = r"../feature/circRNA_miRNA_loc_feature.xlsx"

    # 将结果转换为 pandas DataFrame，列名为位置的标签（7个位置），行名为 circRNA 的名字
    df_result = pd.DataFrame(circRNA_miRNA_ratio_vector, columns=["Cytoplasm", "Exomere", "Extracellular exosome",
                                                                  "Extracellular vesicle", "Microvesicle",
                                                                  "Mitochondrion", "Supermere"],
                             index=df_circRNA_miRNA.index)

    # 保存为 Excel 文件，index=False 表示不保存行名
    df_result.to_excel(output_file, index=True)

    # 打印保存完成的信息
    print(f"处理完成，结果已保存到 {output_file}")
