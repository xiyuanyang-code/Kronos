import pandas as pd
from sklearn.decomposition import PCA


def pca(target_df, target_col, target):
    # 提取目标列数据
    data = target_df[target_col].values
    # 计算要保留的特征数量
    n_components = round(len(target_col) * target)
    # 初始化PCA模型
    pca_model = PCA(n_components=n_components)
    # 进行PCA降维
    pca_result = pca_model.fit_transform(data)
    # 将降维结果转换为DataFrame
    pca_df = pd.DataFrame(pca_result, columns=[f'alpha{i}' for i in range(n_components)])
    # 获取不需要进行PCA的列
    other_columns = [col for col in target_df.columns if col not in target_col]
    # 从原始DataFrame中选择不需要PCA的列
    other_df = target_df[other_columns]
    # 合并非PCA列和PCA结果
    result_df = pd.concat([other_df, pca_df], axis=1)
    return result_df
