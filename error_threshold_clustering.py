import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """加载预测数据"""
    df = pd.read_csv('../dev-tools/filtered_data_3/MMF_GNN_POS_with_predictions.csv')
    print(f"数据加载完成，共有 {len(df)} 个化合物")
    return df

def calculate_errors(df):
    """计算预测误差"""
    # 计算绝对误差，使用正确的列名
    df['abs_error'] = abs(df['Predicted'] - df['Pimephales_promelas_toxicity'])
    return df

def binary_classification(df, threshold=30):
    """
    基于误差阈值进行二分类
    threshold: 误差阈值，高于此值为高误差，低于此值为低误差
    """
    df['error_class'] = np.where(df['abs_error'] >= threshold, 'High_Error', 'Low_Error')
    print("二分类结果:")
    print(df['error_class'].value_counts())
    return df

def calculate_molecular_features(smiles_list):
    """
    计算分子的理化特征和指纹
    """
    features = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"无法解析的SMILES: {smiles}")
                # 添加默认特征值
                feature_vector = [0] * 1035  # 11个描述符 + 1024个指纹位
                features.append(feature_vector)
                continue
                
            # 计算各种分子描述符
            mol_weight = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            num_atoms = mol.GetNumAtoms()
            num_heavy_atoms = mol.GetNumHeavyAtoms()
            num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            num_h_donors = Descriptors.NumHDonors(mol)
            num_h_acceptors = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            num_rings = Descriptors.RingCount(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            
            # 计算分子指纹特征
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            # 将ExplicitBitVect转换为numpy数组
            fp_arr = np.zeros((1024,))
            for idx in fp.GetOnBits():
                fp_arr[idx] = 1
            
            # 组合所有特征
            feature_vector = [mol_weight, logp, num_atoms, num_heavy_atoms, 
                            num_rotatable_bonds, num_h_donors, num_h_acceptors,
                            tpsa, num_rings, aromatic_rings] + list(fp_arr)
            features.append(feature_vector)
            
        except Exception as e:
            print(f"处理 {smiles} 时发生错误: {e}")
            # 添加默认特征值
            feature_vector = [0] * 1035  # 11个描述符 + 1024个指纹位
            features.append(feature_vector)
    
    return np.array(features)

def perform_clustering(data, n_clusters=3):
    """
    执行多种聚类算法
    """
    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_data)
    
    # DBSCAN聚类
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(scaled_data)
    
    return kmeans_labels, dbscan_labels, scaled_data

def analyze_clusters(df, kmeans_labels, dbscan_labels):
    """
    分析聚类结果
    """
    # 添加聚类标签到数据框
    df['kmeans_cluster'] = kmeans_labels
    df['dbscan_cluster'] = dbscan_labels
    
    print("\nK-means聚类统计:")
    kmeans_stats = df.groupby('kmeans_cluster')['abs_error'].agg(['count', 'mean', 'std']).reset_index()
    kmeans_stats.columns = ['Cluster', 'Count', 'Mean_Abs_Error', 'Std_Abs_Error']
    kmeans_stats['Method'] = 'KMeans'
    print(kmeans_stats)
    
    print("\nDBSCAN聚类统计:")
    dbscan_stats = df.groupby('dbscan_cluster')['abs_error'].agg(['count', 'mean', 'std']).reset_index()
    dbscan_stats.columns = ['Cluster', 'Count', 'Mean_Abs_Error', 'Std_Abs_Error']
    dbscan_stats['Method'] = 'DBSCAN'
    print(dbscan_stats)
    
    # 分析每个聚类中的误差类别分布
    print("\nK-means聚类中各误差类别的分布:")
    kmeans_cross = pd.crosstab(df['kmeans_cluster'], df['error_class'], normalize='index')
    print(kmeans_cross)
    
    print("\nDBSCAN聚类中各误差类别的分布:")
    dbscan_cross = pd.crosstab(df['dbscan_cluster'], df['error_class'], normalize='index')
    print(dbscan_cross)
    
    return kmeans_cross, dbscan_cross

def visualize_results(df, scaled_data, kmeans_labels, dbscan_labels):
    """
    可视化聚类结果
    """
    # 使用PCA降维以便可视化
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(scaled_data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 基于误差阈值的分类可视化
    error_colors = {'Low_Error': 'blue', 'High_Error': 'red'}
    colors = [error_colors[cls] for cls in df['error_class']]
    axes[0].scatter(data_pca[:, 0], data_pca[:, 1], c=colors, alpha=0.6)
    axes[0].set_title('基于误差阈值的二分类')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    
    # K-means聚类结果可视化
    scatter = axes[1].scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
    axes[1].set_title('K-means聚类结果')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[1])
    
    # DBSCAN聚类结果可视化
    scatter = axes[2].scatter(data_pca[:, 0], data_pca[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.6)
    axes[2].set_title('DBSCAN聚类结果')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('binary_classification_and_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 加载数据
    df = load_data()
    
    # 计算误差
    df = calculate_errors(df)
    
    # 基于误差阈值进行二分类
    df = binary_classification(df, threshold=30)
    
    # 计算分子特征
    print("\n正在计算分子特征...")
    features = calculate_molecular_features(df['SMILES'].tolist())
    print(f"特征矩阵形状: {features.shape}")
    
    # 执行聚类分析
    print("\n正在进行聚类分析...")
    kmeans_labels, dbscan_labels, scaled_data = perform_clustering(features, n_clusters=3)
    
    # 分析聚类结果
    kmeans_cross, dbscan_cross = analyze_clusters(df, kmeans_labels, dbscan_labels)
    
    # 可视化结果
    print("\n正在生成分析图表...")
    visualize_results(df, scaled_data, kmeans_labels, dbscan_labels)
    
    # 保存结果
    df.to_csv('binary_classification_and_clustering_results.csv', index=False)
    
    # 打印详细分析结果
    print("\n=== 详细分析结果 ===")
    print("1. 各误差类别中的统计信息:")
    error_stats = df.groupby('error_class')['abs_error'].agg(['count', 'mean', 'std'])
    print(error_stats.round(2))
    
    print("\n2. K-means聚类中各误差类别的分布:")
    print(kmeans_cross.round(3))
    
    print("\n3. DBSCAN聚类中各误差类别的分布:")
    print(dbscan_cross.round(3))

if __name__ == "__main__":
    main()