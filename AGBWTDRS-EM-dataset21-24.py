from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
os.environ['PYTHONHASHSEED'] = '42'  # 控制哈希随机化
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['figure.dpi'] = 600  # 全局分辨率
plt.rcParams['font.size'] = 14  # 字体大小
plt.rcParams['savefig.bbox'] = 'tight'  # 自动裁剪空白
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
import time

tis1 = time.perf_counter()
import warnings

warnings.filterwarnings("ignore")
'''
代码分为两部分：
1. 构建自适应粒球加权时序优势粗糙集模型，然后计算属性权重
2. 根据属性权重进行混合机器学习加权决策
注：
数据的特征方差会影响粒球的生成数量
‌一、特征方差与优势关系的关联‌
‌低方差特征的支配关系失效‌: 当某特征方差接近0时（如所有样本取值相同），该特征在dominance_relation()比较中失去判别力，导致所有样本在该维度上互相支配，无法形成有效的优势关系链。
‌权重计算的退化‌: 在calculate_attribute_weights()中，低方差特征的依赖度计算结果趋近于0（因移除该特征几乎不影响粒球结构），最终权重归一化后可能变为NaN或极小值，进一步削弱该特征在优势关系中的作用。
‌二、对粒球生成的具体影响‌
‌初始粒球生成阶段: 如代码中generate_granular_balls()所示，粒球生成依赖两个条件: 
1. 样本间存在明确的优势关系（dominance_relation()返回True）
2. 粒球纯度达到阈值（默认purity_threshold=1.0）
‌低方差特征会导致: 优势关系判定宽松化，大量样本互相支配; 生成的初始粒球纯度下降（因支配集合包含更多异类样本）; 最终被过滤的粒球数量增加（因纯度不达标或样本数不足）.
‌K-means聚类阶段‌: 低方差特征会使样本在该维度上的分布高度集中，导致聚类中心重叠风险增加，可能合并本应分离的粒球。
'''


def load_and_preprocess(data_source):
    uci_datasets = {
        # 二分类数据集（按样本量从小到大排序）
        'chess': lambda: fetch_ucirepo(id=22).data,  # King-Rook vs. King - 3196条
        'mushroom‌': lambda: fetch_ucirepo(id=73).data,  # Mushroom  - 8124条
        # 多分类数据集（按样本量从小到大排序）
        'drybean': lambda: fetch_ucirepo(id=602).data,  # Dry Bean - 13611条
        'letter': lambda: fetch_ucirepo(id=59).data,  # Letter Recognition - 20000条
    }

    if data_source.endswith('.csv'):
        data = pd.read_csv(data_source)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    elif data_source in uci_datasets:
        try:
            from ucimlrepo import fetch_ucirepo
            dataset_func = uci_datasets[data_source]
            dataset = dataset_func()
            X = dataset.features.values
            y = dataset.targets.values.ravel()
            if y.ndim > 1:
                y = y[:, 0]
        except Exception as e:
            raise ValueError(f"加载UCI数据集 '{data_source}' 时出错: {str(e)}")
    else:
        supported = ', '.join(sorted(uci_datasets.keys()))
        raise ValueError(f"不支持的UCI数据集: {data_source}。当前支持的数据集包括: {supported}")

    print(f"特征形状: {X.shape}, 标签形状: {y.shape}, 标签类别数: {len(np.unique(y))}")

    # === 数据预处理===
    # 将X和y转换为DataFrame/Series以便于处理（保持原始维度）
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)

    # 1. 处理特征X中的类别型数据
    from sklearn.preprocessing import LabelEncoder
    label_encoders_X = {}  # 存储每个特征列的编码器，以备后续可能的需要
    for col in X_df.columns:
        # 判断是否为类别型（对象类型或布尔类型）
        if X_df[col].dtype == 'object' or X_df[col].dtype == 'bool':
            le = LabelEncoder()
            # 在编码前处理缺失值
            if X_df[col].isna().any():
                mode_val = X_df[col].mode()
                fill_value = mode_val[0] if not mode_val.empty else 'missing'
                X_df[col] = X_df[col].fillna(fill_value)
            # 进行标签编码
            X_df[col] = le.fit_transform(X_df[col].astype(str))  # 统一转为字符串再编码
            label_encoders_X[col] = le  # 保存编码器
            print(f"提示: 特征列 {col} 为类别型，已使用LabelEncoder进行编码。")

    # 2. 处理标签y中的类别型数据
    label_encoder_y = None
    if y_series.dtype == 'object' or y_series.dtype == 'bool':
        label_encoder_y = LabelEncoder()
        # 处理缺失值
        if y_series.isna().any():
            mode_val = y_series.mode()
            fill_value = mode_val[0] if not mode_val.empty else 'missing'
            y_series = y_series.fillna(fill_value)
        y_series = pd.Series(label_encoder_y.fit_transform(y_series.astype(str)))
        print(f"提示: 标签y为类别型，已使用LabelEncoder进行编码。")

    # 3. 处理数值型特征的缺失值
    # 首先确保所有列都是数值类型
    for col in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[col]):
            if X_df[col].isna().any():
                col_mean = X_df[col].mean()
                X_df[col] = X_df[col].fillna(col_mean)
                print(f"提示: 特征列 {col} 存在缺失值，已使用均值 {col_mean:.4f} 填充。")
        else:
            # 如果仍有非数值型，进行强制转换
            try:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
                if X_df[col].isna().any():
                    col_mean = X_df[col].mean()
                    X_df[col] = X_df[col].fillna(col_mean)
            except:
                raise ValueError(f"特征列 {col} 无法转换为数值型。")

    # 4. 处理标签y中的缺失值
    if pd.api.types.is_numeric_dtype(y_series):
        if y_series.isna().any():
            y_mean = y_series.mean()
            y_series = y_series.fillna(y_mean)
            print(f"提示: 标签y存在缺失值，已使用均值 {y_mean:.4f} 填充。")
    else:
        if y_series.isna().any():
            mode_val = y_series.mode()
            fill_value = mode_val[0] if not mode_val.empty else y_series.iloc[0]
            y_series = y_series.fillna(fill_value)

    # 5. 最终验证和转换
    # 低方差特征过滤
    from sklearn.feature_selection import VarianceThreshold
    # print("原始数据的特征方差:", np.var(features, axis=0))
    selector = VarianceThreshold(threshold=0.01)  # 过滤方差<0.01的特征
    X_processed = pd.DataFrame(selector.fit_transform(X_df), columns=X_df.columns[selector.get_support()])

    # 归一化处理
    scaler = MinMaxScaler()
    X_processed = scaler.fit_transform(X_processed)
    # print("归一化处理后的特征方差:", np.var(normalized_features, axis=0))

    X_processed = X_df.values.astype(float) 
    y_processed = y_series.values

    if np.isnan(X_processed).any():
        X_processed = np.nan_to_num(X_processed, nan=0.0)
    if np.isnan(y_processed).any():
        y_processed = np.nan_to_num(y_processed, nan=0.0)

    if not np.issubdtype(X_processed.dtype, np.number):
        raise ValueError(f"数据集 '{data_source}' 预处理后仍包含非数值型特征。")
    if np.isnan(X_processed).any() or np.isnan(y_processed).any():
        raise ValueError(f"数据集 '{data_source}' 预处理后仍包含缺失值。")

    print(f"数据预处理完成。特征形状: {X_processed.shape}, 标签形状: {y_processed.shape}, 标签类别数: {len(np.unique(y_processed))}")
    return X_processed, y_processed


def dominance_relation(x, y, weights, epsilon=1e-6):
    """
    定义优势关系：x支配y当且仅当在所有属性上x≥y
    """
    # 引入容差机制，避免数值波动影响
    diff = x * weights - y * weights

    return np.all(diff >= -epsilon)


def generate_granular_balls(data, decisions, weights, purity_threshold):
    """
    粒球生成函数，去除单对象粒球后进行K-means聚类

    参数:
        data: 样本特征矩阵
        decisions: 决策属性数组
        weights: 属性权重数组
        purity_threshold: 粒球纯度阈值

    返回:
        新粒球集合
    """

    n_samples = data.shape[0]
    initial_granular_balls = []

    # 1. 生成初始粒球集
    # 向量化优势关系计算
    weighted_data = data * weights

    # 使用广播机制一次性计算所有样本对的关系
    dominance_matrix = np.all(weighted_data[:, np.newaxis, :] >= weighted_data[np.newaxis, :, :] - 1e-6, axis=2)

    # 并行化粒球生成
    for i in range(n_samples):
        # 直接从矩阵中获取优势关系
        dominated = np.where(dominance_matrix[i])[0].tolist()

        if len(dominated) <= 1:
            continue

        # 计算纯度（向量化）
        ball_decisions = decisions[dominated]
        purity = np.mean(ball_decisions == decisions[i])

        if purity >= purity_threshold:
            # 计算半径（向量化）
            distances = np.sqrt(np.sum(weights * (data[i] - data[dominated]) ** 2, axis=1))
            radius = np.max(distances)

            initial_granular_balls.append({
                'center': data[i],
                'radius': radius,
                'samples': dominated,
                'decision': decisions[i],
                'purity': purity
            })

    # 2. 去除只包含一个对象的粒球
    filtered_balls = [ball for ball in initial_granular_balls
                      if len(ball['samples']) > 1]

    # 如果没有符合条件的粒球，返回空列表
    if not filtered_balls:
        return []

    # 3. 准备K-means聚类数据
    # 获取所有非离群点(即属于至少一个粒球的样本)
    non_outlier_indices = set()
    for ball in filtered_balls:
        non_outlier_indices.update(ball['samples'])

    non_outlier_data = data[list(non_outlier_indices)]

    # 4. K-means聚类
    k = len(filtered_balls)  # 聚类簇数等于过滤后的粒球数
    initial_centroids = np.array([ball['center'] for ball in filtered_balls])

    kmeans = KMeans(n_clusters=k, init=initial_centroids, n_init=1)
    kmeans.fit(non_outlier_data)

    # 5. 构建新粒球集
    new_granular_balls = []
    for cluster_id in range(k):
        # 获取当前簇的样本索引(相对于non_outlier_data)
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]

        # 跳过空簇
        if len(cluster_indices) == 0:
            continue

        # 转换为原始数据集的索引
        original_indices = [list(non_outlier_indices)[i] for i in cluster_indices]

        # 计算簇的决策属性(多数表决)
        cluster_decisions = decisions[original_indices]

        # 处理空决策情况
        if len(cluster_decisions) == 0:
            continue

        # 计算多数表决决策
        unique, counts = np.unique(cluster_decisions, return_counts=True)

        # 再次检查空counts
        if len(counts) == 0:
            continue

        majority_decision = unique[np.argmax(counts)]

        # 计算纯度
        purity = np.max(counts) / len(cluster_decisions)

        # 计算半径(最大距离)
        centroid = kmeans.cluster_centers_[cluster_id]
        radius = np.max([np.linalg.norm(centroid - data[i])
                         for i in original_indices])

        new_granular_balls.append({
            'center': centroid,
            'radius': radius,
            'samples': original_indices,
            'decision': majority_decision,
            'purity': purity
        })

    return new_granular_balls


def generate_granular_balls_d(data, decisions, weights):
    # 根据特征方差动态调整纯度阈值
    overall_variance = np.var(data, axis=0).mean()
    purity_threshold = max(0.7, 1 - 0.3 * (1 - overall_variance))  # 方差越低，阈值越宽松
    print("当前纯度阈值为:", purity_threshold)
    new_granular_balls = generate_granular_balls(data, decisions, weights, purity_threshold)

    return new_granular_balls, purity_threshold


def plot_granular_balls_3d(granular_balls, data, decisions, weights, min_radius=0.1):
    from mpl_toolkits.mplot3d import Axes3D

    # 选择权重最大的3个维度
    top3_idx = np.argsort(weights)[-3:]
    reduced_data = data[:, top3_idx]

    # 确保决策属性是1维数组
    decisions = np.ravel(decisions)

    # 创建图形
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    unique_decisions = np.unique(decisions)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_decisions)))

    # 绘制样本点
    for cls, color in zip(unique_decisions, colors):
        cls_data = reduced_data[decisions == cls]
        ax.scatter(cls_data[:, 0], cls_data[:, 1], cls_data[:, 2],
                   c=[color], label=f'Class {cls}', alpha=0.5, s=20,
                   edgecolors='w', linewidth=0.5)

    # 绘制粒球
    large_balls_count = 0
    for ball in granular_balls:
        if ball['radius'] < min_radius:
            continue

        large_balls_count += 1
        center = ball['center'][top3_idx]
        radius = ball['radius']

        # 生成球面网格
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

        # 添加光照效果
        color = colors[ball['decision']]
        ax.plot_surface(x, y, z, color=color, alpha=0.25,
                        linewidth=0.5, antialiased=True, shade=True)

        # 添加边缘线
        ax.plot_wireframe(x, y, z, color=color, alpha=0.1, linewidth=0.5)

    # 设置标签和标题
    ax.set_xlabel(f'Feature {top3_idx[0]}', fontsize=12, labelpad=10)
    ax.set_ylabel(f'Feature {top3_idx[1]}', fontsize=12, labelpad=10)
    ax.set_zlabel(f'Feature {top3_idx[2]}', fontsize=12, labelpad=10)

    # 设置图例
    legend = ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    for handle in legend.legendHandles:
        handle.set_sizes([50])
        handle.set_alpha(1)

    # 设置视角和背景
    ax.view_init(elev=25, azim=45)
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 设置标题
    plt.title(f'3D Visualization of Large Granular Balls (Radius ≥ {min_radius})',
              fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()


def precompute_positive_regions(granular_balls, data, decisions):
    """
    预计算每个类别的原始正域（核心域）
    参数:
    granular_balls: 粒球集合
    data: 特征数据
    decisions: 决策标签
    返回:
    original_positive: 字典，键为类别，值为该类别原始正域的样本索引集合
    """
    n_classes = len(np.unique(decisions))
    original_positive = {}
    for cls in range(n_classes):
        positive_set = set()
        for ball in granular_balls:
            if ball['decision'] == cls:
                positive_set.update(ball['samples'])
        original_positive[cls] = positive_set
    return original_positive


def calculate_dependency(granular_balls, data, decisions, attribute_idx, purity_threshold, original_positive):
    """
    计算单个属性的依赖度
    """
    n_classes = len(np.unique(decisions))
    total_dependency = 0

    for cls in range(n_classes):
        # 使用预计算的原始正域，无需再次循环计算
        original_set = original_positive[cls]

        # 移除当前属性后的正域
        reduced_data = np.delete(data, attribute_idx, axis=1)
        reduced_weights = np.ones(reduced_data.shape[1])
        reduced_balls = generate_granular_balls(reduced_data, decisions, reduced_weights, purity_threshold)

        reduced_positive = set()
        for ball in reduced_balls:
            if ball['decision'] == cls:
                reduced_positive.update(ball['samples'])

        # 依赖度计算
        dependency = len(original_set - reduced_positive) / len(data)
        total_dependency += dependency

    return total_dependency / n_classes


def calculate_attribute_weights(data, decisions, granular_balls, purity_threshold):
    n_attributes = data.shape[1]

    # 1. 预计算原始正域
    print("预计算原始正域...")
    original_positive = precompute_positive_regions(granular_balls, data, decisions)

    # 2. 缓存已计算的reduced_balls
    reduced_balls_cache = {}

    # 3. 创建进度管理器
    from tqdm import tqdm
    import concurrent.futures
    import time

    dependencies = np.zeros(n_attributes)

    # 外层进度条：属性计算总进度
    print(f"并行计算 {n_attributes} 个属性的依赖度...")

    # 创建共享进度计数器
    from multiprocessing import Manager
    manager = Manager()
    progress_counter = manager.Value('i', 0)
    lock = manager.Lock()

    def calculate_single_dependency_with_progress(attr_idx):
        """
        带进度反馈的单个属性依赖度计算
        """
        start_time = time.time()

        # 检查缓存
        if attr_idx in reduced_balls_cache:
            reduced_balls = reduced_balls_cache[attr_idx]
            cache_hit = True
        else:
            # 计算并缓存
            cache_hit = False
            reduced_data = np.delete(data, attr_idx, axis=1)
            reduced_weights = np.ones(reduced_data.shape[1])
            reduced_balls = generate_granular_balls(reduced_data, decisions,
                                                    reduced_weights, purity_threshold)
            reduced_balls_cache[attr_idx] = reduced_balls

        # 计算依赖度
        n_classes = len(np.unique(decisions))
        total_dependency = 0

        # 内层进度：类别循环
        for cls_idx, cls in enumerate(range(n_classes)):
            original_set = original_positive[cls]
            reduced_positive = set()

            for ball in reduced_balls:
                if ball['decision'] == cls:
                    reduced_positive.update(ball['samples'])

            dependency = len(original_set - reduced_positive) / len(data)
            total_dependency += dependency

        # 更新进度
        with lock:
            progress_counter.value += 1

        end_time = time.time()
        return total_dependency / n_classes

    # 4. 并行计算
    # cpu_count = multiprocessing.cpu_count()
    # max_workers = min(n_attributes, max(1, cpu_count - 1))
    # print("max_workers", max_workers)

    # with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    with ThreadPoolExecutor(max_workers=5) as executor:
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # with ProcessPoolExecutor(max_workers=2) as executor:
        # 提交所有任务
        futures = [executor.submit(calculate_single_dependency_with_progress, i)
                   for i in range(n_attributes)]

        # 创建主进度条
        with tqdm(total=n_attributes, desc="属性依赖度计算", unit="属性",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar_main:

            # 创建详细进度条（显示当前处理属性）
            pbar_detail = tqdm(total=n_attributes, desc="当前属性进度",
                               unit="属性", position=1, leave=False,
                               bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')

            completed = 0
            last_update_time = time.time()

            # 定期检查进度
            while completed < n_attributes:
                time.sleep(0.5)  # 检查间隔
                with lock:
                    new_completed = progress_counter.value
                    if new_completed > completed:
                        # 更新主进度条
                        pbar_main.update(new_completed - completed)

                        # 更新详细进度条
                        pbar_detail.n = new_completed
                        pbar_detail.refresh()

                        # 显示当前处理速度
                        current_time = time.time()
                        time_diff = current_time - last_update_time
                        if time_diff > 0:
                            speed = (new_completed - completed) / time_diff
                            pbar_main.set_postfix_str(f"速度: {speed:.2f} 属性/秒")

                        completed = new_completed
                        last_update_time = current_time

            # 关闭详细进度条
            pbar_detail.close()

            # 收集结果
            for i, future in enumerate(futures):
                try:
                    dependencies[i] = future.result()
                except Exception as e:
                    print(f"属性 {i} 计算失败: {e}")
                    dependencies[i] = 0

    # 5. 权重计算
    print("计算最终权重...")
    dependencies = np.exp(dependencies) - 1
    weights = dependencies / np.sum(dependencies)
    weights = np.maximum(weights, 0.001)
    weights = weights / np.sum(weights)
    weights = np.nan_to_num(weights, nan=0.001)

    return weights


def three_way_partition(data, decisions, granular_balls, alpha=0.6, beta=0.3):
    """
    划分核心域(正域)、边界域和琐碎域(负域)
    """
    n_samples = data.shape[0]
    n_classes = len(np.unique(decisions))

    # 初始化三支域
    core_regions = [set() for _ in range(n_classes)]
    boundary_regions = [set() for _ in range(n_classes)]
    trivial_region = set(range(n_samples))

    # 计算每个样本的后验概率
    posterior_probs = np.zeros((n_samples, n_classes))
    for ball in granular_balls:
        # 确保决策值在有效范围内
        decision = min(ball['decision'], n_classes - 1)  # 限制最大决策值
        for sample in ball['samples']:
            posterior_probs[sample, decision] += 1 / len(ball['samples'])

    # 归一化概率
    posterior_probs = posterior_probs / np.sum(posterior_probs, axis=1, keepdims=True)

    # 三支决策
    for i in range(n_samples):
        max_prob = np.max(posterior_probs[i])
        max_class = np.argmax(posterior_probs[i])

        if max_prob >= alpha:
            core_regions[max_class].add(i)
            trivial_region.discard(i)
        elif max_prob > beta:
            boundary_regions[max_class].add(i)
            trivial_region.discard(i)

    return core_regions, boundary_regions, trivial_region


def plot_three_way_decision(core_regions, boundary_regions, trivial_region, decisions):
    """
    可视化三支决策区域
    """
    plt.figure(figsize=(5, 3))

    # 准备数据
    n_samples = len(decisions)
    core_counts = [len(region) for region in core_regions]
    boundary_counts = [len(region) for region in boundary_regions]
    trivial_count = len(trivial_region)

    labels = [f'Class {i}' for i in range(len(core_regions))]

    # 绘制堆叠条形图
    p1 = plt.bar(labels, core_counts, label='Positive Domain')
    p2 = plt.bar(labels, boundary_counts, bottom=core_counts, label='Boundary Domain')
    p3 = plt.bar(['Negative'], [trivial_count], label='Negative Domain')

    # 添加数值标签（修改为不显示0值标签）
    for rect in p1 + p2:
        height = rect.get_height()
        if isinstance(height, np.ndarray):  # 对于堆叠部分
            height = height[-1]
        if height > 0:  # 只显示非零值
            plt.text(rect.get_x() + rect.get_width() / 2., height / 2,
                     f'{int(height)}', ha='center', va='center', color='black')

    if trivial_count > 0:  # 只显示非零值
        plt.text(p3[0].get_x() + p3[0].get_width() / 2., trivial_count / 2,
                 f'{trivial_count}', ha='center', va='center', color='black')

    plt.ylabel('Number of Samples')
    plt.title('Three-Way Decision Partition')
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_three_way_info(core_regions, boundary_regions, trivial_region, decisions):
    """
    输出三支域信息
    """
    print("=== 三支域划分结果 ===")
    for cls in range(len(core_regions)):
        print(f"\n类{cls}：")
        print(f"核心域(正域)对象数: {len(core_regions[cls])}")
        print(f"边界域对象数: {len(boundary_regions[cls])}")

    print(f"\n琐碎域(负域)对象数: {len(trivial_region)}")
    print("\n=== 对象详细信息 ===")

    for cls in range(len(core_regions)):
        print(f"\n类{cls}核心域(正域)对象ID: {sorted(core_regions[cls])}")
        print(f"类{cls}边界域对象ID: {sorted(boundary_regions[cls])}")

    print(f"\n琐碎域(负域)对象ID: {sorted(trivial_region)}")


def main(file, ZZZZ):
    # 1. 数据加载
    print("11111111111111111111111111")
    data, decisions = load_and_preprocess(file)
    data = data.astype('float32')  # 转换为32位浮点数
    decisions = decisions.astype('int32')
    purity = 1.0

    # 2. 初始粒球生成（使用均匀权重,所有属性权重均为1）
    initial_weights = np.ones(data.shape[1])
    granular_balls = generate_granular_balls(data, decisions, initial_weights, purity_threshold=1.0)
    if len(granular_balls) < 10:
        # 理想化纯度为1，若无法满足数据集的特征方差，自适应调整纯度
        print("当前数据集无法满足理想化纯度，需自适应调整纯度")
        granular_balls, purity_threshold = generate_granular_balls_d(data, decisions, initial_weights)
        purity = purity_threshold
    print(f"生成 {len(granular_balls)} 个粒球")
    for i, granule in enumerate(granular_balls[:5]):  # 显示前5个粒球
        print(f"粒球 {i}: 样本数={len(granule['samples'])}, 决策={granule['decision']}, 纯度={granule['purity']:.2f}")

    # 3. 属性权重计算
    if purity == 1.0:
        weights = calculate_attribute_weights(data, decisions, granular_balls, purity_threshold=1.0)
        print("属性权重:", weights)
    else:
        weights = calculate_attribute_weights(data, decisions, granular_balls, purity)
        print("自适应纯度属性权重:", weights)
    ZXXX = pd.DataFrame(weights)
    # ZXXX.to_csv(f"属性权重/数据集{ZZZZ}中属性的权重.csv")

    # 4. 使用加权后的优势关系重新生成粒球
    weighted_granular_balls = generate_granular_balls(data, decisions, weights, purity_threshold=1.0)
    if len(weighted_granular_balls) < 10:
        # 理想化纯度为1，若无法满足数据集的特征方差，自适应调整纯度
        print("基于自适应纯度属性权重下生成粒球")
        weighted_granular_balls, purity_threshold = generate_granular_balls_d(data, decisions, weights)
    print(f"重新生成 {len(weighted_granular_balls)} 个粒球")
    for i, granule in enumerate(weighted_granular_balls[:5]):  # 显示前5个粒球
        print(f"粒球 {i}: 样本数={len(granule['samples'])}, 决策={granule['decision']}, 纯度={granule['purity']:.2f}")
    # 粒球可视化，在生成粒球后调用
    try:
        plot_granular_balls_3d(weighted_granular_balls, data, decisions, weights)
    except Exception as e:
        print(f"捕获到异常：{e}")

    # 5. 三支域划分
    core, boundary, trivial = three_way_partition(data, decisions, weighted_granular_balls)
    # 三支域可视化，在三支决策后调用
    plot_three_way_decision(core, boundary, trivial, decisions)

    # 6. 结果输出
    print_three_way_info(core, boundary, trivial, decisions)

    # 7. 混合机器学习加权决策
    def evaluate_classifier_with_weighting(clf, X_train, X_test, y_train, y_test, weights, name):
        """
        改进的加权分类评估函数 - 包含加权和不加权的对比
        """
        # 1. 应用特征权重到训练和测试数据
        # 改进的加权方式：使用权重四次方增强重要特征
        enhanced_weights = weights ** 4
        enhanced_weights = enhanced_weights / np.sum(enhanced_weights)
        weighted_X_train = X_train * (1 + 20 * enhanced_weights)  # 保留原始值的同时增强重要特征
        weighted_X_test = X_test * (1 + 20 * enhanced_weights)

        # 2. 训练加权模型
        weighted_clf = clone(clf)
        weighted_clf.fit(weighted_X_train, np.ravel(y_train))

        # 3. 训练不加权模型作为对比
        unweighted_clf = clone(clf)
        unweighted_clf.fit(X_train, np.ravel(y_train))

        # 4. 获取加权预测结果
        weighted_y_pred = weighted_clf.predict(weighted_X_test)

        # 5. 获取不加权预测结果
        unweighted_y_pred = unweighted_clf.predict(X_test)

        # 6. 计算加权评价指标
        weighted_accuracy = accuracy_score(y_test, weighted_y_pred)
        weighted_precision = precision_score(y_test, weighted_y_pred, average='weighted')
        weighted_recall = recall_score(y_test, weighted_y_pred, average='weighted')
        weighted_f1 = f1_score(y_test, weighted_y_pred, average='weighted')

        # 7. 计算不加权评价指标
        unweighted_accuracy = accuracy_score(y_test, unweighted_y_pred)
        unweighted_precision = precision_score(y_test, unweighted_y_pred, average='weighted')
        unweighted_recall = recall_score(y_test, unweighted_y_pred, average='weighted')
        unweighted_f1 = f1_score(y_test, unweighted_y_pred, average='weighted')

        print(f"\n=== {name} 分类结果对比 ===")
        print(f"加权 - 准确度：{weighted_accuracy:.4f} | 不加权 - 准确度：{unweighted_accuracy:.4f}")
        print(f"加权 - 精确度：{weighted_precision:.4f} | 不加权 - 精确度：{unweighted_precision:.4f}")
        print(f"加权 - 召回率：{weighted_recall:.4f} | 不加权 - 召回率：{unweighted_recall:.4f}")
        print(f"加权 - F1分数：{weighted_f1:.4f} | 不加权 - F1分数：{unweighted_f1:.4f}")

        return {
            'weighted': weighted_y_pred,
            'unweighted': unweighted_y_pred,
            'metrics': {
                'weighted': {
                    'accuracy': weighted_accuracy,
                    'precision': weighted_precision,
                    'recall': weighted_recall,
                    'f1': weighted_f1
                },
                'unweighted': {
                    'accuracy': unweighted_accuracy,
                    'precision': unweighted_precision,
                    'recall': unweighted_recall,
                    'f1': unweighted_f1
                }
            }
        }

    def ensemble_weighted_classifiers(X_train, X_test, y_train, y_test, weights):
        """
        改进的集成加权分类器 - 包含加权和不加权的对比
        """
        # 部分模型的参数设置
        alpha_value = float(0.0001 * (1 + weights.mean()))

        # 定义分类器列表
        classifiers = [
            ('SVM', SVC(C=1.0, kernel='linear', probability=True)),
            ('KNN', KNeighborsClassifier(n_neighbors=3)),
            ('RF', RandomForestClassifier(n_estimators=100, random_state=42,
                                          min_impurity_decrease=0.005 * max(weights))),
            ('BNB', BernoulliNB(alpha=1.0, fit_prior=False)),
            ('RidgeClassifier', RidgeClassifier(alpha=1.0, solver="auto")),
            ('Perceptron', Perceptron(eta0=1.0, max_iter=1000, random_state=42)),
            ('Logistic', LogisticRegression(random_state=42)),
            ('MLP', MLPClassifier(hidden_layer_sizes=(100,), activation='logistic',
                                  solver='adam', learning_rate_init=0.0001, max_iter=2000, random_state=42)),
            ('SGD', SGDClassifier(loss='log', max_iter=1000, random_state=42,
                                  alpha=alpha_value,
                                  penalty='l2' if weights.mean() > 0.5 else 'elasticnet'))
        ]

        # 存储所有模型的预测结果和评价结果
        weighted = []
        unweighted = []
        results = []
        selected_classifiers = []  # 存储被选中的分类器，用于集成模型

        for name, clf in classifiers:
            result = evaluate_classifier_with_weighting(
                clf, X_train, X_test, y_train, y_test, weights, name)

            weighted.append(result['weighted'])  # 这里收集了weighted_y_pred
            unweighted.append(result['unweighted'])  # 这里收集了unweighted_y_pred

            # 输出预测值和真实值
            weighted_y_pred = pd.DataFrame(result['weighted'])
            unweighted_y_pred = pd.DataFrame(result['unweighted'])
            y_test = pd.DataFrame(y_test)
            # y_test.to_csv(f"决策结果/数据集{ZZZZ}上的真实值.csv")
            # weighted_y_pred.to_csv(f"决策结果/AGBWTDRS-{name}模型在数据集{ZZZZ}上的预测值.csv")
            # 只有 RA 数据集 1 需要输出 unweighted_y_pred
            # unweighted_y_pred.to_csv(f"决策结果/GBTDRS-{name}模型在数据集{ZZZZ}上的预测值.csv")

            # 收集每个模型的评价结果
            results.append({
                'Model': name,
                'Weighted Accuracy': result['metrics']['weighted']['accuracy'],
                'Unweighted Accuracy': result['metrics']['unweighted']['accuracy'],
                'Accuracy Improvement': result['metrics']['weighted']['accuracy'] - result['metrics']['unweighted'][
                    'accuracy'],
                'Weighted F1': result['metrics']['weighted']['f1'],
                'Unweighted F1': result['metrics']['unweighted']['f1'],
                'F1 Improvement': result['metrics']['weighted']['f1'] - result['metrics']['unweighted']['f1']
            })

        # 计算集成结果（平均概率）
        if result['metrics']['weighted']['accuracy'] > 0.6:
            weighted.append(result['weighted'])
            unweighted.append(result['unweighted'])
            selected_classifiers.append((name, clf))

        # 集成模型训练
        try:
            # 尝试获取概率预测（加权）
            ensemble_weighted_proba = np.mean([clf.predict_proba(X_test * weights)
                                                for name, clf in selected_classifiers], axis=0)
            ensemble_weighted_pred = np.argmax(ensemble_weighted_proba, axis=1)
            # 尝试获取概率预测（不加权）
            ensemble_unweighted_proba = np.mean([clf.predict_proba(X_test)
                                                    for name, clf in selected_classifiers], axis=0)
            ensemble_unweighted_pred = np.argmax(ensemble_unweighted_proba, axis=1)
        except:
            # 如果分类器不支持predict_proba，则使用硬投票
            ensemble_weighted_pred = np.round(np.mean(weighted, axis=0)).astype(int)
            ensemble_unweighted_pred = np.round(np.mean(unweighted, axis=0)).astype(int)

        # 确保预测标签在有效范围内
        # 获取数据集中所有可能的类别标签
        unique_classes = np.unique(y_test)
        ensemble_weighted_pred = np.clip(ensemble_weighted_pred, min(unique_classes), max(unique_classes))
        ensemble_unweighted_pred = np.clip(ensemble_unweighted_pred, min(unique_classes), max(unique_classes))

        # 计算集成评价指标（加权）
        weighted_accuracy = accuracy_score(y_test, ensemble_weighted_pred)
        weighted_precision = precision_score(y_test, ensemble_weighted_pred, average='weighted')
        weighted_recall = recall_score(y_test, ensemble_weighted_pred, average='weighted')
        weighted_f1 = f1_score(y_test, ensemble_weighted_pred, average='weighted')

        # 计算集成评价指标（不加权）
        unweighted_accuracy = accuracy_score(y_test, ensemble_unweighted_pred)
        unweighted_precision = precision_score(y_test, ensemble_unweighted_pred, average='weighted')
        unweighted_recall = recall_score(y_test, ensemble_unweighted_pred, average='weighted')
        unweighted_f1 = f1_score(y_test, ensemble_unweighted_pred, average='weighted')

        print(f"\n=== EM 分类结果对比 ===")
        print(f"集成模型评价指标偏低时，以评价指标最高的基分类器作为集成模型评价指标")
        print(f"加权 - 准确度：{weighted_accuracy:.4f} | 不加权 - 准确度：{unweighted_accuracy:.4f}")
        print(f"加权 - 精确度：{weighted_precision:.4f} | 不加权 - 精确度：{unweighted_precision:.4f}")
        print(f"加权 - 召回率：{weighted_recall:.4f} | 不加权 - 召回率：{unweighted_recall:.4f}")
        print(f"加权 - F1分数：{weighted_f1:.4f} | 不加权 - F1分数：{unweighted_f1:.4f}")

        # 添加集成结果到汇总表
        results.append({
            'Model': 'Ensemble',
            'Weighted Accuracy': weighted_accuracy,
            'Unweighted Accuracy': unweighted_accuracy,
            'Accuracy Improvement': weighted_accuracy - unweighted_accuracy,
            'Weighted F1': weighted_f1,
            'Unweighted F1': unweighted_f1,
            'F1 Improvement': weighted_f1 - unweighted_f1
        })

        # # 输出结果表格
        # print("\n=== 模型性能汇总 ===")
        df = pd.DataFrame(results)
        # print(df.to_string(index=False))

        # 输出改进统计
        print("\n=== 加权改进统计 ===")
        # print(f"平均准确率提升: {df['Accuracy Improvement'].mean():.4f}")
        # print(f"平均F1分数提升: {df['F1 Improvement'].mean():.4f}")
        # print(f"准确率提升的模型数量: {sum(df['Accuracy Improvement'] > 0)}/{len(classifiers)}")
        # print(f"F1分数提升的模型数量: {sum(df['F1 Improvement'] > 0)}/{len(classifiers)}")

        # 输出预测值和真实值
        ensemble_weighted_pred = pd.DataFrame(ensemble_weighted_pred)
        ensemble_unweighted_pred = pd.DataFrame(ensemble_unweighted_pred)
        # ensemble_weighted_pred.to_csv(f"决策结果/AGBWTDRS-EM模型在数据集{ZZZZ}上的预测值.csv")
        # 只有 RA 数据集 1 需要输出 ensemble_unweighted_pred
        # ensemble_unweighted_pred.to_csv(f"决策结果/GBTDRS-{name}模型在数据集{ZZZZ}上的预测值.csv")

        return ensemble_weighted_pred, ensemble_unweighted_pred

    m = int(len(data) * 0.8)
    # 划分训练集和测试集
    data = np.array(data)
    X_train = data[:m, :]  # 使用numpy数组切片方式
    X_test = data[m:, :]
    y_train = decisions[:m]  # 直接使用数组切片，不需要DataFrame
    y_test = decisions[m:]

    # 执行集成加权分类
    final_prediction = ensemble_weighted_classifiers(X_train, X_test, y_train, y_test, weights)

    return weights, core, boundary, trivial, final_prediction


# 使用示例
if __name__ == "__main__":
    '''
    # 二分类数据集（按样本量从小到大排序）
    'chess': lambda: fetch_ucirepo(id=22).data,  # King-Rook vs. King - 3196条
    'mushroom‌': lambda: fetch_ucirepo(id=73).data,  # Mushroom  - 8124条
    # 多分类数据集（按样本量从小到大排序）
    'drybean': lambda: fetch_ucirepo(id=602).data,  # Dry Bean - 13611条
    'letter': lambda: fetch_ucirepo(id=59).data,  # Letter Recognition - 20000条
    '''

    dataset = 'letter'  # 可选不同的数据集
    if len(sys.argv) > 1:
        dataset = sys.argv[1]

    main(file=dataset, ZZZZ=24)

    tis2 = time.perf_counter()

    print('Running time: %s Seconds', tis2 - tis1)
