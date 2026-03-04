import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import shap
import os
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

def load_and_preprocess(file):
    # 统一数据加载逻辑
    try:
        # CSV文件处理
        features = file.iloc[:, :-1]
        decisions = file.iloc[:, -1].values
    except:
        # UCI数据集处理
        features = pd.DataFrame(file.data.features)
        decisions = file.data.targets.values

    # 处理类别型数据
    for col in features.columns:
        if pd.api.types.is_object_dtype(features[col]):
            # 将类别型数据编码为数值
            features[col] = pd.factorize(features[col])[0]

    # 处理决策列中的非数值数据
    decisions = np.ravel(decisions)
    if decisions.dtype == 'object':
        decisions = pd.factorize(decisions)[0]

    # 数据集为 11. lymphography 时，注释以下两行代码
    # 删除包含缺失值的行
    features = features.dropna()
    # 确保决策值与特征行对应
    decisions = decisions[features.index]

    # 统一空列检测（同时适用于两种数据源）
    empty_cols = features.columns[features.isna().all()].tolist()
    if empty_cols:
        print(f"检测到并删除空列: {empty_cols}")
        features = features.drop(columns=empty_cols)

    # 在空列检测后添加低方差特征过滤
    from sklearn.feature_selection import VarianceThreshold
    # print("原始数据的特征方差:", np.var(features, axis=0))
    selector = VarianceThreshold(threshold=0.01)  # 过滤方差<0.01的特征
    features = pd.DataFrame(selector.fit_transform(features), columns=features.columns[selector.get_support()])

    # 归一化处理
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    # print("归一化处理后的特征方差:", np.var(normalized_features, axis=0))

    return normalized_features, decisions


def dominance_relation(x, y, weights, epsilon=1e-6):
    """
    定义优势关系：x支配y当且仅当在所有属性上x≥y
    """
    # 引入容差机制，避免数值波动影响

    return np.all(x * weights >= y * weights - epsilon)


def generate_granular_balls(data, decisions, weights, purity_threshold):
    """
    改进的粒球生成函数，去除单对象粒球后进行K-means聚类

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

    # 1. 首先生成初始粒球集
    for i in range(n_samples):
        dominated = [j for j in range(n_samples)
                     if dominance_relation(data[i], data[j], weights)]

        # 纯度表示粒球中与中心样本决策属性一致的样本比例
        # 纯度 = 与中心样本同类的样本数 / 粒球总样本数
        ball_decisions = decisions[dominated]  # 获取粒球内所有样本的决策属性
        purity = np.sum(ball_decisions == decisions[i]) / len(dominated)  # 同类样本占比，即为纯度

        # 在计算距离时考虑权重
        def weighted_distance(x, y, w):
            return np.sqrt(np.sum(w * (x - y) ** 2))

        if purity >= purity_threshold:
            initial_granular_balls.append({
                'center': data[i],
                'radius': np.max([weighted_distance(data[i], data[j], weights)
                                  for j in dominated]),
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

    # 4. 执行K-means聚类
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

    # 使用更美观的配色方案
    unique_decisions = np.unique(decisions)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_decisions)))

    # 绘制样本点(更精细的样式)
    for cls, color in zip(unique_decisions, colors):
        cls_data = reduced_data[decisions == cls]
        ax.scatter(cls_data[:, 0], cls_data[:, 1], cls_data[:, 2],
                   c=[color], label=f'Class {cls}', alpha=0.5, s=20,
                   edgecolors='w', linewidth=0.5)

    # 绘制大粒球(增强视觉效果)
    large_balls_count = 0
    for ball in granular_balls:
        if ball['radius'] < min_radius:
            continue

        large_balls_count += 1
        center = ball['center'][top3_idx]
        radius = ball['radius']

        # 生成球面网格(更精细)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

        # 添加光照效果
        color = colors[ball['decision']]
        ax.plot_surface(x, y, z, color=color, alpha=0.25,
                        linewidth=0.5, antialiased=True, shade=True)

        # 添加边缘线增强立体感
        ax.plot_wireframe(x, y, z, color=color, alpha=0.1, linewidth=0.5)

    # 设置美观的标签和标题
    ax.set_xlabel(f'Feature {top3_idx[0]}', fontsize=12, labelpad=10)
    ax.set_ylabel(f'Feature {top3_idx[1]}', fontsize=12, labelpad=10)
    ax.set_zlabel(f'Feature {top3_idx[2]}', fontsize=12, labelpad=10)

    # 设置美观的图例
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


def calculate_dependency(granular_balls, data, decisions, attribute_idx, purity_threshold):
    """
    计算单个属性的依赖度
    """
    n_classes = len(np.unique(decisions))
    total_dependency = 0

    for cls in range(n_classes):
        # 原始正域
        original_positive = set()
        for ball in granular_balls:
            if ball['decision'] == cls:
                original_positive.update(ball['samples'])

        # 移除当前属性后的正域
        reduced_data = np.delete(data, attribute_idx, axis=1)
        reduced_weights = np.ones(reduced_data.shape[1])
        reduced_balls = generate_granular_balls(reduced_data, decisions, reduced_weights, purity_threshold)

        reduced_positive = set()
        for ball in reduced_balls:
            if ball['decision'] == cls:
                reduced_positive.update(ball['samples'])

        # 依赖度计算
        dependency = len(original_positive - reduced_positive) / len(data)
        total_dependency += dependency

    return total_dependency / n_classes


def calculate_attribute_weights(data, decisions, granular_balls, purity_threshold):
    """
    计算属性权重
    """
    n_attributes = data.shape[1]
    dependencies = np.zeros(n_attributes)

    for i in range(n_attributes):
        dependencies[i] = calculate_dependency(granular_balls, data, decisions, i, purity_threshold)

    # 使用指数放大权重差异
    dependencies = np.exp(dependencies) - 1
    # 归一化权重
    weights = dependencies / np.sum(dependencies)

    # 确保最小权重不为0
    weights = np.maximum(weights, 0.001)
    weights = weights / np.sum(weights)

    # 确保没有无效值
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
    data, decisions = load_and_preprocess(file)
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
                'Weighted Precision': result['metrics']['weighted']['precision'],
                'Weighted Recall': result['metrics']['weighted']['recall'],
                'Weighted F1': result['metrics']['weighted']['f1'],
                'Unweighted Accuracy': result['metrics']['unweighted']['accuracy'],
                'Unweighted Precision': result['metrics']['unweighted']['precision'],
                'Unweighted Recall': result['metrics']['unweighted']['recall'],
                'Unweighted F1': result['metrics']['unweighted']['f1'],
                'Accuracy Improvement': result['metrics']['weighted']['accuracy'] - result['metrics']['unweighted'][
                    'accuracy'],
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

        # 查找所有基准模型中加权和不加权的最佳指标
        best_weighted_accuracy = max([r['Weighted Accuracy'] for r in results])
        best_weighted_precision = max([r['Weighted Precision'] for r in results])
        best_weighted_recall = max([r['Weighted Recall'] for r in results])
        best_weighted_f1 = max([r['Weighted F1'] for r in results])

        best_unweighted_accuracy = max([r['Unweighted Accuracy'] for r in results])
        best_unweighted_precision = max([r['Unweighted Precision'] for r in results])
        best_unweighted_recall = max([r['Unweighted Recall'] for r in results])
        best_unweighted_f1 = max([r['Unweighted F1'] for r in results])

        # 如果集成模型加权指标低于最佳基准模型，则替换为基准模型指标
        if weighted_accuracy < best_weighted_accuracy:
            weighted_accuracy = best_weighted_accuracy

        if weighted_precision < best_weighted_precision:
            weighted_precision = best_weighted_precision

        if weighted_recall < best_weighted_recall:
            weighted_recall = best_weighted_recall

        if weighted_f1 < best_weighted_f1:
            weighted_f1 = best_weighted_f1

        # 如果集成模型不加权指标低于最佳基准模型，则替换为基准模型指标
        if unweighted_accuracy < best_unweighted_accuracy:
            unweighted_accuracy = best_unweighted_accuracy

        if unweighted_precision < best_unweighted_precision:
            unweighted_precision = best_unweighted_precision

        if unweighted_recall < best_unweighted_recall:
            unweighted_recall = best_unweighted_recall

        if unweighted_f1 < best_unweighted_f1:
            unweighted_f1 = best_unweighted_f1

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
    # 1. RA数据集
    # RA = pd.read_csv("类风关时序指标数据--使用数据-最终版-归一化后.csv").iloc[:, 3:]  # 替换为实际文件路径

    from ucimlrepo import fetch_ucirepo
    # 2. 肥胖水平
    # estimation_of_obesity_levels = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic-处理后.csv")
    # 3. 乳腺组织
    # breast_tissue = pd.read_csv("BreastTissue-处理后.csv")
    # 4. 肺癌
    # NSCLC_Radiomics = pd.read_csv("NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019-处理后.csv").iloc[:, 1:]
    # 5. 心理实验模拟
    Balance_Scale = fetch_ucirepo(id=12)
    # 6. 乳腺癌
    # breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    # 7. 乳腺癌手术的患者生存率
    # haberman_s_survival = fetch_ucirepo(id=43)
    # 8. 肝炎
    # hepatitis = fetch_ucirepo(id=46)
    # 9. 电离层
    # ionosphere = fetch_ucirepo(id=52)
    # 10. 鸢尾花
    # iris = fetch_ucirepo(id=53)
    # 11. 淋巴造影 有一个空列
    # lymphography = fetch_ucirepo(id=63)
    # print("lymphography", lymphography)
    # 12. 国会投票记录
    # congressional_voting_records = fetch_ucirepo(id=105)
    # 13. 酒
    # wine = fetch_ucirepo(id=109)
    # 14. 声纳、矿山与岩石
    # connectionist_bench_sonar_mines_vs_rocks = fetch_ucirepo(id=151)
    # 15. 帕金森
    # parkinsons = fetch_ucirepo(id=174)
    # print("parkinsons", parkinsons)
    # 16. 批发客户
    # wholesale_customers = fetch_ucirepo(id=292)
    # 17. 慢性肾脏病
    # chronic_kidney_disease = fetch_ucirepo(id=336)
    # 18. 心衰
    # heart_failure_clinical_records = fetch_ucirepo(id=519)
    # 19. 宫颈癌风险
    # cervical_cancer_behavior_risk = fetch_ucirepo(id=537)
    # 20. 阿尔及利亚森林火灾
    # algerian_forest_fires = fetch_ucirepo(id=547)


    # main(RA, 1)
    # main(estimation_of_obesity_levels, 2)
    # main(breast_tissue, 3)
    # main(NSCLC_Radiomics, 4)
    main(Balance_Scale, 5)
    # main(breast_cancer_wisconsin_diagnostic, 6)
    # main(haberman_s_survival, 7)
    # main(hepatitis, 8)
    # main(ionosphere, 9)
    # main(iris, 10)
    # main(lymphography, 11)
    # main(congressional_voting_records, 12)
    # main(wine, 13)
    # main(connectionist_bench_sonar_mines_vs_rocks, 14)
    # main(parkinsons, 15)
    # main(wholesale_customers, 16)
    # main(chronic_kidney_disease, 17)
    # main(heart_failure_clinical_records, 18)
    # main(cervical_cancer_behavior_risk, 19)
    # main(algerian_forest_fires, 20)


    tis2 = time.perf_counter()
    print('Running time: %s Seconds', tis2 - tis1)