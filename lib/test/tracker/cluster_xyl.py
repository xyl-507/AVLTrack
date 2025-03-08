import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np

# from kmeans import kmeans
# from TSNE import tsne as tsne_hgd
from sklearn.manifold import TSNE as tsne_scikit
from openTSNE import TSNE as tsne_open
from sklearn.decomposition import PCA  # pip install scikit-learn
from sklearn.cluster import KMeans
import time

# global h,m,s
# t = time.localtime()  # 获取时间,生成文件名后缀
# hour=t.tm_hour
# minute=t.tm_min
# sec=t.tm_sec

# cluster_kmeans(y, topk=300, n_components=2, pca=False, openTSNE=True, scikit=False, hgd=False, title='Local Representation')

def cluster_kmeans(x, topk=500, n_components=2, pca=False, openTSNE=False, scikit=False, hgd=False, plot_kmeans=False, title=''):
    """
    用法：cluster_kmeans(x=p2, n_components=2, scikit=True)  # pca=True, openTSNE=False, scikit=False, hgd=False, plot_kmeans=False, title=''
    其中：
    hgd是哈工大的t-SNE代码 https://github.com/heucoder/dimensionality_reduction_alo_codes
    openTSNE是 https://opentsne.readthedocs.io/en/latest/examples/index.html
    scikit是 pip install scikit-learn
    """
    t = time.localtime()  # 获取时间,生成文件名后缀
    hour=t.tm_hour
    minute=t.tm_min
    sec=t.tm_sec
    global pca_result
    # 可视化输入x
    # for k in x:
    #     show_tensor(k.sum(dim=0), 8, 'input')

    # 定义聚类的参数和输入
    b, c, h, w = x.shape  # p2 = 1*70*70*512
    n = n_components  # n 是聚类数
    # x = x.view(-1, c).detach().cpu().numpy()  # 输入
    x = x.squeeze(dim=0).permute(2, 1, 0).reshape(-1, c).detach().cpu().numpy()  # 输入
    # plot_only = h * w  # h*w是全部点都画 / =500 是只画前500个点
    plot_only = topk  # h*w是全部点都画 / =500 是只画前500个点
    x_cluster = x[:plot_only, :]

    # 使用kmeans 得到t-SNE需要的标签
    # labels = kmeans(x, n=n, title=title)  # n 是聚类数
    labels = kmeans(x_cluster, n=n, title=title, plot_kmeans=plot_kmeans)  # n 是聚类数
    labels = labels[:plot_only]

    print("lable_pred's shape is : {} \nlable_pred: {}".format(labels.shape, labels))

    if pca:  # PCA 降维至 2 维
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(x_cluster)
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, label="PCA")  # 使用kmeans预测的标签
        plt.title("PCA")
        plt.colorbar()
        # plt.show()  # plt.show()函数来实现图像的停留显示，导致后面的无法运行
        plt.pause(0.001)

    if openTSNE:  # 使用openTSNE 的t-SNE， 好用就是需要标签y，才能对每个点画颜色
        tsne = tsne_open(perplexity=30, n_components=n, metric="euclidean", n_jobs=8, random_state=42, verbose=True)
        embedding_train = tsne.fit(x_cluster)  # x_cluster / pca_result
        # colors = np.random.rand(4900)
        # colors = np.stack((np.zeros(2450), np.ones(2450)))  # 纵向堆叠两个数组
        plt.figure()
        plt.scatter(embedding_train[:, 0], embedding_train[:, 1], c=labels, label="t-SNE", cmap='bwr')  # 使用kmeans预测的标签, cmap='viridis'
        plt.title("t-SNE of openTSNE")
        plt.colorbar()
        plt.savefig('/home/xyl/newdrive/xyl-code2/All-in-One-ACMMM2023/output/vis/t-sne/{}_{}_{}-t-SNE-{}.png'.format(hour, minute, sec, title), dpi=600)
        # plt.show()  # plt.show()函数来实现图像的停留显示，导致后面的无法运行
        plt.pause(0.001)

    if scikit:  # 使用scikit-learn的t-SNE
        tsne = tsne_scikit(perplexity=30, n_components=n, init='pca', n_iter=5000)
        # 对中间层输出进行tsne降维
        low_dim_embs = tsne.fit_transform(x_cluster)
        # low_dim_embs = tsne.fit_transform(pca_result) # 在PCA降维的基础上进行聚类分析
        colors = np.random.rand(500)
        plt.figure()
        plt.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], c=labels, label="t-SNE", cmap='bwr')
        plt.title("t-SNE of scikit-learn")
        plt.colorbar()
        plt.savefig('/home/xyl/newdrive/xyl-code2/All-in-One-ACMMM2023/output/vis/t-sne/{}_{}_{}-scikit-{}.png'.format(hour, minute, sec, title), dpi=600)
        # plt.show()  # plt.show()函数来实现图像的停留显示，导致后面的无法运行
        plt.pause(0.001)

    if hgd:  # 使用哈工大写的t-SNE
        data_2d = tsne_hgd(x_cluster, no_dims=n)  # no_dims为聚类数
        # plt.scatter(data_2d[:, 0], data_2d[:, 1], c = Y)
        plt.figure()
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='bwr')  # p2 = 1*70*70*512
        plt.title("t-SNE of hgd")
        plt.colorbar()
        plt.savefig('/home/xyl/newdrive/xyl-code2/All-in-One-ACMMM2023/output/vis/t-sne/{}_{}_{}-hgd-{}.png'.format(hour, minute, sec, title), dpi=600)
        # plt.show()  # plt.show()函数来实现图像的停留显示，导致后面的无法运行
        plt.pause(0.001)
    return labels

# kmeans from https://www.jb51.net/article/129821.htm
def kmeans(data, n=3, title='', plot_kmeans=False):
    data = data
    estimator = KMeans(n_clusters=n)
    res = estimator.fit_predict(data)
    lable_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    inertia = estimator.inertia_
    # print("res", res)
    # print("lable_pred", lable_pred)
    # print("centroids",centroids)
    # print("inertia",inertia)

    # 画图
    if plot_kmeans:
        plt.figure()
        for i in range(len(data)):
            if int(lable_pred[i]) == 0:
                plt.scatter(data[i][0], data[i][1], color='red')
            if int(lable_pred[i]) == 1:
                plt.scatter(data[i][0], data[i][1], color='black')
            if int(lable_pred[i]) == 2:
                plt.scatter(data[i][0], data[i][1], color='blue')
        plt.title("The results of K-means")
        # plt.show()  # plt.show()函数来实现图像的停留显示，导致后面的无法运行
        plt.savefig(r'/home/xyl/newdrive/xyl-code2/All-in-One-ACMMM2023/output/vis/t-sne\{}_{}_{}-kmeans-{}.png'.format(hour, minute, sec, title))
        plt.pause(0.001)
    return lable_pred


def show_tensor(a: torch.Tensor, fig_num=None, title=None):
    """Display a 2D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    # a_np = a
    if a_np.ndim == 3:
        a_np = np.transpose(a_np, (1, 2, 0))
    plt.figure(fig_num)
    plt.tight_layout()
    # plt.tight_layout(pad=0)  # 也可以设为默认的，但这样图大点
    plt.cla()
    plt.imshow(a_np)
    # plt.imshow(a_np.astype('uint8')) # 画彩色图 https://blog.csdn.net/weixin_43669978/article/details/121963218
    plt.axis('off')
    plt.axis('equal')
    plt.colorbar()  # 创建颜色条
    if title is not None:
        plt.title(title)
    plt.draw()
    # plt.show()  # imshow是对图像的处理，show是展示图片
    plt.pause(0.1)


# ---------------------------------------------------------------- 哈工大的t-SNE代码, 运行较慢
import time


def tsne_hgd(x, no_dims=2, perplexity=30.0, max_iter=1000):
    """Runs t-SNE on the dataset in the NxD array x
    to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(x, no_dims, perplexity),
    where x is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array x should have type float.")
        return -1

    (n, d) = x.shape

    # 动量
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    # 随机初始化Y
    y = np.random.randn(n, no_dims)
    # dy梯度
    dy = np.zeros((n, no_dims))
    # iy是什么
    iy = np.zeros((n, no_dims))

    gains = np.ones((n, no_dims))

    # 对称化
    P = seach_prob(x, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)  # pij
    # early exaggeration
    # pi\j，提前夸大
    print("T-SNE DURING:%s" % time.clock())
    P = P * 4
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_y = np.sum(np.square(y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)  # qij
        Q = np.maximum(Q, 1e-12)  # X与Y逐位比较取其大者

        # Compute gradient
        # np.tile(A,N) 重复数组AN次 [1],5 [1,1,1,1,1]
        # pij-qij
        PQ = P - Q
        # 梯度dy
        for i in range(n):
            dy[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (y[i, :] - y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        # 迭代
        iy = momentum * iy - eta * (gains * dy)
        y = y + iy
        y = y - np.tile(np.mean(y, 0), (n, 1))
        # Compute current value of cost function\
        if (iter + 1) % 100 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration ", (iter + 1), ": error is ", C)
            if (iter + 1) != 100:
                ratio = C / oldC
                print("ratio ", ratio)
                if ratio >= 0.95:
                    break
            oldC = C
        # Stop lying about P-values
        if iter == 100:
            P = P / 4
    print("finished training!")
    return y


def cal_pairwise_dist(x):
    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    # 返回任意两个点之间距离的平方
    return dist


def cal_perplexity(dist, idx=0, beta=1.0):
    '''计算perplexity, D是距离向量，
    idx指dist中自己与自己距离的位置，beta是高斯分布参数
    这里的perp仅计算了熵，方便计算
    '''
    prob = np.exp(-dist * beta)
    # 设置自身prob为0
    prob[idx] = 0
    sum_prob = np.sum(prob)
    if sum_prob < 1e-12:
        prob = np.maximum(prob, 1e-12)
        perp = -12
    else:
        perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob
        prob /= sum_prob

    return perp, prob


def seach_prob(x, tol=1e-5, perplexity=30.0):
    '''二分搜索寻找beta,并计算pairwise的prob
    '''

    # 初始化参数
    print("Computing pairwise distances...")
    (n, d) = x.shape
    dist = cal_pairwise_dist(x)
    dist[dist < 0] = 0
    pair_prob = np.zeros((n, n))
    beta = np.ones((n, 1))
    # 取log，方便后续计算
    base_perp = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %s of %s ..." % (i, n))

        betamin = -np.inf
        betamax = np.inf
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])

        # 二分搜索,寻找最佳sigma下的prob
        perp_diff = perp - base_perp
        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # 更新perb,prob值
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        # 记录prob值
        pair_prob[i,] = this_prob
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    # 每个点对其他点的条件概率分布pi\j
    return pair_prob