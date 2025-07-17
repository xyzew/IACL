import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


def plot_tsne(features, labels):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''

    tsne = TSNE(n_components=2, init='pca', random_state=0)


    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4

    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    print('tsne_features的shape:', tsne_features.shape)
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1])  # 将对降维的特征进行可视化
    plt.show()

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", class_num),
                    data=df)
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    digits = datasets.load_digits(n_class=5)
    features, labels = digits.data, digits.target
    print(features.shape)
    print(labels.shape)
    plot_tsne(features, labels)
