#%%
from email.encoders import encode_noop
from matplotlib import markers
import numpy as np
import pandas as pd
import os

s = os.path.join('https://archive.ics.uci.edu',
'ml','machine-learning-databases', 'iris', 'iris.data')

df = pd.read_csv(s, header=None, encoding='utf-8')
df.tail()

#%%
import matplotlib.pyplot as plt

y = df.iloc[0:100, 4].values  # 正解ラベルの取得
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='virginica')
plt.xlabel('selap length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# %%
from perceptron import Perceptron  # 自作モジュール

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of update')
plt.show()


# %% グラフ描画のための関数定義
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))]) # np.unique(ndarray) -> unique elements sorted

    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)): # ラベルでfor文を回してる (-1と1)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

# %% 決定領域のプロット
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# %% ADALINE での学習
from adaline import AdalineGD

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = AdalineGD(eta=0.01, n_iter=10).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_)+1,), np.log10(ada1.cost_))
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('log(error-square-sum)')
ax[0].set_title('Adaline - learning rate 0.01')

ada2 = AdalineGD(eta=0.0001, n_iter=10).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_)+1,), np.log10(ada2.cost_))
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('log(error-square-sum)')
ax[1].set_title('Adaline - learning rate 0.000s1')
plt.show()

# %%
