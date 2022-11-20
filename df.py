import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

df = pd.read_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\흥행수치(최종) (3).csv')
df = df[df['좌석수'] <= 300]
df1 = [i for i in df['흥행수치_1'] if i <= 1]
df2 = [i for i in df['흥행수치_2'] if i <= 1]

'''
k_means = KMeans(init="k=means++", n_clusters=4, n_init=12)
k_means.fit(df['흥행수치_1'])
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

fig = plt.figure(figsize=(6, 4))
color = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(4), color):
    my_members = (k_means_labels == k)

    # 중심 정의
    cluster_center = k_means_cluster_centers[k]

    # 중심 그리기
    ax.plot(df['흥행수치_1'][my_members, 0], df['흥행수치_1'][my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title('K-Means')
ax.set_xticks(())
ax.set_yticks(())
plt.show()
''' #클러스터링 시도

'''
fig, ax = plt.subplots(2, 2, figsize=(15, 13))
fig.subplots_adjust(hspace=0.2)

ax[0, 0].hist(df1)
ax[0, 0].set_title('criteria_1')

ax[0, 1].hist(df1, bins=100, linewidth=0.5, log=True)
ax[0, 1].set_title('criteria_1_loged')

ax[1, 0].hist(df2)
ax[1, 0].set_title('criteria_2')

ax[1, 1].hist(df2, bins=100, linewidth=0.5, log=True)
ax[1, 1].set_title('criteria_2_loged')

plt.show()
''' #히스토그램 분포

list_1 = []
for k in df['흥행수치_1']:
  if k <= 0.2:
    list_1.append(1)
  elif k <= 0.4:
    list_1.append(2)
  elif k <= 0.6:
    list_1.append(3)
  elif k <= 0.8:
    list_1.append(4)
  elif k <= 1:
    list_1.append(5)
  else:
    list_1.append(0)

list_2 = []
for k in df['흥행수치_2']:
  if k <= 0.2:
    list_2.append(1)
  elif k <= 0.4:
    list_2.append(2)
  elif k <= 0.6:
    list_2.append(3)
  elif k <= 0.8:
    list_2.append(4)
  elif k <= 1:
    list_2.append(4)
  else:
    list_2.append(0)

df['흥행수치_1_label'] = list_1
df['흥행수치_2_label'] = list_2

df.to_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\small_scale.csv')

#레이블 분리
