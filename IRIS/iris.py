# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:58:50 2017

@author: shweta
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
data = pd.read_csv("Iris.csv")

data_one_hot_encoded = pd.get_dummies(data)
train = data_one_hot_encoded.sample(frac=0.8, random_state=200)
test = data_one_hot_encoded.drop(train.index)

train_input = train.filter(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
train_label = train.filter(['Species_Iris-setosa', 'Species_Iris-versicolor', 'Species_Iris-virginica'])
test_input = test.filter(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
test_label = test.filter(['Species_Iris-setosa', 'Species_Iris-versicolor', 'Species_Iris-virginica'])
q =train.filter(['SepalLengthCm'])
p =train.filter(['PetalWidthCm'])

"""plt.scatter(q,p,color={'r','g','b'},s=30,cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()"""
sns.FacetGrid(data, hue="Species", size=5) \
   .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(train_input)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1],X_reduced[:, 2],c=train_label,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()

x = tf.placeholder(tf.float32,[None, 4])

W = tf.Variable(tf.zeros([4, 3]))
b = tf.Variable(tf.zeros([3]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 3])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
for _ in range(1000):
    #Usually send batches to the training step. But since the dataset is small sending all
    sess.run(train_step, feed_dict={x: train_input, y_: train_label})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy : ', sess.run(accuracy, feed_dict={x: test_input, y_: test_label}))