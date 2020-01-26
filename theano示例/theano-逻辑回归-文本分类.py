# coding:utf-8
"""
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入全部的训练集和测试集
news = fetch_20newsgroups(subset="all")
# 数据集划分为训练集、验证集
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, random_state=42)
# 向量化
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)

import numpy
from theano import theano
from theano import sparse
import theano.tensor as T

# 输入文本向量维度
n_in = X_train.shape[1]
# 输出类别数
n_out=len(numpy.unique(y_train))

# 输入文本集向量变量
x = T.matrix('x')
# 输出类别向量（类别分布）
y = T.ivector('y')
# 模型参数W矩阵
W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
# 模型参数b向量
b = theano.shared(
    value=numpy.zeros(
        (n_out,),
        dtype=theano.config.floatX
    ),
    name='b',
    borrow=True
)
# 模型预测公式
model = T.nnet.softmax(T.dot(x, W) + b)
# 选择输出类别分布中概率最大的类别的序号
y_pred = T.argmax(model, axis=1)
# 误差计算公式 预测错误的实例数/实例总数
error = T.mean(T.neq(y_pred, y))
# 损失函数
cost = -T.mean(T.log(model)[T.arange(y.shape[0]), y])
# 损失函数对W求梯度
g_W = T.grad(cost=cost, wrt=W)
# 损失函数对b求梯度
g_b = T.grad(cost=cost, wrt=b)

# 编译
learning_rate=0.13
train_model = theano.function(
    inputs=[x, y],
    outputs=[cost,error],
    updates=[(W, W - learning_rate * g_W),(b, b - learning_rate * g_b)],
)

validate_model = theano.function(
    inputs=[x,y],
    outputs=[cost,error]
)

# epochs数量（遍历训练集的次数）
n_epochs = 100
# 一个批次含有的样本量
batch_size = 600

# 遍历训练集所有实例所需批次数量
n_train_batches = X_train.shape[0] // batch_size
# 训练过程需要的总迭代次数
n_iters = n_epochs * n_train_batches
# 记录每次迭代的训练集损失
train_loss = numpy.zeros(n_iters)
# 记录每次迭代的训练集误差
train_error = numpy.zeros(n_iters)
# 每100次迭代后，计算1次验证损失
validation_interval = 100

# 遍历验证集所有实例所需批次数量
n_valid_batches = X_test.shape[0] // batch_size
# 记录每次迭代的验证集损失
valid_loss = numpy.zeros(n_iters / validation_interval)
# 记录每次迭代的验证集误差
valid_error = numpy.zeros(n_iters / validation_interval)


for epoch in range(n_epochs):
    for minibatch_index in range(n_train_batches):
        iteration = minibatch_index + n_train_batches * epoch
        train_loss[iteration], train_error[iteration] = train_model(numpy.asarray(X_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size].A, dtype=theano.config.floatX),
                                                                    numpy.asarray(y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]))

        if iteration % validation_interval == 0 :
            val_iteration = iteration // validation_interval
            valid_loss[val_iteration], valid_error[val_iteration] = numpy.mean([
                    validate_model(
                        numpy.asarray(X_test[i * batch_size: (i + 1) * batch_size].A, dtype=theano.config.floatX),
                        numpy.asarray(y_test[i * batch_size: (i + 1) * batch_size], dtype="int32")
                        )
                        for i in range(n_valid_batches)
                     ],axis=0)

            print('epoch {}, minibatch {}/{}, validation error {:02.2f} %, validation loss {}'.format(
                epoch,
                minibatch_index + 1,
                n_train_batches,
                valid_error[val_iteration] * 100,
                valid_loss[val_iteration]
            ))