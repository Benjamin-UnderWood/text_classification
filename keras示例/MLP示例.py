# coding:utf-8
"""
多层感知器MLP（前馈神经网络/全连接神经网络）神经网络情感分类示例
"""

# 数据预处理
import os
import re
# 导入分词类
from keras.preprocessing.text import Tokenizer
# 导入数字列表类
from keras.preprocessing import sequence
# 导入线性堆叠模型类
from keras.models import Sequential
# 导入嵌入层类
from keras.layers.embeddings import Embedding
# 导入Dropout层类
from keras.layers.core import Dropout
# 导入平坦层类
from keras.layers.core import Flatten
# 导入隐藏层类
from keras.layers.core import Dense
# 导入绘制模型类
from keras.utils import plot_model



# 移除特殊字符
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


# 读取文件函数
def read_files(filetype):
    path = "../data/aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print('read', filetype, 'files:', len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)

    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_labels, all_texts


def main():
    # 读取训练文本集
    y_train, train_text = read_files("train")
    # 读取测试文本集
    y_test, test_text = read_files("test")
    # 构建分词器，由该分词器构建词典的词语数量最多为2000
    token = Tokenizer(num_words=2000)
    # 由训练文本构建词典
    token.fit_on_texts(train_text)
    # 将训练文本集转换成数字列表
    x_train_seq = token.texts_to_sequences(train_text)
    # 将测试文本集转换成数字列表
    x_test_seq = token.texts_to_sequences(test_text)
    # 截长补短让所有“数字列表”的长度都为100
    x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
    x_test = sequence.pad_sequences(x_test_seq, maxlen=100)

    # 构建神经网络模型
    model = Sequential()
    # 加入嵌入层
    model.add(Embedding(output_dim=32,  # 嵌入层输出维度为32
                        input_dim=2000,  # 嵌入层输入维度为2000维的词典
                        input_length=100))  # 数字列表每一项有100个数字
    # 加入Dropout层以避免过拟合
    model.add(Dropout(0.2))
    # 加入平坦层
    model.add(Flatten())
    # 加入隐藏层
    model.add(Dense(units=256,   # 神经元256个
                    activation='relu'))  # 激活函数'relu'
    # 加入Dropout层以避免过拟合
    model.add(Dropout(0.25))  # 放弃25%的神经元以避免过拟合
    # 输出层
    model.add(Dense(units=1,   # 1个神经元
                    activation='sigmoid'))  # 激活函数为sigmoid
    # 输出模型摘要
    model.summary()
    # 绘制模型
    plot_model(model, to_file='MLP_Model.png')

    # 使用compile方法对训练模型进行设置
    model.compile(loss='binary_crossentropy',  # 损失函数
                  optimizer='adam',  # 优化方法
                  metrics=['accuracy'])
    # 训练
    train_history = model.fit(x_train, y_train,
                              batch_size=100,   # 每一批次100项数据
                              epochs=10,   # 执行10个训练周期
                              verbose=2,   # 显示训练过程
                              validation_split=0.2)  # 20%验证数据
    # 评测
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("precision=", scores[1])


if __name__ == "__main__":
    main()
