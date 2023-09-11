import csv

import numpy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorized = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorized.fit_transform(corpus)
    return vectorized, features


# split_csv_file('mail.csv', 'mail1.csv', 'mail2.csv')
def split_csv_file(filename, output1, output2):
    # Read the original CSV file
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Calculate the split index
    split_index = len(rows) // 2

    # Write the first set of rows to output1
    with open(output1, 'w', newline='', encoding='utf-8') as file1:
        writer = csv.writer(file1)
        for row in rows[:split_index]:
            writer.writerow(row)

    # Write the second set of rows to output2
    with open(output2, 'w', newline='', encoding='utf-8') as file2:
        writer = csv.writer(file2)
        for row in rows[split_index:]:
            writer.writerow(row)


def print_top_n_indices(arr, n, bow_vectorized):
    top_N_indices = np.asarray(arr)[0].argsort()[-n:][::-1]
    for key, value in bow_vectorized.vocabulary_.items():
        if value in top_N_indices:
            print(key)


def test2(bow_vectorized: CountVectorizer, train_features, train_labels):
    """
    输入1: 训练样本矩阵
    输入2: 训练样本标签
    输出: 含有N个特征w_i时, 是垃圾邮件还是普通邮件

    目标: 计算在当前邮件所代表的特征w_i(12335维度中的a个维度中的特征)下, 此邮件为垃圾邮件的概率---P(c1|W_i), 此邮件为正常邮件的概率---P(c0|W_i)
    1. 是垃圾邮件, 且有特征w_i的概率P(w_i, c1)
      1. 计算P(c1), 即垃圾邮件的概率, 变量名为p_spam
      2. 计算P(w_i|c1), 即垃圾邮件这个小样本中, 含有特征w_i的概率: w_i_given_c1_p
    2. 计算含有特征W_i的小样本概率P(W_i): w_i_p
      P(c1|W_i) = P(W_i, c1) / P(W_i) = P(W_i|c1) * P(c1) / P(W_i) = p(w_0|c1) * p(w_1|c1) * ... * p(w_43359|c1) / p(w_i)
    """
    total_mails = train_features.shape[0]
    total_features = train_features.shape[1]
    spam_num = sum(train_labels)
    p_spam = spam_num / total_mails
    w_i_given_c1_cnt = numpy.zeros(total_features)
    w_i_given_c0_cnt = numpy.zeros(total_features)
    w_i_cnt = numpy.zeros(total_features)
    for i in range(total_mails):
        if train_labels[i] == 1:
            w_i_given_c1_cnt += train_features[i]
        else:
            w_i_given_c0_cnt += train_features[i]
        w_i_cnt += train_features[i]
    print('最容易被判定为垃圾邮件的30个特征')
    print_top_n_indices(w_i_given_c0_cnt, 30, bow_vectorized)
    print('最容易被判定为非垃圾邮件的30个特征')
    print_top_n_indices(w_i_given_c1_cnt, 30, bow_vectorized)
    p_w_i_given_c1 = w_i_given_c1_cnt / spam_num
    p_w_i_given_c0 = w_i_given_c0_cnt / total_features
    p_w_i = w_i_cnt / total_mails
    print(p_w_i, p_w_i_given_c1, p_w_i_given_c0)



