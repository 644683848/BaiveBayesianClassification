import csv
import pickle

import numpy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def load_model(name):
    with open(name + '.pickle', 'rb') as infile:
        model = pickle.load(infile)
        return model


def train(train_features, train_labels):
    total_mails = train_features.shape[0]
    total_features = train_features.shape[1]
    spam_num = sum(train_labels)
    p_spam = spam_num / total_mails
    w_i_given_c1_cnt = numpy.ones(total_features)
    w_i_given_c0_cnt = numpy.ones(total_features)
    c1_words_cnt = 2
    c0_words_cnt = 2
    w_i_cnt = numpy.zeros(total_features)
    for i in range(total_mails):
        if train_labels[i] == 1:
            w_i_given_c1_cnt += train_features[i]
            c1_words_cnt += np.sum(train_features[0])
        else:
            w_i_given_c0_cnt += train_features[i]
            c0_words_cnt += np.sum(train_features[0])
        w_i_cnt += train_features[i]
    p_w_i_given_c1 = w_i_given_c1_cnt / c1_words_cnt
    p_w_i_given_c0 = w_i_given_c0_cnt / c0_words_cnt
    return p_spam, p_w_i_given_c1, p_w_i_given_c0


def test(bow_train_features, train_labels, bow_test_features, test_labels):
    """
    1. 从训练数据中获取:P(c1), P(w_i|c1);
    2. 根据待判别向量, 计算P(w_i);
    3. P(w_i) = P(w_1) * P(w_2) * ... * P(w_N)
       log(P(w_i)) = log(P(w_1)) + log(P(w_2)) + ... + log(P(w_N)): log_p_w_i = sum(log_w_i)
   4. log(P(w_i,c1)) = log(P(w_1|c1)) + log(P(w_2|c1)) + ... + log(P(w_N|c1)) + log(P(c1)): log_p_w_i_given_c1 = sum(log_p_w_i_given_c1)
   5. log(P(c1|w_i)) = log(p(w_i, c1)) - log(p(w_i))

   6. test_feature的
        1. log(p(w_i)): sum(log(test_feature * p(w_i)))
        2. log(p(w_i, c1))): sum(log(test_feature * p(w_i|c1) * p(c1)))
        3. log(p(c1|w_i): log(p(w_i, c1)) - log(p(w_i))
    """
    # 加载训练数据
    # bow_train_features = load_model('bow_train_features')
    # train_labels = load_model('train_labels')
    #
    # bow_test_features = load_model('bow_test_features')
    p_spam, p_w_i_given_c1, p_w_i_given_c0 = train(bow_train_features, train_labels)
    # 计算log(p(w_i, c1))和log(p(w_i, c0)), 比较大小
    # log(p(w_i, c1))
    log_p_w_i_given_c1 = bow_test_features.A * np.log(p_w_i_given_c1).T + np.log(p_spam)
    log_p_w_i_given_c0 = bow_test_features.A * np.log(p_w_i_given_c0).T + np.log(1 - p_spam)
    predictions = np.asarray((log_p_w_i_given_c1 > log_p_w_i_given_c0).astype(int))
    accuracy = accuracy_score(test_labels, predictions)
    # Specify the zero_division parameter
    precision = precision_score(test_labels, predictions, average='binary', zero_division=0)
    recall = recall_score(test_labels, predictions, average='binary', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='binary', zero_division=0)
    return accuracy, precision, recall, f1


def test_train_function():
    # Test 1: Basic functionality
    train_features = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
    train_labels = np.array([1, 0, 1, 0])
    print(train(train_features, train_labels))


