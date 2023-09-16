import json
import pickle
import re

import jieba
import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from test import test

size = 5000


# 数据集来源: https://plg.uwaterloo.ca/~gvcormac/treccorpus06/about.html
def load_email(paths):
    """
    Load email data from two CSV files and return a corpus of emails and their spam labels.

    Args:
    - paths (tuple): A tuple containing paths to the two CSV files.
    - size (int, optional): The number of emails to load. If None, all emails are loaded.

    Returns:
    - tuple: A tuple containing the email corpus and their labels.
    """

    dfs = []  # list to hold dataframes from both CSVs

    for path in paths:
        maildf = pd.read_csv(path, header=None,
                             names=['Sender', 'Receiver', 'CarbonCopy', 'Subject', 'Date', 'Body', 'isSpam'])
        dfs.append(maildf)

    # Concatenate data from both CSV files
    combined_df = pd.concat(dfs, axis=0).reset_index(drop=True)

    # Filter rows where 'Body' column is not null
    filteredmaildf = combined_df[combined_df['Body'].notnull()]

    # Prepare corpus and labels
    corpus = filteredmaildf['Body']
    labels = filteredmaildf['isSpam']

    # If size is specified, slice the corpus and labels
    if size is not None:
        corpus = list(corpus)[:size]
        labels = list(labels)[:size]

    return corpus, labels


# 来源: https://github.com/stopwords-iso/stopwords-zh
def load_stopwords(path):
    with open(path, 'r', encoding='utf-8') as file:
        stopwords = json.load(file)
    return set(stopwords)


STOPWORDS = load_stopwords('stopwords-zh.json')


def textParse(text):
    listOfTokens = jieba.lcut(text)
    newList = [re.sub(r'\W+', '', s) for s in listOfTokens]  # Updated regex pattern
    filtered_text = [tok for tok in newList if len(tok) > 0]
    return filtered_text


def remove_stopwords(tokens):
    filtered_tokens = [token for token in tokens if token not in STOPWORDS]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus, tokenize=False):
    normalized_corpus = []
    for text in corpus:
        filtered_text = textParse(text)
        filtered_text = remove_stopwords(filtered_text)

        normalized_corpus.append(filtered_text)

    return normalized_corpus


def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorized = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorized.fit_transform(corpus)
    return vectorized, features


def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix


def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorized = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorized.fit_transform(corpus)
    return vectorized, features


def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    # Train the model
    classifier.fit(train_features, train_labels)

    # Predict on the test set
    predictions = classifier.predict(test_features)

    # Evaluate the predictions
    accuracy = accuracy_score(test_labels, predictions)

    # Specify the zero_division parameter
    precision = precision_score(test_labels, predictions, average='binary', zero_division=0)
    recall = recall_score(test_labels, predictions, average='binary', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='binary', zero_division=0)

    # Return the evaluation metrics
    return accuracy, precision, recall, f1


def dump_model(model, name):
    with open(name + '.pickle', 'wb') as outfile:
        # "wb" argument opens the file in binary mode
        pickle.dump(model, outfile)


def load_model(name):
    with open(name + '.pickle', 'rb') as infile:
        model = pickle.load(infile)
        return model


# print("加载邮件数据")
corpus, labels = load_email(('mail1.csv', 'mail2.csv'))
# print("归一化训练集")
normalized_corpus = normalize_corpus(corpus)
# print("向量化训练集")
bow_vectorized, bow_features = bow_extractor(normalized_corpus)
# print("划分数据集")
bow_train_features, bow_test_features, train_labels, test_labels = train_test_split(bow_features, labels, test_size=0.3,
                                                                                    random_state=0)

# dump_model(bow_train_features, 'bow_train_features')
# dump_model(train_labels, 'train_labels')
# dump_model(bow_test_features, 'bow_test_features')
# dump_model(test_labels, 'test_labels')
print("加载模型")
mnb = MultinomialNB()
svm = SGDClassifier(loss='hinge', n_iter_no_change=100)
lr = LogisticRegression()
# 基于词袋模型的多项朴素贝叶斯
print("基于词袋模型特征的贝叶斯分类器")
mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
                                                   train_features=bow_train_features,
                                                   train_labels=train_labels,
                                                   test_features=bow_test_features,
                                                   test_labels=test_labels)
print(mnb_bow_predictions)
print("自己写的基于词袋模型特征的贝叶斯分类器")
print(test(bow_train_features, train_labels, bow_test_features, test_labels))

# 基于tfidf的多项式朴素贝叶斯模型
# print("基于tfidf的贝叶斯模型")
# tfidf 特征
# tfidf_vectorized, tfidf_train_features = tfidf_extractor(norm_train_corpus)
# tfidf_test_features = tfidf_vectorized.transform(norm_test_corpus)
# mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
#                                                      train_features=tfidf_train_features,
#                                                      train_labels=train_labels,
#                                                      test_features=tfidf_test_features,
#                                                      test_labels=test_labels)
# print(mnb_tfidf_predictions)
