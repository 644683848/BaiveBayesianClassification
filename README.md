# BaiveBayesianClassification

## 数据集

### 来源

 https://plg.uwaterloo.ca/~gvcormac/treccorpus06/about.html

### 说明

 为了方便处理, 已将邮件转为Excel格式, 由于github的单文件大小限制, 故将其分为两个csv

## stopwords

### 来源

 https://github.com/stopwords-iso/stopwords-zh

## 环境

安装好conda后,执行以下代码创建环境, yml文件见代码目录

``` bash
conda env create --name bayesian --file=environments.yml
```

## 运行

直接运行main.py, 即可得到sklearn.naive_bayes.MultinomialNB与手写分类器的accuracy, precision, recall和f1效果对比