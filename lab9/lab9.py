# 导入必要的库
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import fasttext
import fasttext.util

# 下载20 Newsgroups数据集
newsgroups = fetch_20newsgroups(subset='all')
X, y = newsgroups.data, newsgroups.target

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 方法1：TfidfVectorizer + Logistic Regression
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 训练逻辑回归模型
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# 预测和评估
y_pred_tfidf = lr_model.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
print(f'TfidfVectorizer + Logistic Regression Accuracy: {accuracy_tfidf:.4f}')

# 方法2：fastText
# 保存训练和测试数据到文件
train_data = 'train.txt'
test_data = 'test.txt'

with open(train_data, 'w') as f:
    for text, label in zip(X_train, y_train):
        f.write('__label__{} {}\n'.format(label, text.replace("\n", " ")))

with open(test_data, 'w') as f:
    for text, label in zip(X_test, y_test):
        f.write('__label__{} {}\n'.format(label, text.replace("\n", " ")))

# 训练fastText模型
# fasttext.util.download_model('en', if_exists='ignore')  # 下载预训练的英语词向量
model = fasttext.train_supervised(input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2)

# 评估fastText模型
def fasttext_evaluate(model, test_data):
    correct = 0
    total = 0
    with open(test_data, 'r') as f:
        for line in f:
            label, text = line.split(' ', 1)
            label = label.replace('__label__', '').strip()
            pred_label = model.predict(text.strip())[0][0].replace('__label__', '')
            if label == pred_label:
                correct += 1
            total += 1
    return correct / total

accuracy_fasttext = fasttext_evaluate(model, test_data)
print(f'fastText Accuracy: {accuracy_fasttext:.4f}')
