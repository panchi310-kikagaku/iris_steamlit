from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


# データ準備
iris = datasets.load_iris()
features = pd.DataFrame(iris['data'], columns=iris['feature_names'])
target = iris['target']

# モデル構築
model = RandomForestClassifier()
model.fit(features, target)

# モデルの保存
import pickle
pickle.dump(model, open('models/model_iris', 'wb'))
