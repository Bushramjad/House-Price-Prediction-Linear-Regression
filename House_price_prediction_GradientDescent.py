import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


# Helping Functions
def read_data(path):
    return pd.read_csv(path, index_col=0)


# Train dataset
dataframe = read_data('train.csv')
x_train = dataframe[dataframe.columns.drop('SalePrice')]
y_train = dataframe['SalePrice']
x_train.insert(0, 'x0', np.ones(len(x_train)))

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
y_train = y_train.to_numpy()
y_train = y_train[:, np.newaxis]

# Label Encoding
dict1 = defaultdict(LabelEncoder)
x_train = x_train.astype(str).apply(lambda x_train: dict1[x_train.name].fit_transform(x_train))
x_test = x_test.astype(str).apply(lambda x_test: dict1[x_test.name].fit_transform(x_test))

# Model Prediction
lg = LinearRegression()
lg.fit(x_train, y_train)
y_pred = lg.predict(x_test)

# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
# R2 Score
print('R2 Score: %.2f' % r2_score(y_test, y_pred))

# Visualization
plt.plot(np.arange(100), y_test[0:100], 'g', label="Actual Result ")
plt.plot(np.arange(100), y_pred[0:100], 'r', label="Predicted")
plt.legend(loc=2)
plt.show()
