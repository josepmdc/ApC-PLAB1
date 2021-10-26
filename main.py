from math import cos
from numpy.lib.function_base import gradient
from pandas.io.formats import style
from scipy.sparse import data
import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

pd.set_option('display.float_format', lambda x: '%.3f' % x)

dataset = pd.read_csv('dataset.csv', decimal=',').dropna()

dataset.head()
dataset.tail()

dataset.columns = ['date', 'avg_temp', 'min_temp', 'max_temp', 'precipitation', 'weekend', 'consumption_liters']
dataset['consumption_liters'] = pd.to_numeric(dataset['consumption_liters'])

dataset.describe()


# plt.figure(figsize = (10, 7))
# ax = sns.heatmap(dataset.corr(), annot=True, linewidths=.5)


weekends_liters = sum(dataset[dataset.weekend == 1]['consumption_liters'])
weekdays_liters = sum(dataset[dataset.weekend == 0]['consumption_liters'])

# fig = plt.figure()
# labels = ["Caps de setmana", "Dies laborals"]
# ax = fig.add_axes([0,0,1,1])
# ax.bar(labels, [weekends_liters, weekdays_liters], color=["r", "b"])
# plt.title("Diferència en el consum de cervesa en dies laborals i cap de setmana", figure=fig)
# plt.ylabel("Litres", figure=fig)
# plt.show()


weekends_liters = len(dataset[dataset.weekend == 1])
weekdays_liters = len(dataset[dataset.weekend == 0])

# fig = plt.figure()
# labels = ["Caps de setmana", "Dies laborals"]
# ax = fig.add_axes([0,0,1,1])
# ax.bar(labels, [weekends_liters, weekdays_liters], color=["r", "b"])
# plt.title("Diferència en la quantitat de dies laborals i de cap de setmana", figure=fig)
# plt.ylabel("Dies", figure=fig)
# plt.show()


dataset['date'] = pd.to_datetime(dataset['date'])
dataset['month'] = dataset['date'].apply(lambda x: x.strftime('%B'))
dataset['day'] = dataset['date'].apply(lambda x: x.strftime('%A'))
dataset.head()


# plt.figure(figsize=(15, 5))
# plt.title("Beer consumption by months")
# ax = sns.boxplot(data=dataset, x = "day", y = "consumption_liters")
# plt.show()

# plt.figure(figsize=(15, 5))
# plt.title("Beer consumption by months")
# ax = sns.boxplot(data=dataset, x = "month", y = "consumption_liters")
# plt.show()


###########################################################################################################
###########################################################################################################
######################                                                          ###########################
######################    !!  A partir d'aquí no està al notebook encara !!     ###########################
######################                                                          ###########################
###########################################################################################################
###########################################################################################################


def apply_linear_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model

# We encode the month and day since they are not numerical attributes
label_encoder = LabelEncoder()
dataset['month'] = label_encoder.fit_transform(dataset['month'])
dataset['day'] = label_encoder.fit_transform(dataset['day'])


def plot_model(predictions, y_test):
    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Comparació entre els valors predits i els valors reals")
    ax.set_xlabel("Valor Reals")
    ax.set_ylabel("Valor Predits")
    ax.scatter(predictions, y_test)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", color="r")
    plt.show()


X, y = dataset.drop(['consumption_liters', 'date'], axis=1), dataset['consumption_liters']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# for attr in X_train.columns:
#     # Train the model
#     # we need to reshape the arrays into a 2d arrays for the linear regression
#     model = apply_linear_regression(
#         X_train[attr].values.reshape(-1, 1),
#         y_train.values.reshape(-1, 1)
#     )

#     # Test the model
#     predictions = model.predict(
#         X_test[attr].values.reshape(-1, 1)
#     )

#     plot_model(predictions, y_test)

#     print(f"Mean squeared error: {mean_squared_error(y_test, predictions)}")
#     print(f"R2 score: {r2_score(y_test, predictions)}\n")

# model = apply_linear_regression(X_train, y_train)

# # Test the model
# predictions = model.predict(X_test)

# plot_model(predictions, y_test)
# print(f"Mean squeared error: {mean_squared_error(y_test, predictions)}")
# print(f"R2 score: {r2_score(y_test, predictions)}\n")

# pca = PCA(n_components=2)
# pca_train = pca.fit_transform(X_train)
# pca_test = pca.fit_transform(X_test)

# model = apply_linear_regression(pca_train, y_train)
# predictions = model.predict(pca_test)

# plot_model(predictions, y_test)
# print(f"Mean squeared error: {mean_squared_error(y_test, predictions)}")
# print(f"R2 score: {r2_score(y_test, predictions)}\n")


################################################################################
################################################################################
######################                               ###########################
######################    !!  Gradient descent !!    ###########################
######################                               ###########################
################################################################################
################################################################################

class Regressor(object):
    def __init__(self, X, y, alpha=0.01):
        self.theta = np.zeros(X.shape[1])
        self.alpha = alpha
        self.X = X
        self.y = y
        
    def predict(self, x):
        return x.dot(self.theta)
    
    def train(self, epoch=1000):
        m = self.y.shape[0]
        cost_history = []
        
        for i in range(epoch):
            hypothesis = self.X.dot(self.theta)
            loss = hypothesis - self.y

            cost =  np.sum(loss**2) / 2 * m  # mean_squared_error(self.y, hypothesis) 
            cost_history.append(cost)

            gradient = self.X.T.dot(loss) / m
            self.theta = self.theta - self.alpha * gradient

        return self.theta, cost_history


std_X_train = StandardScaler().fit_transform(X_train)

reg = Regressor(std_X_train, y_train)
theta, cost_history = reg.train()
predictions = reg.predict(X_test)

fig, ax1 = plt.subplots()
ax1.plot(cost_history, label='Funció de perdua', color='r')
ax1.set_xlabel('Iteracions')
ax1.set_ylabel('Cost (MSE)')
plt.show()