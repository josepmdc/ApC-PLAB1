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
from sklearn.preprocessing import LabelEncoder

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


def regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model

# We encode the month and day since they are not numerical attributes
label_encoder = LabelEncoder()
dataset['month'] = label_encoder.fit_transform(dataset['month'])
dataset['day'] = label_encoder.fit_transform(dataset['day'])

print(set(dataset['month']))

X, y = dataset.drop(['consumption_liters', 'date'], axis=1), dataset['consumption_liters']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dataset
for attr in X_train.columns:
    # we need to reshape the array into a 2d array for the linear regression
    reshaped_X_train = X_train[attr].values.reshape(-1, 1)
    reshaped_X_test = X_test[attr].values.reshape(-1, 1)
    reshaped_y_train = y_train.values.reshape(-1, 1)
    model = regression(reshaped_X_train, reshaped_y_train)
    predictions = model.predict(reshaped_X_test)

    print(attr)
    print(f"Mean squeared error: {mean_squared_error(y_test, predictions)}")
    print(f"R2 score: {r2_score(y_test, predictions)}")
    print()

    f, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"{attr} <-> Consumption Litres")
    ax.set_xlabel("Valor Reals")
    ax.set_ylabel("Valor Predits")
    ax.scatter(predictions, y_test)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", color="r")
    plt.show()

