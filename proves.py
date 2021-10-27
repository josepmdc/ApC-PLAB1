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
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
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


def plot_model(y_test, predictions):
    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Comparació entre els valors predits i els valors reals")
    ax.set_xlabel("Valor Reals")
    ax.set_ylabel("Valor Predits")
    ax.scatter(y_test, predictions)
    # ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", color="r")
    ax.plot([min(y_test), max(y_test)], [min(predictions), max(predictions)], color='red', ls="--")
    plt.show()


X, y = dataset.drop(['consumption_liters', 'date'], axis=1), dataset['consumption_liters']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# for attr in X_train.columns:
# #     # Train the model
# #     # we need to reshape the arrays into a 2d arrays for the linear regression
#     model = apply_linear_regression(
#         X_train[attr].values.reshape(-1, 1),
#         y_train.values.reshape(-1, 1)
#     )

# #     # Test the model
#     predictions = model.predict(
#         X_test[attr].values.reshape(-1, 1)
#     )

#     plot_model(predictions, y_test)

#     print(f"Mean squeared error: {mean_squared_error(y_test, predictions)}")
#     print(f"R2 score: {r2_score(y_test, predictions)}\n")


X, y = dataset.drop(['consumption_liters', 'date'], axis=1), dataset['consumption_liters']
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = apply_linear_regression(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

plot_model(y_test, predictions)
print(f"Mean squeared error: {mean_squared_error(y_test, predictions)}")
print(f"R2 score: {r2_score(y_test, predictions)}\n")



pca = PCA(n_components="mle")
pca_train = pca.fit_transform(X_train)
pca_test = pca.fit_transform(X_test)

model = apply_linear_regression(pca_train, y_train)
predictions = model.predict(pca_test)

plot_model(y_test, predictions)
print(f"Mean squeared error: {mean_squared_error(y_test, predictions)}")
print(f"R2 score: {r2_score(y_test, predictions)}\n")


"""
**1. Quin són els atributs més importants per fer una bona predicció?**
Els atributs que seran mes importants per fer una prediccio decent seran determinats per el seu MSE, buscant el que el tingui de menor valor.

**2. Amb quin atribut s'assoleix un MSE menor?**
En el nostre cas els atributs que ens donen menys error son la temperatura mitjana i la temperatura maxima.

**3. Quina correlació hi ha entre els atributs de la vostra base de dades?**
Els atributs de temperatura estan correlacionats ja que si la temperatura es major la gent veu mes cervesa ja sigui per refrescarse o 
perque es quan tenen vacances per tant mes temps lliure i s'ho poden permetre. Els atributs de data estan correlacionats ja que la gent 
veu mes en el cap de setmana perque no ha de treballar al dia seguent i els mesos de estiu on fa mes calor i tenen vacances tenen el mateix motiu.
En quan al atribut de precipitacio no te massa correlacio.

**4. Com influeix la normalització en la regressió?**
Nosaltres no hem vist cap millora, posiblement sigui perque la majoria de les dades ja estan normalitzades com pot ser la temperatura 
que esta a una escala similar a els llitres de cervesa.

**5. Com millora la regressió quan es filtren aquells atributs de les mostres que no contenen informació?**
Ja els haviem filtrat al principi per tant no hen notat cap millora. 
Pero assumim que si tens moltes dades nules la prediccio sera menys eficaç ja que estara considerant dades buides com informacio per a fer el model.

**6. Si s'aplica un PCA, a quants components es redueix l'espai? Per què?**
Es redueix a 2 components. Perque hem considerat que amb 2 component podem fer una prediccio suficientment bona. Amb la temperatura i el dia 
ja podem fer una prediccio bona ja que sabent el dia ja es te en compte si es cap de setmana i la temperatura mitjana per saber si es un dia caluros o no.
"""





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
        theta_history = []
        
        for i in range(epoch):
            hypothesis = self.X.dot(self.theta)
            loss = hypothesis - self.y

            cost =  np.sum(loss**2) / (2 * m)  # mean_squared_error(self.y, hypothesis) 
            cost_history.append(cost)

            gradient = self.X.T.dot(loss) / m
            self.theta = self.theta - self.alpha * gradient
            theta_history.append(self.theta)

        return self.theta, cost_history, theta_history


std_X_train = StandardScaler().fit_transform(X_train)
std_X_test = StandardScaler().fit_transform(X_test)

reg = Regressor(std_X_train, y_train, alpha=0.0005)
theta, cost_history, theta_history = reg.train(100000)
predictions = reg.predict(std_X_test)

fig, ax1 = plt.subplots()
ax1.plot(cost_history, label='Cost history', color='r')
ax1.set_xlabel('Iteracions')
ax1.set_ylabel('Cost (MSE)')
# ax2 = ax1.twinx()
# ax2.plot(theta_history, label=X.columns, color='b')
# ax2.set_ylabel('Theta values')
plt.show()

plot_model(y_test, predictions)
print(f"Mean squeared error: { mean_squared_error(y_test, predictions) }")
print(f"R2 score: { r2_score(y_test, predictions) }\n")
