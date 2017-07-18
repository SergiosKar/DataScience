import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


'''retrieve data from SQL'''
conn = sqlite3.connect('C:/SQLite/GUI.Titanic.db')
cursor = conn.cursor()
cursor.execute(
    "select Survived,PClass,Sex,Age,SibSp,Parch,Fare from Passenger "
)
m_data=cursor.fetchall()
pdata = pd.DataFrame(m_data)
pdata.columns = [i[0] for i in  cursor.description]
conn.close()

conn = sqlite3.connect('C:/SQLite/GUI.Titanic.db')
cursor = conn.cursor()
cursor.execute(
    "select PClass,Sex,Age,SibSp,Fare,Parch from PassengerTest "
)
m_datatest=cursor.fetchall()
testdata=pd.DataFrame(m_datatest)
testdata.columns = [i[0] for i in  cursor.description]
conn.close()




'''
pdata = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')
'''
combined=[pdata ,testdata]


'''data wrangling'''
#convert to int
for dataset in combined:
    cols = [ 'Pclass', 'Age','SibSp','Parch','Fare']
    dataset[cols] = dataset[cols].apply(pd.to_numeric, errors='coerce', axis=1)
pdata['Survived'] = pdata['Survived'].apply(pd.to_numeric, errors='coerce')


print (pdata.describe().to_string())

class_sex_grouping=pdata.groupby(['Sex','Pclass']).mean()
#class_sex_grouping['Survived'].plot.bar()

group_by_age = pd.cut(pdata["Age"], np.arange(0, 90, 10))
age_grouping = pdata.groupby(group_by_age).mean()#.count() to see the number of survived .mean() to see number surb/total number
#age_grouping['Survived'].plot.bar()


#convert categorical to string by mappning
for dataset in combined:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    #replace NA with mean
    dataset['Age'].fillna((dataset['Age'].mean()), inplace=True)

#create new feature
for dataset in combined:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

family_grouping=pdata[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
#family_grouping['Survived'].plot.bar()

for dataset in combined:
    # there's one missing fare value - replacing it with the mean.
    dataset['Fare'].fillna((dataset['Fare'].mean()), inplace=True)
    dataset['FareBand'] = pd.qcut(dataset['Fare'], 4)
    #pdata[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean()

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

#remove feature
pdata=pdata.drop(['Parch','FareBand','SibSp'], axis=1)
testdata=testdata.drop(['Parch','FareBand','SibSp'], axis=1)
combine = [pdata, testdata]

print(pdata.head().to_string())
print(testdata.head().to_string())
#plt.show()

'''models and prediction'''

X_train = pdata.drop("Survived", axis=1)
Y_train = pdata["Survived"]
X_test  = testdata
#print(X_train.shape, Y_train.shape, X_test.shape)


#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print('Losistic regression', acc_log)

coeff_df = pd.DataFrame(pdata.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False))

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print('SVM',acc_svc)

#K Neerest Neighbours
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print('KNN',acc_knn)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print('GNB',acc_gaussian)


# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print('Linear SVC',acc_linear_svc)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print('Perceptron',acc_perceptron)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print('SGD',acc_sgd)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print('Dec Tree',acc_decision_tree)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('Random Forest',acc_random_forest)

#Gradient Boosting Classifier
gradient_boosting = GradientBoostingClassifier(n_estimators=100)
gradient_boosting.fit(X_train, Y_train)
Y_pred = gradient_boosting.predict(X_test)
acc_grad_boost = round(gradient_boosting.score(X_train, Y_train) * 100, 2)
print('Gradient Boosting Classifier',acc_grad_boost)

#Neural Network //better to implement with tensorflow cause its slow
'''
mlp = MLPClassifier(
    activation='tanh',
    solver='lbfgs',
    hidden_layer_sizes=(80),
    max_iter=20000,
    learning_rate_init=1e-5,

)
mlp.fit(X_train,Y_train)
predicted = mlp.predict(X_test)
acc_mlp = round(mlp.score(X_train, Y_train) * 100, 2)
print('Neural Network',acc_mlp)
'''


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree','Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree,acc_grad_boost]})
print(models.sort_values(by='Score', ascending=False))