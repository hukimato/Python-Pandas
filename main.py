import pandas as pd
import numpy as np
from collections import Counter
import sklearn
import random
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
"""

data = pd.read_csv('train.csv')

print("Mужчины 1 класса: ")
a = data[data['Pclass'] == 1]
a = a[a['Sex'] == 'male']
all = a.shape[0]
a = a[a['Survived'] == 1].shape[0]
print('Выжило: ' + str(a) + ' Всего: ' + str(all))
male_1_k = a/all #нужно для 6 задания

print("Женщины 1 класса: ")
a = data[data['Pclass'] == 1]
a = a[a['Sex'] == 'female']
all = a.shape[0]
a = a[a['Survived'] == 1].shape[0]
print('Выжило: ' + str(a) + ' Всего: ' + str(all))
female_1_k = a/all #нужно для 6 задания

print("Mужчины 2 класса: ")
a = data[data['Pclass'] == 2]
a = a[a['Sex'] == 'male']
all = a.shape[0]
a = a[a['Survived'] == 1].shape[0]
print('Выжило: ' + str(a) + ' Всего: ' + str(all))
male_2_k = a/all #нужно для 6 задания

print("Женщины 2 класса: ")
a = data[data['Pclass'] == 2]
a = a[a['Sex'] == 'female']
all = a.shape[0]
a = a[a['Survived'] == 1].shape[0]
print('Выжило: ' + str(a) + ' Всего: ' + str(all))
female_2_k = a/all #нужно для 6 задания

print("Mужчины 3 класса: ")
a = data[data['Pclass'] == 3]
a = a[a['Sex'] == 'male']
all = a.shape[0]
a = a[a['Survived'] == 1].shape[0]
print('Выжило: ' + str(a) + ' Всего: ' + str(all))
male_3_k = a/all #нужно для 6 задания

print("Женщины 3 класса: ")
a = data[data['Pclass'] == 3]
a = a[a['Sex'] == 'female']
all = a.shape[0]
a = a[a['Survived'] == 1].shape[0]
print('Выжило: ' + str(a) + ' Всего: ' + str(all))
female_3_k = a/all #нужно для 6 задания

#2 задание
pd.set_option("display.max_rows", None, "display.max_columns", 12)

print("Статистика для мужчин")
a = data[data['Sex'] == 'male']
print(a.describe())
print("Статистика для женщин")
a = data[data['Sex'] == 'female']
print(a.describe())

#3 Задание
a = data[data['Embarked'] == 'C']
a = a[a['Survived'] == 1].shape[0]
print("Процент выживших из порта Cherbourg: " + str(a*100/891))
C_k = a/891#нужно для 6 задания

a = data[data['Embarked'] == 'Q']
a = a[a['Survived'] == 1].shape[0]
print("Процент выживших из порта Queenstown: " + str(a*100/891))
Q_k = a/891#нужно для 6 задания

a = data[data['Embarked'] == 'S']
a = a[a['Survived'] == 1].shape[0]
print("Процент выживших из порта Southampton: " + str(a*100/891))
S_k = a/891#нужно для 6 задания

#4 Задание

name_list = data['Name'].tolist()
names = []
for element in name_list:
    r = element.find(',')
    element = element[0:r]
    names.append(element)

c = Counter(names).most_common(10)
print(c)
count = 0
for element in c:
    if count < 10:
        print(element)
    count += 1

#5 Задание. Дать среднее значение можно только возрасту и тарифу
print(data['Age'])
medians = data.median(axis=0)
print(medians)
data['Age'] = data['Age'].fillna(medians['Age'])
data['Fare'] = data['Fare'].fillna(medians['Fare'])


#6 Задание


data_inputs = data[["Pclass", "Age", "Sex", 'Embarked']]

expected_output = data[["survived"]]
data_inputs["Embarked"].replace("Q", 3, inplace = True)
data_inputs["Embarked"].replace("S", 2, inplace = True)
data_inputs["Embarked"].replace("C", 1, inplace = True)

data_inputs["sex"] = np.where(data_inputs["sex"] == "female", 0, 1)
data_inputs.head()

inputs_train, inputs_test, expected_output_train, expected_output_test = train_test_split(data_inputs, expected_output, test_size = 0.33, random_state = 42)

rf = RandomForestClassifier (n_estimators=100)
rf.fit(inputs_train, expected_output_train)

accuracy = rf.score(inputs_test, expected_output_test)
print("Accuracy = {}%".format(accuracy * 100))

joblib.dump(rf, "titanic_model1", compress=9)

rf = pd.read_csv('test.csv')
test_data = rf[["Pclass", "Age", "Sex", 'Embarked']]
test_data["Embarked"].replace("Q", 3, inplace = True)
test_data["Embarked"].replace("S", 2, inplace = True)
test_data["Embarked"].replace("C", 1, inplace = True)

test_data["sex"] = np.where(test_data["sex"] == "female", 0, 1)

pred = rf.predict(test_data)


rf = pd.read_csv('test.csv')






