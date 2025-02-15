# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Загрузка данных
data = pd.read_csv('./titanic.csv', index_col='PassengerId')

# Вывод размера таблицы
print("Размер таблицы (строки, столбцы):", data.shape)
print("\n")

# Вывод первых 5 строк данных
print("Первые 5 строк данных:")
print(data.head())
print("\n")

# Описательная статистика
print("Описательная статистика числовых данных:")
print(data.describe())
print("\n")

# Количество выживших и погибших
survived_counts = data['Survived'].value_counts()
print("Количество выживших и погибших:")
print(f"Выжило: {survived_counts[1]}")
print(f"Погибло: {survived_counts[0]}")
print("\n")

# Шансы на спасение в зависимости от класса
survival_by_class = data.groupby('Pclass')['Survived'].mean()
print("Шансы на спасение в зависимости от класса:")
print(survival_by_class)
print("\n")

# Шансы на спасение в зависимости от пола
survival_by_sex = data.groupby('Sex')['Survived'].mean()
print("Шансы на спасение в зависимости от пола:")
print(survival_by_sex)
print("\n")

# Распределение мужчин и женщин по классам
sex_class_distribution = pd.crosstab(data['Pclass'], data['Sex'])
print("Распределение мужчин и женщин по классам:")
print(sex_class_distribution)
print("\n")

# Пассажиры из Шербура с оплатой более 200 у.е.
cherbourg_high_fare = data[(data['Embarked'] == 'C') & (data['Fare'] > 200)].sort_values('Fare', ascending=False)
print("Пассажиры из Шербура, заплатившие более 200 у.е.:")
print(cherbourg_high_fare)
print("\n")

# Категоризация возраста
def age_category(age):
    if age < 30:
        return 1
    elif 30 <= age < 55:
        return 2
    else:
        return 3

data['AgeCategory'] = data['Age'].apply(age_category)
print("Добавлен категориальный признак возраста (AgeCategory).")
print("\n")

# Распределение по полу
sex_distribution = data['Sex'].value_counts()
print("Распределение пассажиров по полу:")
print(sex_distribution)
print("\n")

# Распределение Pclass и по полу
print("Распределение пассажиров по классам и полу:")
print(pd.crosstab(data['Pclass'], data['Sex']))
print("\n")

# Доля выживших по классам
survival_rate_by_class = data.groupby('Pclass')['Survived'].mean()
print("Доля выживших по классам:")
print(survival_rate_by_class)
print("\n")

# Медиана и стандартное отклонение платежей
fare_stats = data['Fare'].agg(['median', 'std'])
print("Медиана и стандартное отклонение платежей (Fare):")
print(fare_stats.round(2))
print("\n")

# Доли выживших среди молодых и пожилых
young_survival = data[data['Age'] < 30]['Survived'].mean()
old_survival = data[data['Age'] > 60]['Survived'].mean()
print("Доля выживших среди молодых (младше 30 лет):", young_survival)
print("Доля выживших среди пожилых (старше 60 лет):", old_survival)
print("\n")

# Доли выживших среди мужчин и женщин
male_survival = data[data['Sex'] == 'male']['Survived'].mean()
female_survival = data[data['Sex'] == 'female']['Survived'].mean()
print("Доля выживших среди мужчин:", male_survival)
print("Доля выживших среди женщин:", female_survival)
print("\n")

# Самое популярное имя среди мужчин
male_names = data[data['Sex'] == 'male']['Name']
most_common_male_name = male_names.mode()[0]
print("Самое популярное имя среди мужчин:", most_common_male_name)
print("\n")

# Средний возраст мужчин и женщин по классам
average_age_by_sex_class = data.groupby(['Sex', 'Pclass'])['Age'].mean()
print("Средний возраст мужчин и женщин по классам:")
print(average_age_by_sex_class)
print("\n")

# Визуализация пропусков
print("Визуализация пропущенных данных:")
msno.matrix(data)
plt.show()

# Обработка пропусков
data.drop('Cabin', axis=1, inplace=True)
data['Age'].fillna(data['Age'].median(), inplace=True)
data.dropna(inplace=True)
print("Обработаны пропуски: удален столбец Cabin, пропуски в Age заполнены медианой.")
print("\n")

# Столбчатые диаграммы для категориальных переменных
print("Столбчатая диаграмма распределения по классам (Pclass):")
sns.countplot(x='Pclass', data=data)
plt.show()

# Попарные зависимости признаков
print("Попарные зависимости признаков:")
sns.pairplot(data[['Age', 'Fare', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Survived']])
plt.show()

# Boxplot зависимости Fare от Pclass
print("Boxplot зависимости Fare от Pclass:")
sns.boxplot(x='Pclass', y='Fare', data=data)
plt.show()

# Соотношение погибших и выживших по полу
print("Соотношение погибших и выживших по полу:")
sns.countplot(x='Sex', hue='Survived', data=data)
plt.show()

# Соотношение погибших и выживших по классу каюты
print("Соотношение погибших и выживших по классу каюты:")
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.show()

# Зависимость выживания от возраста
print("Зависимость выживания от возраста:")
sns.boxplot(x='Survived', y='Age', data=data)
plt.show()

# Корреляционная матрица
print("Корреляционная матрица:")
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()