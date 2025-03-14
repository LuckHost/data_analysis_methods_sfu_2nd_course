import pandas as pd

# Загрузка данных
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Просмотр первых строк тренировочных данных
print(train_data.head())

# Просмотр первых строк тестовых данных
print(test_data.head())

print("isnull train:", train_data.isnull().sum())
print("isnull test:", test_data.isnull().sum())

train_data = pd.get_dummies(train_data)


from sklearn.impute import SimpleImputer

# Заполнение пропусков для числовых колонок
numeric_columns = ['ClientPeriod', 'MonthlySpending', 'TotalSpent']
numeric_imputer = SimpleImputer(strategy='mean')
train_data[numeric_columns] = numeric_imputer.fit_transform(train_data[numeric_columns])
test_data[numeric_columns] = numeric_imputer.transform(test_data[numeric_columns])

# Заполнение пропусков для категориальных колонок
categorical_columns = ['Sex', 'HasPartner', 'HasChild', 'HasPhoneService', 'HasMultiplePhoneNumbers', 
                       'HasInternetService', 'HasOnlineSecurityService', 'HasOnlineBackup', 
                       'HasDeviceProtection', 'HasTechSupportAccess', 'HasOnlineTV', 
                       'HasMovieSubscription', 'HasContractPhone', 'IsBillingPaperless', 'PaymentMethod']
categorical_imputer = SimpleImputer(strategy='most_frequent')
train_data[categorical_columns] = categorical_imputer.fit_transform(train_data[categorical_columns])
test_data[categorical_columns] = categorical_imputer.transform(test_data[categorical_columns])

from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column])
    test_data[column] = le.transform(test_data[column])
    label_encoders[column] = le
    
    X_train = train_data.drop(['id', 'Churn'], axis=1)
y_train = train_data['Churn']
X_test = test_data.drop('id', axis=1)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
log_reg = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

print("Лучшие параметры для логистической регрессии:", grid_search.best_params_)

from sklearn.neighbors import KNeighborsClassifier

param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid, cv=5, scoring='f1')
grid_search_knn.fit(X_train, y_train)

print("Лучшие параметры для KNN:", grid_search_knn.best_params_)

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
importances = rf.feature_importances_
print("Важность признаков:", importances)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Используем лучшую модель (например, логистическую регрессию)
best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)

# Сохранение результатов
output = pd.DataFrame({'id': test_data['id'], 'Churn': y_pred_test})
output.to_csv('predictions.csv', index=False)