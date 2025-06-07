import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


train = pd.read_csv('Training_BOP.csv', low_memory=False)
test = pd.read_csv('Testing_BOP.csv', low_memory=False)

# Data info
print("Train Data Info:")
print(train.info())
print("Test Data Info:")
print(test.info())

# First 5 rows
print("First 5 rows of Train Data:")
print(train.head())

# Last 5 rows
print("Last 5 rows of Train Data:")
print(train.tail())

# float64 to float32
def reduce_memory(df):
  for col in df.select_dtypes(include=['float64']):
      df[col] = df[col].astype('float32')
  for col in df.select_dtypes(include=['int64']):
      df[col] = df[col].astype('int32')
  return df

train = reduce_memory(train)
test = reduce_memory(test)

# Null values (For Train Data)
train.isnull().sum()

# Percentage of null values (For Train Data)
train.isnull().sum()/train.shape[0]*100

# Null values (For Test Data)
test.isnull().sum()

# Percentage of null values (For Test Data)
test.isnull().sum()/test.shape[0]*100

# Checking duplicate values (For Test Data)
print(train.duplicated().sum())
# Checking duplicate values (For Test Data)
print(test.duplicated().sum())

numeric_cols = train.select_dtypes(include=['number']).columns
categorical_cols = train.select_dtypes(include=['object']).columns

# Checking garbage values (For Train Data)
for i in categorical_cols:
  print(train[i].value_counts())
  print("****"*10)

# Checking garbage values (For Test Data)
for i in categorical_cols:
  print(test[i].value_counts())
  print("****"*10)

# Describe Data
print("=== Train Data ===")
print(train.describe().T)

print("=== Test Data ===")
print(test.describe().T)

# Describe data
print("=== Train Data ===")
train.describe(include="object")

print("=== Test Data ===")
test.describe(include="object")

train = train.drop(columns=['sku'])
test = test.drop(columns=['sku'])

numeric_cols = train.select_dtypes(include=['number']).columns
categorical_cols = train.select_dtypes(include=['object']).columns

# Histogram Plot (For Both Train and Test)
def histogram(df):
  plt.figure(figsize=(15, 20))
  for i, col in enumerate(numeric_cols, 1):
      plt.subplot(5, 3, i)
      plt.hist(df[col], bins=30, color='skyblue', edgecolor='black')
      plt.title(f'Histogram of {col}', fontsize=10)
      plt.xlabel(col, fontsize=8)
      plt.ylabel('Frequency', fontsize=8)
      plt.grid(axis='y', alpha=0.5)

  plt.tight_layout()
  return plt.show()

histogram(train)
histogram(test)

# Boxplot (For Both Train and Test)
def boxplot(df):
  plt.figure(figsize=(15, 20))
  for i, col in enumerate(numeric_cols, 1):
      plt.subplot(5, 3, i)
      sns.boxplot(data=df, x=col, color='lightblue', flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'red'})
      plt.title(f'Boxplot of {col}', fontsize=10)
      plt.xlabel('')

      if col in ['national_inv', 'forecast_3_month', 'sales_1_month']:
          q1 = df[col].quantile(0.25)
          q3 = df[col].quantile(0.75)
          iqr = q3 - q1
          plt.xlim(q1 - 3*iqr, q3 + 3*iqr)

  plt.tight_layout()
  return plt.show()

boxplot(train)
boxplot(test)

# Scatter plot
def pairwise_scatter(df, sample_size=None, grid=True):
  numeric_cols = df.select_dtypes(include=['number']).columns
  n_cols = len(numeric_cols)

  if sample_size and len(df) > sample_size:
      df = df.sample(sample_size)

  plt.figure(figsize=(25, 25))


  combinations = list(itertools.combinations(numeric_cols, 2))

  for i, (x_col, y_col) in enumerate(combinations, 1):
    plt.subplot(n_cols, n_cols, i)

    plt.scatter(x=df[x_col],
                y=df[y_col],
                color='skyblue',
                alpha=0.5,
                s=8,
                edgecolor='black',
                linewidth=0.3)

    plt.title(f'{x_col} vs {y_col}', fontsize=8, pad=4)
    plt.xlabel(x_col, fontsize=6)
    plt.ylabel(y_col, fontsize=6)

    if grid:
        plt.grid(alpha=0.2)

  plt.tight_layout(pad=1.0)
  plt.show()

pairwise_scatter(train)

# Pie Chart
def pie_chart(df):
  class_counts = df['went_on_backorder'].value_counts()

  plt.figure(figsize=(8, 6))
  plt.pie(class_counts,
          labels=['No', 'Yes'],
          autopct='%1.1f%%',
          startangle=90,
          colors=['#66b3ff', '#ff9999'],
          explode=(0.1, 0))

  plt.title('Distribution of Backorder Classes', fontsize=14, pad=20)


  plt.axis('equal')
  plt.tight_layout()
  plt.show()

pie_chart(train)
pie_chart(test)

# Correlation
train.select_dtypes(include="number").corr()

# Correlation
plt.figure(figsize=(15, 15))
sns.heatmap(train.select_dtypes(include="number").corr(), annot=True)

# Negative count
negative_index = []

for i in numeric_cols:
  for x in train.index:
    if train.loc[x, i] < 0:
      negative_index.append(x)

print(train.loc[negative_index])
print(len(negative_index))

def negative_to_nan(df):
  df[numeric_cols] = df[numeric_cols].applymap(lambda x: np.nan if x < 0 else x)

negative_to_nan(train)
negative_to_nan(test)

# Negative count
negative_index = []

for i in numeric_cols:
  for x in train.index:
    if train.loc[x, i] < 0:
      negative_index.append(x)

print(len(negative_index))

print(train.isnull().sum())

print(test.isnull().sum())

train = train.dropna()
test = test.dropna()

# Null values (For Train Data)
train.isnull().sum()

test.isnull().sum()

# Delete columns
train = train.drop(columns=['pieces_past_due', 'local_bo_qty'])
test = test.drop(columns=['pieces_past_due', 'local_bo_qty'])

numeric_cols = train.select_dtypes(include=['number']).columns
categorical_cols = train.select_dtypes(include=['object']).columns

# Checking duplicate values (For Test Data)
print(train.duplicated().sum())
# Checking duplicate values (For Test Data)
print(test.duplicated().sum())

train = train.drop_duplicates()
test = test.drop_duplicates()

boxplot(train)
boxplot(test)

# Outliers
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Outliers
for col in numeric_cols:
    train = cap_outliers(train, col)
    test = cap_outliers(test, col)

# Data info
print("Train Data Info:")
print(train.info())
print("Test Data Info:")
print(test.info())

# First 5 rows
print("First 5 rows of Train Data:")
print(train.head())

# Last 5 rows
print("Last 5 rows of Train Data:")
print(train.tail())

train.isnull().sum()

test.isnull().sum()

# Checking duplicate values (For Test Data)
print(train.duplicated().sum())
# Checking duplicate values (For Test Data)
print(test.duplicated().sum())

train = train.drop_duplicates()
test = test.drop_duplicates()

# Checking garbage values (For Train Data)
for i in categorical_cols:
  print(train[i].value_counts())
  print("****"*10)

# Checking garbage values (For Test Data)
for i in categorical_cols:
  print(test[i].value_counts())
  print("****"*10)

# Describe Data
print("=== Train Data ===")
print(train.describe().T)

print("=== Test Data ===")
print(test.describe().T)

# Describe data
print("=== Train Data ===")
train.describe(include="object")

print("=== Test Data ===")
test.describe(include="object")

histogram(train)
histogram(test)

boxplot(train)
boxplot(test)

pairwise_scatter(train, sample_size=1000)

pie_chart(train)
pie_chart(test)

# Correlation
train.select_dtypes(include="number").corr()

# Correlation
plt.figure(figsize=(15, 15))
sns.heatmap(train.select_dtypes(include="number").corr(), annot=True)

# Categorical Encoding
le = LabelEncoder()
for col in categorical_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Normalaization
scaler = MinMaxScaler(feature_range=(0, 1))
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])

print(train.head())
print(test.head())

# Resampling
X_train = train.drop('went_on_backorder', axis=1)
y_train = train['went_on_backorder']

print("Before resampling:", Counter(y_train))

desired_counts = {0: 40000, 1: 20000}

smote = SMOTE(sampling_strategy={1: 20000}, random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy={0: 40000}, random_state=42)

X_res, y_res = smote.fit_resample(X_train, y_train)
X_res, y_res = under_sampler.fit_resample(X_res, y_res)

print("After resampling:", Counter(y_res))

train = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res, columns=['went_on_backorder'])], axis=1)

pie_chart(train)
pie_chart(test)

# Data info
print("Train Data Info:")
print(train.info())
print("Test Data Info:")
print(test.info())

# First 5 rows
print("First 5 rows of Train Data:")
print(train.head())

# Last 5 rows
print("Last 5 rows of Train Data:")
print(train.tail())

train.isnull().sum()
test.isnull().sum()

# Checking duplicate values (For Test Data)
print(train.duplicated().sum())
# Checking duplicate values (For Test Data)
print(test.duplicated().sum())

# Checking count of Yes/No values (For Train Data)
for i in categorical_cols:
  print(train[i].value_counts())
  print("****"*10)

# Checking count of Yes/No values (For Test Data)
for i in categorical_cols:
  print(test[i].value_counts())
  print("****"*10)

# Describe Data
print("=== Train Data ===")
print(train.describe().T)

print("=== Test Data ===")
print(test.describe().T)

histogram(train)
histogram(test)

boxplot(train)
boxplot(test)

pairwise_scatter(train, sample_size=1000)

# Correlation
train.select_dtypes(include="number").corr()

# Correlation
plt.figure(figsize=(15, 15))
sns.heatmap(train.select_dtypes(include="number").corr(), annot=True)

# Save to new csv file
train.to_csv('Train_Preprocess.csv', index=False)
test.to_csv('Test_Preprocess.csv', index=False)
