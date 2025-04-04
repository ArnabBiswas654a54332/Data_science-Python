import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

# Load the dataset
df = pd.read_csv('titanic.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Check for missing values in all columns
print("\nMissing values in each column:")
missing_values = df.isnull().sum()
print(missing_values)

# Basic statistics
print("\nBasic statistics of the dataset:")
print(df.describe())

# Count of passengers by gender
print("\nCount of passengers by gender:")
print(df['Sex'].value_counts())

# Count of passengers by passenger class
print("\nCount of passengers by passenger class:")
print(df['Pclass'].value_counts())

# Survival rate
print("\nSurvival rate:")
print(df['Survived'].value_counts(normalize=True))

# Visualization: Survival count by gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.show()

# Visualization: Age distribution of passengers
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'].dropna(), kde=True, bins=20)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Visualization: Fare distribution by passenger class
plt.figure(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution by Passenger Class')
plt.show()

# Handling missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing Embarked values with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop the Cabin column due to excessive missing values
df.drop('Cabin', axis=1, inplace=True)

# Using KNN Imputation for Age (optional, if you want to use advanced imputation)
imputer = KNNImputer(n_neighbors=5)
df['Age'] = imputer.fit_transform(df[['Age']])

# Check if missing values are handled
print("\nMissing values after handling:")
print(df.isnull().sum())

# Display the cleaned dataset
print("\nCleaned dataset:")
print(df.head())
