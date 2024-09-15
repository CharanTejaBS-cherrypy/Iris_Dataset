# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset from seaborn
df = sns.load_dataset('iris')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Basic statistical analysis
print("\nBasic statistical description of the dataset:")
print(df.describe())

# Checking the distribution of the species column
print("\nDistribution of species:")
print(df['species'].value_counts())

# Data Visualization
# 1. Pairplot to visualize relationships between features
print("\nDisplaying pairplot...")
sns.pairplot(df, hue='species')
plt.show()

# 2. Correlation Heatmap
# Calculate correlation matrix
corr_matrix = df.drop(columns='species').corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# 3. Visualizing individual feature distributions
# Histogram for Sepal Length
plt.figure()
sns.histplot(df['sepal_length'], kde=True, color='blue')
plt.title('Sepal Length Distribution')
plt.show()

# Boxplot for Sepal Width across different species
plt.figure()
sns.boxplot(x='species', y='sepal_width', data=df)
plt.title('Sepal Width by Species')
plt.show()
