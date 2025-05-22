import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Sample 16 rows from the dataset
sample_df = df.sample(n=100)

# Rename columns
sample_df = sample_df.rename(columns={'sepal length (cm)': 'sepal_length', 
                                      'sepal width (cm)': 'sepal_width', 
                                      'petal length (cm)': 'petal_length', 
                                      'petal width (cm)': 'petal_width'})

# Save to csv
sample_df.to_csv('data/new_data.csv', index=False)
