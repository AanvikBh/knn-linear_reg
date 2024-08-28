import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os

# Load the data
df = pd.read_csv('../../data/interim/1/knn/spotify_unique.csv')  # Replace with your actual file name

# Separate numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns


print(df['key'].isnull().sum())  # Count of null values in the 'key' column
df = df.dropna(subset=['key'])  # Drop rows where 'key' is null

# stop = input()



# Normalize numeric data
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Save normalized data
df_normalized.to_csv('../../data/interim/1/plots/normalized_spotify.csv', index=False)

# Create a folder for plots
if not os.path.exists('spotify_plots_2'):
    os.makedirs('spotify_plots_2')

# # Plot original vs normalized data
# for col in numeric_columns:
#     plt.figure(figsize=(12, 6))
#     plt.subplot(121)
#     plt.hist(df[col], bins=50)
#     plt.title(f'Original {col}')
#     plt.subplot(122)
#     plt.hist(df_normalized[col], bins=50)
#     plt.title(f'Normalized {col}')
#     plt.tight_layout()
#     plt.savefig(f'spotify_plots/original_vs_normalized_{col}.png')
#     plt.close()


# Scatter plots for original vs normalized data
for col in numeric_columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(df.index, df[col], alpha=0.5, label='Original')
    plt.scatter(df.index, df_normalized[col], alpha=0.5, label='Normalized')
    plt.xlabel('Index')
    plt.ylabel(col)
    plt.title(f'Original vs Normalized: {col}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'spotify_plots/original_vs_normalized_{col}.png')
    plt.close()



# Bar graphs for categorical data
categorical_columns = ['artists', 'album_name', 'track_name', 'track_genre']
for col in categorical_columns:
    plt.figure(figsize=(12, 6))
    df[col].value_counts().head(50).plot(kind='bar')
    plt.title(f'Top 50 {col}')
    plt.tight_layout()
    plt.savefig(f'spotify_plots/bar_graph_{col}.png')
    plt.close()

# Scatter plots for numeric data
for i in range(len(numeric_columns)):
    for j in range(i+1, len(numeric_columns)):
        plt.figure(figsize=(10, 6))
        plt.scatter(df_normalized[numeric_columns[i]], df_normalized[numeric_columns[j]], alpha=0.5)
        plt.xlabel(numeric_columns[i])
        plt.ylabel(numeric_columns[j])
        plt.title(f'Scatter plot: {numeric_columns[i]} vs {numeric_columns[j]}')
        plt.savefig(f'spotify_plots/scatter_{numeric_columns[i]}_{numeric_columns[j]}.png')
        plt.close()

# Heatmap of correlations
plt.figure(figsize=(12, 10))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('spotify_plots/correlation_heatmap.png')
plt.close()

# Box plots for numeric data
for col in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='track_genre', y=col, data=df_normalized)
    plt.xticks(rotation=90)
    plt.title(f'Box plot: {col} by Track Genre')
    plt.tight_layout()
    plt.savefig(f'spotify_plots/boxplot_{col}.png')
    plt.close()

# Heatmaps for numeric data
for col in numeric_columns:
    plt.figure(figsize=(12, 8))
    pivot = df.pivot_table(index='track_genre', columns='key', values=col, aggfunc='mean')
    sns.heatmap(pivot, annot=True, cmap='YlGnBu')
    plt.title(f'Heatmap: Average {col} by Track Genre and Key')
    plt.tight_layout()
    plt.savefig(f'spotify_plots/heatmap_{col}.png')
    plt.close()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import MinMaxScaler
# import os

# # Load the data
# df = pd.read_csv('../../data/interim/1/knn/spotify_unique.csv')

# # Separate numeric columns
# numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# # Normalize numeric data
# scaler = MinMaxScaler()
# df_normalized = df.copy()
# df_normalized[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# # Save normalized data
# df_normalized.to_csv('../../data/interim/1/plots/normalized_spotify.csv', index=False)

# # Create a folder for plots
# if not os.path.exists('spotify_plots'):
#     os.makedirs('spotify_plots')

# # Scatter plots for original vs normalized data
# for col in numeric_columns:
#     plt.figure(figsize=(10, 6))
#     plt.scatter(df.index, df[col], alpha=0.5, label='Original')
#     plt.scatter(df.index, df_normalized[col], alpha=0.5, label='Normalized')
#     plt.xlabel('Index')
#     plt.ylabel(col)
#     plt.title(f'Original vs Normalized: {col}')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'spotify_plots/original_vs_normalized_{col}.png')
#     plt.close()

# # Bar graphs for categorical data (showing all classes)
# categorical_columns = ['artists', 'album_name', 'track_name', 'track_genre']
# for col in categorical_columns:
#     plt.figure(figsize=(20, 10))
#     df[col].value_counts().plot(kind='bar')
#     plt.title(f'All classes in {col}')
#     plt.xlabel(col)
#     plt.ylabel('Count')
#     plt.xticks(rotation=90)
#     plt.tight_layout()
#     plt.savefig(f'spotify_plots/bar_graph_{col}.png')
#     plt.close()

# # Scatter plots for numeric data (using original data)
# for i in range(len(numeric_columns)):
#     for j in range(i+1, len(numeric_columns)):
#         plt.figure(figsize=(10, 6))
#         plt.scatter(df[numeric_columns[i]], df[numeric_columns[j]], alpha=0.5)
#         plt.xlabel(numeric_columns[i])
#         plt.ylabel(numeric_columns[j])
#         plt.title(f'Scatter plot: {numeric_columns[i]} vs {numeric_columns[j]}')
#         plt.savefig(f'spotify_plots/scatter_{numeric_columns[i]}_{numeric_columns[j]}.png')
#         plt.close()

# # Heatmap of correlations
# plt.figure(figsize=(12, 10))
# sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.tight_layout()
# plt.savefig('spotify_plots/correlation_heatmap.png')
# plt.close()

# # Box plots for numeric data (using original data)
# for col in numeric_columns:
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x='track_genre', y=col, data=df)
#     plt.xticks(rotation=90)
#     plt.title(f'Box plot: {col} by Track Genre')
#     plt.tight_layout()
#     plt.savefig(f'spotify_plots/boxplot_{col}.png')
#     plt.close()

# # Heatmaps for numeric data
# for col in numeric_columns:
#     plt.figure(figsize=(12, 8))
#     pivot = df.pivot_table(index='track_genre', columns='key', values=col, aggfunc='mean')
#     sns.heatmap(pivot, annot=True, cmap='YlGnBu')
#     plt.title(f'Heatmap: Average {col} by Track Genre and Key')
#     plt.tight_layout()
#     plt.savefig(f'spotify_plots/heatmap_{col}.png')
#     plt.close()