import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False  # Pour éviter les problèmes avec les signes moins
from collections import Counter
matplotlib.use('TkAgg')  # Or try 'Agg' if you're running this in a non-interactive environment
from sklearn.linear_model import LinearRegression

matplotlib.use('Agg')  # Use the Agg backend

# Your plotting code here


# Charger le dataset prétraité
df = pd.read_csv("twitter_cleaned.csv")

# Distribution of 'likes'
sns.histplot(df['likes'], kde=True, color='royalblue')
plt.title('Distribution of Likes')
plt.xlabel('Likes')
plt.ylabel('Frequency')
plt.show()

reposts = df['reposts']

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(reposts, bins=30, color='skyblue', edgecolor='black')

# Customize the plot
plt.title('Distribution of Retweets (Reposts)')
plt.xlabel('Number of Retweets')
plt.ylabel('Frequency')
plt.grid(True)

# Show the plot
plt.show()

# Distribution of 'replies'
sns.histplot(df['replies'], kde=True, color='firebrick')
plt.title('Distribution of Replies')
plt.xlabel('Replies')
plt.ylabel('Frequency')
plt.show()

# Distribution of 'quotes'
sns.histplot(df['quotes'], kde=True, color='darkorange')
plt.title('Distribution of Quotes')
plt.xlabel('Quotes')
plt.ylabel('Frequency')
plt.show()

# Distribution of 'bookmarks'
sns.histplot(df['bookmarks'], kde=True, color='goldenrod')
plt.title('Distribution of Bookmarks')
plt.xlabel('Bookmarks')
plt.ylabel('Frequency')
plt.show()

# Visualisation du score d'engagement
sns.histplot(df['engagement_score'], kde=True, color='purple')
plt.title('Distribution of engagement_score')
plt.xlabel('engagement_score')
plt.ylabel('Frequency   ')
plt.show()

# Followers Distribution
sns.histplot(df['followers'], kde=True, color='orange')
plt.title('Followers Distribution')
plt.xlabel('Followers')
plt.ylabel('Frequency')
plt.show()

# Views Distribution
sns.histplot(df['views'], kde=True, color='red')
plt.title('Views Distribution')
plt.xlabel('Views')
plt.ylabel('Frequency')
plt.show()





