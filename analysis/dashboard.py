import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False  # Pour éviter les problèmes avec les signes moins
from collections import Counter
matplotlib.use('TkAgg')  # Or try 'Agg' if you're running this in a non-interactive environment
from sklearn.linear_model import LinearRegression

matplotlib.use('Agg') 

#dashboard
# Charger le dataset prétraité
df = pd.read_csv("twitter_cleaned.csv")
# Calculating correlations
corr = df[['likes', 'reposts', 'replies', 'quotes', 'bookmarks', 'followers', 'views']].corr()

#partie 1
# Displaying the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Between Different Engagement Metrics')
plt.show()


# Number of hashtags per post
df['num_hashtags'] = df['hashtags'].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)

sns.histplot(df['num_hashtags'], kde=True, color='brown')
plt.title('Distribution of Number of Hashtags per Post')
plt.xlabel('Number of Hashtags')
plt.ylabel('Frequency')
plt.show()


# Configurer Matplotlib pour ignorer les avertissements de glyphes manquants
matplotlib.rcParams['axes.unicode_minus'] = False  # Pour éviter les problèmes avec les signes moins

# Assuming df is your cleaned DataFrame
# Convert 'hashtags' column to a list of hashtags
df['hashtags'] = df['hashtags'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

# Flatten the list of hashtags and count occurrences
all_hashtags = [hashtag for sublist in df['hashtags'] for hashtag in sublist]
hashtag_counts = Counter(all_hashtags)

# Get the top 10 trending hashtags
top_10_hashtags = hashtag_counts.most_common(10)

# Prepare data for the bar chart
hashtags, counts = zip(*top_10_hashtags)

# Create the bar chart with adjusted size
plt.figure(figsize=(6, 4))  
plt.barh(hashtags, counts, color='skyblue')
plt.xlabel('Frequency')
plt.ylabel('Hashtags')
plt.title('Top 10 Trending Hashtags')
plt.gca().invert_yaxis()  # To display the highest counts at the top

# Show the plot
plt.show()


# Your previous code to create the bar chart
top_followed = df[['user_posted', 'followers']].sort_values(by='followers', ascending=False).head(5)

plt.figure(figsize=(10, 6))
plt.bar(top_followed['user_posted'], top_followed['followers'], color='lightcoral')
plt.title('Top 5 Most Followed Users')
plt.xlabel('Users')
plt.ylabel('Number of Followers')
plt.xticks(rotation=45, ha='right')

# Save the plot as a PNG image
plt.savefig('top_5_followed_users.png')

# Optionally, close the plot to free memory
plt.close()

#partie2


# Create a scatter plot with a different color (e.g., 'red')
plt.figure(figsize=(10, 6))
sns.scatterplot(x='likes', y='reposts', data=df, color='red', alpha=0.6)

# Add titles and labels
plt.title('Scatter Plot of Tweets based on Likes and Reposts', fontsize=16)
plt.xlabel('Number of Likes', fontsize=14)
plt.ylabel('Number of Reposts', fontsize=14)

# Save the plot to a file (e.g., 'scatter_plot.png')
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')

# Optionally, close the plot to free up memory
plt.close()

# Step 1: Separate the dataset based on content types (photos, videos, and description)
# Posts with photos
photo_posts = df[df['photos'].notnull() & (df['photos'] != '')]
# Posts with videos
video_posts = df[df['videos'].notnull() & (df['videos'] != '')]
# Posts with description (assuming 'description' content means posts without photos or videos)
description_posts = df[df['description'].notnull() & (df['description'] != '')]

# Step 2: Create a DataFrame with summed engagement (likes) for each content type
engagement_data = {
    'Content Type': ['Photos', 'Videos', 'description'],
    'Likes': [photo_posts['likes'].sum(), video_posts['likes'].sum(), description_posts['likes'].sum()],
    'Views': [photo_posts['views'].sum(), video_posts['views'].sum(), description_posts['views'].sum()],
    'Reposts': [photo_posts['reposts'].sum(), video_posts['reposts'].sum(), description_posts['reposts'].sum()]
}

engagement_df = pd.DataFrame(engagement_data)

# Step 3: Plot the data on the same bar chart
plt.figure(figsize=(10, 6))

# Plotting each category separately
plt.bar(engagement_df['Content Type'], engagement_df['Likes'], label='Likes', color='#1f77b4', alpha=0.7)
plt.bar(engagement_df['Content Type'], engagement_df['Views'], label='Views', color='#ff7f0e', alpha=0.7, bottom=engagement_df['Likes'])
plt.bar(engagement_df['Content Type'], engagement_df['Reposts'], label='Reposts', color='#2ca02c', alpha=0.7, bottom=engagement_df['Likes'] + engagement_df['Views'])

# Adding labels and title
plt.title('Engagement Distribution for Different Content Types')
plt.xlabel('Content Type')
plt.ylabel('Total Engagement')
plt.legend()

# Save the plot as an image file
plt.savefig("engagement_distribution.png", dpi=300)

# Close the plot to free up resources
plt.close()

#partie3
#engagement_score prediction 
# Prepare features and target
features = ['followers', 'num_hashtags'] #features = ['followers', 'num_hashtags', 'hour', 'day_of_week_num']
target = 'engagement_score'

# Train a regression model
X_train = df[features]
y_train = df[target]

model = LinearRegression()
model.fit(X_train, y_train)

# Predict engagement scores
df['predicted_engagement'] = model.predict(X_train)

# Check if 'predicted_engagement' is correctly added to the dataframe
print(df.head())  # This will display the first few rows and confirm the column is there

# Visualize actual vs predicted engagement scores
plt.figure(figsize=(8, 6))

# Scatter plot for actual engagement scores
sns.scatterplot(x=df['engagement_score'], y=df['predicted_engagement'], 
                color='blue', label='Actual Engagement', s=100, marker='o')

# Scatter plot for predicted engagement scores
sns.scatterplot(x=df['engagement_score'], y=df['predicted_engagement'], 
                color='red', label='Predicted Engagement', s=100, marker='X')

# Add labels and title
plt.title('Actual vs Predicted Engagement Score')
plt.xlabel('Actual Engagement Score')
plt.ylabel('Predicted Engagement Score')

# Add legend
plt.legend(title='Engagement Type', loc='upper left')

# Save the plot as an image file
plt.savefig("predictions.png", dpi=300)
# Close the plot to free up resources
plt.close()
