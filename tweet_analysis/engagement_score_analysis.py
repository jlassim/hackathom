import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from prettytable import PrettyTable
from prophet import Prophet
from collections import Counter
import re
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

df = pd.read_csv("trending_tweets_cleaned.csv")

# Sort by 'Retweet_count' in descending order and get top 10
top_10_by_retweets = df.sort_values(by='Retweet_count', ascending=False).head(10)

# Display the top 10 posts by retweet count
print("Top 10 Posts by Retweet Count:")
print(top_10_by_retweets[['UserName', 'Tweet_text', 'Retweet_count']])

# Sort by 'Like_count' in descending order and get top 10
top_10_by_likes = df.sort_values(by='Like_count', ascending=False).head(10)

# Display the top 10 posts by like count
print("\nTop 10 Posts by Like Count:")
print(top_10_by_likes[['UserName', 'Tweet_text', 'Like_count']])

# Convert 'Time' column to datetime format
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

# Extract date part
df['Date'] = df['Time'].dt.date

# Group by date and calculate average engagement metrics
grouped_df = df.groupby('Date')[['Retweet_count', 'Like_count', 'engagement_score', 'engagement_rate']].mean()

# Plot the trends
plt.plot(grouped_df.index, grouped_df['Retweet_count'], label='Average Retweets', marker='o')
plt.plot(grouped_df.index, grouped_df['Like_count'], label='Average Likes', marker='s')
plt.plot(grouped_df.index, grouped_df['engagement_score'], label='Engagement Score', marker='^')
plt.plot(grouped_df.index, grouped_df['engagement_rate'], label='Engagement Rate', marker='d')

# Date axis format
plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # Show each day
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format as YYYY-MM-DD

# Add titles and labels
plt.xlabel('Date')
plt.ylabel('Average Values')
plt.title('Engagement Trends on Twitter')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# Afficher le graphique
plt.show()

# Step 1: Select relevant metrics (assuming these columns are present in your data)
metrics = ['Retweet_count', 'Like_count', 'Followers_count', 'engagement_score', 'engagement_rate']

# Step 2: Calculate the correlation matrix
correlation_matrix = df[metrics].corr()
# Step 3: Set up the plot size and styling for clarity
plt.figure(figsize=(12, 8))  # Adjust the size as needed

# Step 4: Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', annot_kws={"size": 10})

# Step 5: Set the title
plt.title('Correlation Matrix of Metrics Affecting Trends')

# Step 6: Display the plot
plt.show()

# Define the weights for each engagement metric based on their importance to virality
weights = {
    'Retweet_count': 0.4,
    'Like_count': 0.4,
    'engagement_score': 0.1,
    'engagement_rate': 0.1
}

# Calculate the viral score as a weighted sum of the metrics
df['viral_score'] = (
    df['Retweet_count'] * weights['Retweet_count'] +
    df['Like_count'] * weights['Like_count'] +
    df['engagement_score'] * weights['engagement_score'] +
    df['engagement_rate'] * weights['engagement_rate']
)

# Sort the DataFrame by viral score in descending order
df_sorted_by_viral_score = df.sort_values(by='viral_score', ascending=False)

# Show the top 10 tweets with the highest viral score
top_10_viral_tweets = df_sorted_by_viral_score[['Time', 'UserName', 'cleaned_Tweet_text','Retweet_count', 'Like_count', 'engagement_score', 'engagement_rate', 'viral_score']].head(10)
# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Disable line width limit
pd.set_option('display.max_colwidth', None)  # Show full content in each cell


# Create table
table = PrettyTable()
table.field_names = top_10_viral_tweets.columns.tolist()

# Add rows
for row in top_10_viral_tweets.itertuples(index=False):
    table.add_row(row)

print("Top 10 Tweets by Viral Score:\n")
print(table)
'''# a faire apres plus de données
#time series forecasting on the viral_score
# Ensure 'Time' is in datetime format
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

# Remove any NaT (invalid dates)
df = df.dropna(subset=['Time'])

# Check if 'viral_score' has NaN values
print(df[['Time', 'viral_score']].isna().sum())  # Debug missing values

# Remove NaN viral scores
df = df.dropna(subset=['viral_score'])

# Aggregate viral score per day (taking the mean)
daily_viral_score = df.groupby(df['Time'].dt.date)['viral_score'].mean().reset_index()

# Check if daily_viral_score is empty
if daily_viral_score.empty:
    raise ValueError("Error: No valid data after grouping. Check your dataset!")

# Rename columns for Prophet
daily_viral_score.columns = ['ds', 'y']

# Display the first few rows for debugging
print(daily_viral_score.head())  # Debugging step

# Ensure there are at least 2 data points
if len(daily_viral_score) < 2:
    raise ValueError("Error: Not enough data points for Prophet!")

# Initialize and fit the Prophet model
model = Prophet()
model.fit(daily_viral_score)
# Create a future dataframe for 30 days
future = model.make_future_dataframe(periods=30)

# Predict future viral scores
forecast = model.predict(future)

# Plot the forecast
fig, ax = plt.subplots(figsize=(10, 5))
model.plot(forecast, ax=ax)

# Customize plot
plt.title("Viral Score Forecast for Next 30 Days")
plt.xlabel("Date")
plt.ylabel("Viral Score")
plt.legend(["Actual", "Forecast", "Trend", "Uncertainty Interval"])
plt.grid(True)
plt.show()
'''
# Convert 'Time' to datetime if not already
df['Time'] = pd.to_datetime(df['Time'])

# Group by hour instead of date to work with a single day's data
hourly_viral_score = df.groupby(df['Time'].dt.hour)['viral_score'].mean().reset_index()

# Rename columns for Prophet format
hourly_viral_score.columns = ['ds', 'y']

# Initialize and fit the Prophet model
model = Prophet()
model.fit(hourly_viral_score)

# Create a future dataframe for the next 24 hours
future = model.make_future_dataframe(periods=24, freq='H')
forecast = model.predict(future)

# Plot the forecast
fig, ax = plt.subplots(figsize=(10, 5))
model.plot(forecast, ax=ax)

# Customize the plot
ax.set_title('Hourly Forecast of Viral Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of the Day', fontsize=12)
ax.set_ylabel('Predicted Viral Score', fontsize=12)

plt.show()
# Find the best hour to post
best_hour = forecast.loc[forecast['yhat'].idxmax(), 'ds'].hour

print(f"\nBest Time to Post: {best_hour}:00")

# Calculate average engagement metrics for top viral posts
avg_likes = top_10_viral_tweets['Like_count'].mean()
avg_retweets = top_10_viral_tweets['Retweet_count'].mean()
avg_engagement_score = top_10_viral_tweets['engagement_score'].mean()

print(f"\nAverage Likes in Viral Tweets: {avg_likes:.2f}")
print(f"Average Retweets in Viral Tweets: {avg_retweets:.2f}")
print(f"Average Engagement Score: {avg_engagement_score:.2f}")

#topic modeling (LDA) for the most viral posts only
# Load stopwords
stop_words = stopwords.words('english')

# Remove stopwords from 'cleaned_Tweet_text'
top_10_viral_tweets['cleaned_Tweet_text_no_stopwords'] = top_10_viral_tweets['cleaned_Tweet_text'].apply(
    lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words])
)


# Vectorize the text with TF-IDF (removing English stopwords)
vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w+\b')
X = vectorizer.fit_transform(top_10_viral_tweets['cleaned_Tweet_text_no_stopwords'])

# Check the shape of the matrix to confirm the transformation
print(X.shape)


# Apply LDA for topic modeling
num_topics = 5  # Number of topics to extract
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Get the top words for each topic
terms = vectorizer.get_feature_names_out()
topic_words = {}

for topic_idx, topic in enumerate(lda.components_):
    topic_words[topic_idx] = [terms[i] for i in topic.argsort()[:-11:-1]]  # Get top 10 words for each topic

# Print the top words for each topic
for topic_idx, words in topic_words.items():
    print(f"Topic {topic_idx}: {' '.join(words)}")

# Convert the topic_words dictionary to a DataFrame for better readability
topic_df = pd.DataFrame(topic_words).transpose()
topic_df.columns = [f"Word {i+1}" for i in range(topic_df.shape[1])]

# Save the themes (topics) to a CSV file
topic_df.to_csv("most_common_themes.csv", index=True)