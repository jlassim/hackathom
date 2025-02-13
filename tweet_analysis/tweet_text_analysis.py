import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet

df = pd.read_csv("trending_tweets_cleaned.csv")

# Initialize the Hugging Face pipeline for sentiment analysis using the pre-trained model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
classifier = pipeline('sentiment-analysis', model=model_name)

# Function to get the sentiment from the model
def get_sentiment(tweet):
    result = classifier(tweet)
    return result[0]['label']

# Apply the sentiment model to the cleaned tweets
df['sentiment'] = df['cleaned_Tweet_text'].apply(get_sentiment)

# Visualize the sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df, palette='Set2')
plt.title('Sentiment Distribution of Tweets (Using Roberta Model)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Wordcloud for positive, neutral, and negative tweets
positive_tweets = ' '.join(df[df['sentiment'] == 'LABEL_2']['cleaned_Tweet_text'])
neutral_tweets = ' '.join(df[df['sentiment'] == 'LABEL_1']['cleaned_Tweet_text'])
negative_tweets = ' '.join(df[df['sentiment'] == 'LABEL_0']['cleaned_Tweet_text'])

def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

generate_wordcloud(positive_tweets, 'Positive Sentiment WordCloud')
generate_wordcloud(neutral_tweets, 'Neutral Sentiment WordCloud')
generate_wordcloud(negative_tweets, 'Negative Sentiment WordCloud')

# Trend Analysis: Sentiment Over Time (Daily)
df['Date'] = pd.to_datetime(df['Time']).dt.date
daily_sentiment = df.groupby('Date')['sentiment'].apply(lambda x: x.value_counts().idxmax())

# Visualize sentiment trend over time
plt.figure(figsize=(12, 6))
daily_sentiment.value_counts().sort_index().plot()
plt.title('Sentiment Trend Over Time (Daily)')
plt.xlabel('Date')
plt.ylabel('Sentiment Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Trend Analysis: Sentiment vs Engagement Metrics (Retweets, Likes)
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Retweet_count', y='sentiment', data=df, label='Retweets', color='b', alpha=0.6)
sns.scatterplot(x='Like_count', y='sentiment', data=df, label='Likes', color='g', alpha=0.6)
plt.title('Sentiment vs Engagement Metrics (Retweets and Likes)')
plt.xlabel('Engagement Metrics')
plt.ylabel('Sentiment')
plt.legend()
plt.grid(True)
plt.show()

# Detection Model: Classifying Tweet Sentiments (Positive, Neutral, Negative)
# Convert the sentiment labels to numerical values for classification
encoder = LabelEncoder()
df['sentiment_numeric'] = encoder.fit_transform(df['sentiment'])

# Convert sentiment to numeric values for correlation analysis
encoder = LabelEncoder()
df['sentiment_numeric'] = encoder.fit_transform(df['sentiment'])

# Calculate correlation between sentiment and engagement metrics
correlation = df[['sentiment_numeric', 'Retweet_count', 'Like_count', 'engagement_score', 'engagement_rate']].corr()

# Reduce figure size for better display
plt.figure(figsize=(6, 4))  # Adjusted smaller size

# Generate heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", annot_kws={"size": 8})

# Title and labels
plt.title('Correlation Between Sentiment & Engagement Metrics', fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
# Show plot
plt.show()

# Scatter plots to visualize the correlation
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sentiment_numeric', y='Retweet_count', data=df, color='b', label='Retweets')
sns.scatterplot(x='sentiment_numeric', y='Like_count', data=df, color='g', label='Likes')
plt.title('Sentiment vs Engagement Metrics (Retweets and Likes)')
plt.xlabel('Sentiment (Numeric)')
plt.ylabel('Engagement Metrics')
plt.legend()
plt.show()

'''# we need more postes for diffrent days at least 2 days 
# 2. Forecasting: Using Prophet to predict viral trends

# Prepare the data for forecasting (sentiment and engagement metrics over time)
df['Date'] = pd.to_datetime(df['Time']).dt.date  # Ensure the 'Date' column is in date format

# Aggregating the daily data: We can aggregate by daily mean or sum
daily_data = df.groupby('Date').agg({
    'sentiment_numeric': 'mean',
    'Retweet_count': 'sum',  # You can also use 'mean' for average retweets per day
    'Like_count': 'sum',     # You can also use 'mean' for average likes per day
    'engagement_score': 'mean',
    'engagement_rate': 'mean'
}).reset_index()

# Prepare the data for Prophet: Prophet expects columns 'ds' (date) and 'y' (target variable)
daily_data['ds'] = pd.to_datetime(daily_data['Date'])
daily_data['y'] = daily_data['Retweet_count']  # You can use other engagement metrics here

# Initialize and fit the Prophet model
model = Prophet()
model.fit(daily_data[['ds', 'y']])

# Create a future dataframe for 30 days
future = model.make_future_dataframe(daily_data, periods=30)

# Predict the future values
forecast = model.predict(future)

# Plot the forecast
model.plot(forecast)
plt.title('Viral Trend Forecast for Retweets')
plt.xlabel('Date')
plt.ylabel('Predicted Retweets')
plt.show()
'''


