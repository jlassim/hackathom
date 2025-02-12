import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
import networkx as nx
from collections import Counter


# Charger le dataset prétraité
df = pd.read_csv("twitter_cleaned.csv")

#analyse sentimental
# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to apply sentiment analysis
def analyze_sentiment(text):
    sentiment_score = analyzer.polarity_scores(text)
    # Classify sentiment based on compound score
    if sentiment_score['compound'] >= 0.05:
        return 'positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply the function to the cleaned descriptions
df['sentiment'] = df['cleaned_description'].apply(analyze_sentiment)

sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)


# Visualisation avec un pie chart
plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["#4CAF50", "#9E9E9E", "#F44336"])
plt.title("Répartition des Sentiments des Tweets")
plt.show()
# Strip plot for sentiment vs engagement (with jitter for better visibility)
sns.stripplot(x='sentiment', y='engagement_score', data=df, jitter=True, palette="muted", dodge=True)
plt.title('Engagement Score by Sentiment')
plt.show()

# Sauvegarder les résultats
df.to_csv("twitter_sentiment_analysis.csv", index=False)


#trend topics
# Vectorize the cleaned descriptions using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned_description'])

# Apply LDA to find topics
lda = LatentDirichletAllocation(n_components=5, random_state=42)  # You can choose the number of topics
lda.fit(X)

# Display the topics with the most significant words
n_top_words = 10
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx}:")
    print(" ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))



