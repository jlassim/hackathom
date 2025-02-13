import pandas as pd
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import ast
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px


df = pd.read_csv("trending_tweets_cleaned.csv")

# Flatten the list of hashtags for frequency analysis, excluding empty or malformed entries
all_hashtags = []
for sublist in df['All_Hashtags']:
    if isinstance(sublist, str):  # Ensure it's a string before processing
        # Remove the brackets, split by commas, and strip spaces from each hashtag
        hashtags = sublist.strip("[]").replace("'", "").split(',')
        all_hashtags.extend([tag.strip() for tag in hashtags if tag.strip() != ''])

# Hashtag Frequency Analysis
hashtag_freq = Counter(all_hashtags)
print("Top 10 Most Frequent Hashtags:")
print(hashtag_freq.most_common(10))

# Plot the top 10 hashtags
top_hashtags = hashtag_freq.most_common(10)
top_hashtags_df = pd.DataFrame(top_hashtags, columns=['Hashtag', 'Frequency'])
plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Hashtag', data=top_hashtags_df, palette='viridis')
plt.title('Top 10 Hashtags by Frequency')
plt.xlabel('Frequency')
plt.ylabel('Hashtag')
plt.show()

# Initialize a list to hold the hashtag pairs
hashtag_pairs = []

# Iterate through each row in the 'All_Hashtags' column
for sublist in df['All_Hashtags']:
    if isinstance(sublist, str):  # Ensure it's a string before processing
        # Remove the brackets, split by commas, and strip spaces from each hashtag
        hashtags = sublist.strip("[]").replace("'", "").split(',')
        hashtags = [tag.strip() for tag in hashtags if tag.strip() != '']
        
        if len(hashtags) > 1:  # Make sure there are at least 2 hashtags to form pairs
            # Generate all possible pairs of hashtags in the same tweet
            pairs = itertools.combinations(hashtags, 2)
            hashtag_pairs.extend(pairs)

# Count the frequency of each pair
pair_freq = Counter(hashtag_pairs)

# Show the top 10 co-occurring hashtag pairs
top_pairs = pair_freq.most_common(10)
top_pairs_df = pd.DataFrame(top_pairs, columns=['Hashtag Pair', 'Frequency'])

# Format pair as 'Hashtag1 & Hashtag2'
top_pairs_df['Hashtag Pair'] = top_pairs_df['Hashtag Pair'].apply(lambda x: ' & '.join(x))

# Print the top 10 co-occurring pairs
print("Top 10 Co-occurring Hashtag Pairs:")
print(top_pairs_df)
# Plotting the result with adjustments
plt.figure(figsize=(12, 6))  # Increase figure size to accommodate longer labels

# Create a horizontal bar plot
sns.barplot(x='Frequency', y='Hashtag Pair', data=top_pairs_df, palette='viridis')

# Adjust font size for labels
plt.title('Top 10 Co-occurring Hashtag Pairs', fontsize=14)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Hashtag Pair', fontsize=12)

# Adjust the font size of the y-axis labels for better readability
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Adjust the plot's position to the right
plt.subplots_adjust(left=0.2)  # Shifting the plot to the right
plt.show()

# Step 1: Convert string representations of lists into actual lists if needed
df['All_Hashtags'] = df['All_Hashtags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Step 2: Filter out rows with empty lists in the 'All_Hashtags' column
df_with_hashtags = df[df['All_Hashtags'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

# Step 3: Flatten the lists of hashtags (for clustering purposes)
all_hashtags = []
for sublist in df_with_hashtags['All_Hashtags']:
    all_hashtags.extend(sublist)


# Step 4: Check if we have valid hashtags
if not all_hashtags:
    print("No valid hashtags found. Exiting.")
else:
    # Step 5: Vectorize the hashtags using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w+\b')
    X = vectorizer.fit_transform(all_hashtags)

    # Step 6: Apply KMeans Clustering
    num_clusters = 5  # Adjust this value as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    # Step 7: Get the top terms for each cluster (optional)
    terms = vectorizer.get_feature_names_out()
    cluster_terms = {i: [] for i in range(num_clusters)}

    for i in range(num_clusters):
        cluster_terms[i] = [terms[j] for j in range(len(terms)) if kmeans.cluster_centers_[i, j] > 0.1]

    # Step 8: Assign clusters to hashtags
    labels = kmeans.predict(vectorizer.transform(all_hashtags))

    # Step 9: Create DataFrame with hashtags and their clusters
    hashtags_df = pd.DataFrame({
        'Hashtag': all_hashtags,
        'Cluster': labels
    })

    # Step 10: Apply PCA for dimensionality reduction (for visualization)
    X_reduced = PCA(n_components=2).fit_transform(X.toarray())
    hashtags_df['PCA1'] = X_reduced[:, 0]
    hashtags_df['PCA2'] = X_reduced[:, 1]

    # Step 11: Plot the clusters with labeled clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=hashtags_df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100, alpha=0.7)

    # Step 12: Add cluster names as labels
    for cluster_num, terms in cluster_terms.items():
        # We no longer calculate the mean of non-numeric data
        cluster_center = hashtags_df[hashtags_df['Cluster'] == cluster_num][['PCA1', 'PCA2']].mean()
        plt.text(cluster_center['PCA1'], cluster_center['PCA2'],
                 f"Cluster {cluster_num}: {' '.join(terms)}", fontsize=12, ha='center', va='center')

    plt.title('Hashtag Clustering (KMeans)')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend(title='Cluster', loc='upper right')
    plt.show()

    # Step 13: Print the top hashtags in each cluster
    for cluster_num, terms in cluster_terms.items():
        print(f"Cluster {cluster_num}: {' '.join(terms)}")

# Step 3: Vectorize hashtags using CountVectorizer
vectorizer = CountVectorizer(stop_words='english', token_pattern=r'\b\w+\b')
X = vectorizer.fit_transform(all_hashtags)

# Step 4: Apply LDA for Topic Modeling
num_topics = 5  # Adjust number of topics based on your data
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Step 5: Display the top words for each topic
terms = vectorizer.get_feature_names_out()
topic_words = {}
for topic_idx, topic in enumerate(lda.components_):
    topic_words[topic_idx] = [terms[i] for i in topic.argsort()[:-11:-1]]

# Print the top words in each topic
for topic_idx, words in topic_words.items():
    print(f"Topic {topic_idx}: {' '.join(words)}")

# Step 6: Visualize each topic with a WordCloud
for topic_idx, words in topic_words.items():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Topic {topic_idx} WordCloud')
    plt.show()

# Step 1: Prepare data for interactive visualization with Plotly
topic_proportions = lda.transform(X)

# Step 2: Create a DataFrame to visualize the distribution of topics
topic_df = pd.DataFrame(topic_proportions, columns=[f"Topic {i}" for i in range(num_topics)])

# Add hashtags for context in hover data
topic_df['Hashtags'] = all_hashtags

# Step 3: Generate scatter plots for all unique pairs of topics
topic_pairs = [(i, j) for i in range(num_topics) for j in range(i+1, num_topics)]

# Loop through each pair of topics and create a scatter plot for them
for topic1, topic2 in topic_pairs:
    # Create the scatter plot for the current topic pair
    fig = px.scatter(topic_df, x=f'Topic {topic1}', y=f'Topic {topic2}', color='Hashtags', hover_data=['Hashtags'],
                     title=f'Interactive LDA Topic Modeling - Topics {topic1} vs {topic2}')
    
    # Customize layout
    fig.update_layout(
        title=f'LDA Topic Modeling of Hashtags: Topic {topic1} vs Topic {topic2}',
        xaxis_title=f'Topic {topic1}',
        yaxis_title=f'Topic {topic2}'
    )
    
    # Show the plot for the current pair of topics
    fig.show()
