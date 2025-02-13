import pandas as pd
import emoji
import re
from sklearn.preprocessing import MinMaxScaler
import ast
import re


df = pd.read_csv("trending_tweets.csv")

# Define a threshold for missing values
threshold = 600

# Remove columns where the number of missing values exceeds the threshold
df_cleaned = df.loc[:, df.isnull().sum() <= threshold]

#Removes rows where all columns are identical.
df = df.drop_duplicates()

# Step 1: Fill missing values in the 'Followers_count' column with 0
df['Followers_count'] = pd.to_numeric(df['Followers_count'], errors='coerce')  # Convert to numeric, replacing errors with NaN
df['Followers_count'] = df['Followers_count'].fillna(0)

# Step 3: Fill missing values in the 'Description' column with 'Not specified'
df['Description'] = df['Description'].fillna('Not specified')

# Step 4: Fill missing values in the 'Tweet_text' column with 'No content'
df['Tweet_text'] = df['Tweet_text'].fillna('No content')

#step5:Fill missing values for 'Time' with 'Unknown'
df['Time'] = df['Time'].fillna('Unknown')
#Extract features
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

df['Hour_of_Post'] = df['Time'].apply(lambda x: x.hour if x != 'unknown' else 'unknown')
df['Day_of_Week'] = df['Time'].apply(lambda x: x.dayofweek if x != 'unknown' else 'unknown')
df['Month'] = df['Time'].apply(lambda x: x.month if x != 'unknown' else 'unknown')
df['Year'] = df['Time'].apply(lambda x: x.year if x != 'unknown' else 'unknown')
df['Minute_of_Post'] = df['Time'].apply(lambda x: x.minute if x != 'unknown' else 'unknown')
df['Second_of_Post'] = df['Time'].apply(lambda x: x.second if x != 'unknown' else 'unknown')
df['Is_Weekend'] = df['Time'].apply(lambda x: (x.dayofweek >= 5) if x != 'unknown' else 'unknown')

# Step 6: Fill missing values for 'UserName' with 'Unknown User'
df['UserName'] = df['UserName'].fillna('Unknown User')

# Step 7: Fill missing values for 'All_Hashtags' with 'No hashtags'
df['All_Hashtags'] = df['All_Hashtags'].fillna('No hashtags')
df['All_Hashtags'] = df['All_Hashtags'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") 
    else (x.split() if isinstance(x, str) and x.lower() != "no hashtags" else [])
)

# Fonction pour nettoyer les textes des tweets


# Dictionnaire d'émojis avec sentiments associés
emoji_sentiment_dict = {
    '😀': 'positif', '😊': 'positif', '😍': 'positif', '😎': 'positif',
    '😢': 'négatif', '😞': 'négatif', '😡': 'négatif', '😭': 'négatif',
    '😱': 'négatif', '🙁': 'négatif', '😔': 'négatif',
    '😃': 'positif', '😅': 'positif', '😜': 'positif',
    '💥': 'négatif', '🔥': 'négatif'
    # Ajoutez plus d'émojis et de sentiments si nécessaire
}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    

    # Enlever les URLs, mentions, hashtags et caractères spéciaux
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Supprimer les URLs
    text = re.sub(r"@\w+", "", text)  # Supprimer les mentions (@)
    text = re.sub(r"#\w+", "", text)  # Supprimer les hashtags 
    # Retirer les espaces supplémentaires
    text = re.sub(r'\s+', ' ', text).strip()  
    # Supprimer la ponctuation excessive (répétée 2 fois ou plus)
    text = re.sub(r'([!?.])\1+', r'\1', text)  # Ex: "!!!!" -> "!"


    return text.lower()  # Convertir le texte en minuscule pour l'analyse de sentiment
# Fonction pour extraire et associer les émojis à des sentiments
def extract_emoji_sentiment(text):
    emojis_in_text = emoji.emoji_list(text)  # Liste d'émojis présents dans le texte
    sentiments = []
    
    for emj in emojis_in_text:
        emj_char = emj['emoji']  # Récupérer l'émoji
        if emj_char in emoji_sentiment_dict:
            sentiments.append(emoji_sentiment_dict[emj_char])
    
    return sentiments
# Appliquer le nettoyage directement sur la colonne 'Tweet_text'
df[ 'cleaned_Tweet_text'] = df['Tweet_text'].apply(lambda x:clean_text(str(x)))

# Extraire les sentiments associés aux émojis et les associer au texte nettoyé
df[ 'cleaned_Tweet_text'] = df[ 'cleaned_Tweet_text'].apply(lambda x: ' '.join(extract_emoji_sentiment(x)) + ' ' + x)

# Appliquer le nettoyage directement sur la colonne 'description'
df[ 'cleaned_description'] = df['Description'].apply(lambda x:clean_text(str(x)))

# Extraire les sentiments associés aux émojis et les associer au texte nettoyé
df[ 'cleaned_description'] = df[ 'cleaned_description'].apply(lambda x: ' '.join(extract_emoji_sentiment(x)) + ' ' + x)

# Create an overall engagement score (based on interactions)
df['engagement_score'] = (2 * df['Retweet_count']) + df['Like_count']  # Based on retweets and likes

# Calculate the engagement rate (engagement score relative to the number of followers)
df['engagement_rate'] = df['engagement_score'] / (df['Followers_count'] + 1)  # Avoid division by zero

'''
# Calculate the virality score (engagement relative to views) - assuming you have views data, otherwise skip
# df['virality_score'] = df['engagement_score'] / (df['Views_count'] + 1)  # Use only if the 'Views_count' column exists

# Detect outliers for the likes column using IQR method
Q1 = df['Like_count'].quantile(0.25)
Q3 = df['Like_count'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Like_count'] >= (Q1 - 1.5 * IQR)) & (df['Like_count'] <= (Q3 + 1.5 * IQR))]

# Filter outliers for the followers column
Q1_followers = df['Followers_count'].quantile(0.25)
Q3_followers = df['Followers_count'].quantile(0.75)
IQR_followers = Q3_followers - Q1_followers
df = df[(df['Followers_count'] >= (Q1_followers - 1.5 * IQR_followers)) & (df['Followers_count'] <= (Q3_followers + 1.5 * IQR_followers))]

# Recalculate engagement score after cleaning
df['engagement_score'] = (2 * df['Retweet_count']) + df['Like_count']
df['engagement_rate'] = df['engagement_score'] / (df['Followers_count'] + 1)
# df['virality_score'] = df['engagement_score'] / (df['Views_count'] + 1)  # Uncomment if 'Views_count' is available

# Create a new metric for the number of hashtags
df['num_hashtags'] = df['All_Hashtags'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Generate a popularity score based on the number of likes and retweets
df['popularity_score'] = df['Like_count'] + 2 * df['Retweet_count']
'''
print(df.columns)
df.to_csv("trending_tweets_cleaned.csv", index=False)  # Sauvegarde en CSV


