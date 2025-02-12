import pandas as pd
import emoji
import re
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("twitter-posts.csv")

# Définir un seuil pour les valeurs manquantes
threshold = 800

# Supprimer les colonnes où le nombre de valeurs manquantes est supérieur au seuil
df_cleaned = df.loc[:, df.isnull().sum() <= threshold]

# Vérifier les valeurs manquantes après suppression
missing_after_deletion = df_cleaned.isnull().sum()
print("\nValeurs manquantes après suppression des colonnes avec plus de 900 valeurs manquantes:")
print(missing_after_deletion)

# Vérifier un échantillon des données après suppression des colonnes
print("\nExemple de données après suppression:")
print(df_cleaned.head())

# Étape 1: Remplir les valeurs manquantes pour la colonne 'followers' avec 0
df['followers'] = df['followers'].fillna(0)

# Étape 2: Remplir les valeurs manquantes pour 'likes' avec la valeur la plus fréquente (mode)
# Assurez-vous d'abord que la colonne 'likes' est bien numérique
df['likes'] = pd.to_numeric(df['likes'], errors='coerce')  # Convertir en numérique, en cas d'erreur remplacer par NaN
df['likes'] = df['likes'].fillna(df['likes'].mode()[0])  # Remplacer les NaN par la valeur la plus fréquente

# Étape 3: Remplir les valeurs manquantes pour 'description' avec 'Non spécifié'
df['description'] = df['description'].fillna('Non spécifié')

# Remplir les valeurs manquantes dans 'photos', 'videos', et 'hashtags' par des listes vides
df['photos'] = df['photos'].fillna('[]')  # Utiliser des chaînes vides ou des listes vides, selon le format
df['videos'] = df['videos'].fillna('[]')
df['hashtags'] = df['hashtags'].fillna('[]')

# Si nécessaire, vous pouvez également convertir ces chaînes en listes
# Exemple pour convertir une chaîne de caractères en liste
import ast
df['photos'] = df['photos'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
df['videos'] = df['videos'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
df['hashtags'] = df['hashtags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Remplir les valeurs manquantes dans 'biography' et 'external_url'
df['biography'] = df['biography'].fillna('Non spécifié')
df['external_url'] = df['external_url'].fillna('Non disponible')

# Remplir les valeurs manquantes dans 'tagged_users' par une liste vide ou une chaîne par défaut
df['tagged_users'] = df['tagged_users'].fillna('Non spécifié')

# Remplir les valeurs manquantes dans 'views' par la moyenne ou la médiane (en fonction de la distribution)
df['views'] = df['views'].fillna(df['views'].median())  # Utilisation de la médiane pour éviter les valeurs extrêmes

# Vérifier si des valeurs manquantes existent encore
missing_values = df.isnull().sum()
print("\nValeurs manquantes après remplissage:")
print(missing_values)

# Supprimer les doublons basés sur l'ID du tweet
df.drop_duplicates(subset='id', keep='last', inplace=True)

# Convert 'date_posted' to datetime with a specified format
df['date_posted'] = pd.to_datetime(df['date_posted'], format="%Y-%m-%d", errors='coerce')

# Convert numeric columns
df['followers'] = pd.to_numeric(df['followers'], errors='coerce')
df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
df['reposts'] = pd.to_numeric(df['reposts'], errors='coerce')

import re

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

# Appliquer le nettoyage directement sur la colonne 'description'
df[ 'cleaned_description'] = df['description'].apply(lambda x:clean_text(str(x)))

# Extraire les sentiments associés aux émojis et les associer au texte nettoyé
df[ 'cleaned_description'] = df[ 'cleaned_description'].apply(lambda x: ' '.join(extract_emoji_sentiment(x)) + ' ' + x)

# Vérifier un échantillon des données nettoyées
print("\nExemple de données nettoyées:")
print(df[['description', 'cleaned_description']].head())



# Extraire des caractéristiques temporelles comme l'heure de publication et le jour de la semaine
df['hour'] = df['date_posted'].dt.hour
df['day_of_week'] = df['date_posted'].dt.day_name()

# Créer un score d'engagement global (en fonction des interactions)
df['engagement_score'] = (2 * df['reposts']) + df['likes'] + df['replies'] + df['quotes'] + df['bookmarks']

# Calculer le taux d'engagement (score d'engagement par rapport au nombre de followers)
df['engagement_rate'] = df['engagement_score'] / (df['followers'] + 1)  # Eviter la division par zéro

# Calculer le score de viralité (engagement par rapport aux vues)
df['virality_score'] = df['engagement_score'] / (df['views'] + 1)


# Détection des outliers pour les colonnes de likes et followers
Q1 = df['likes'].quantile(0.25)
Q3 = df['likes'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['likes'] >= (Q1 - 1.5 * IQR)) & (df['likes'] <= (Q3 + 1.5 * IQR))]

# Filtrer les outliers pour les followers
Q1_followers = df['followers'].quantile(0.25)
Q3_followers = df['followers'].quantile(0.75)
IQR_followers = Q3_followers - Q1_followers
df = df[(df['followers'] >= (Q1_followers - 1.5 * IQR_followers)) & (df['followers'] <= (Q3_followers + 1.5 * IQR_followers))]

# Compute new engagement-based features (e.g., engagement score)
df['engagement_score'] = (2 * df['reposts']) + df['likes'] + df['replies'] + df['quotes'] + df['bookmarks']
df['engagement_rate'] = df['engagement_score'] / (df['followers'] + 1)
df['virality_score'] = df['engagement_score'] / (df['views'] + 1)


# Extract day of the week and hour of the day
df['hour'] = df['date_posted'].dt.hour
df['day_of_week'] = df['date_posted'].dt.day_name()

# Créer des métriques supplémentaires comme le nombre de hashtags
df['num_hashtags'] = df['hashtags'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

# Engendrer un score de popularité basé sur le nombre de likes, reposts et replies
df['popularity_score'] = df['likes'] + 2 * df['reposts'] + 1.5 * df['replies']
df.to_csv("twitter_cleaned.csv", index=False)  # Sauvegarde en CSV
