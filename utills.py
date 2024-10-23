
import pandas as pd
import re
from nltk.corpus import stopwords
from settings import *
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

def read_data(file_path):
    return pd.read_csv(dataset_file_path)

def clean_data(data):
    #dropping unneccessary column
    data = data.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)
    # Drop duplicates
    data = data.drop_duplicates()

    # drop value with missing sentiments
    data.dropna(subset=['Text', 'Sentiment'], inplace=True)

    # update the date time format of timestamp to make them unique
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    
    #fill in missing data
    data['Likes'].fillna(data['Likes'].mean(), inplace=True)

    return data

def process_data(data):
    data['Sentiment'] = data['Sentiment'].astype('category')
    data['User'] = data['User'].astype('category')
    data['Platform'] = data['Platform'].astype('category')
    data['Country'] = data['Country'].astype('category')

    data['Retweets'] = data['Retweets'].astype('int')
    data['Likes'] = data['Likes'].astype('int')

    data['Year'] = data['Timestamp'].dt.year
    data['Month'] = data['Timestamp'].dt.month
    data['Day'] = data['Timestamp'].dt.day
    data['Hour'] = data['Timestamp'].dt.hour

    data['Cleaned_Text'] = data['Text'].apply(clean_text)
    data['Tokenized_Text'] = data['Cleaned_Text'].apply(word_tokenize)
    
    return data


def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text




