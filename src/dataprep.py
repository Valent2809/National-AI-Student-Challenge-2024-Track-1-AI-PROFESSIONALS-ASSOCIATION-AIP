import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
import pickle
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Defining stopword in the global scope
stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
stemmer = SnowballStemmer("english")

def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

# This is for creating new features 
def create_new_features(df):
    df['Description'] = df['Description'].fillna("-")
    df['DescriptionWordCount'] = df['Description'].apply(lambda desc: len(desc.split()))
    df['DescriptionLength'] = df['Description'].apply(lambda desc: len(desc))
    df['DescriptionCleaned'] = df['Description'].str.lower()
    df['DescriptionCleaned'] = df['DescriptionCleaned'].apply(cleanPunc)
    df['DescriptionCleaned']= df['DescriptionCleaned'].apply(keepAlpha)
    df['DescriptionCleaned'] = df['DescriptionCleaned'].apply(removeStopWords)
    df['DescriptionCleaned'] = df['DescriptionCleaned'].apply(stemming)
    return df

# processing of the data into the correct type and extracting features needed
def preprocess_data(df):
    # Selecting relevant columns
    df_result = df.iloc[:, [1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,31,32,33,49,50,51]].copy()

    # Encoding categorical features
    for column in df_result.columns:
        if (column == 'DescriptionCleaned'):
            continue
        if df_result[column].dtype == type(object):
            le = preprocessing.LabelEncoder()
            df_result[column] = le.fit_transform(df[column])

    # Handling missing values as removing of stop words make some values to be empty
    df_result['DescriptionCleaned'] = df_result['DescriptionCleaned'].fillna('')

    return df_result

# Vectorizing text data
def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2', max_features=1000)
    vectorizer.fit(X_train['DescriptionCleaned'])

    filename = 'tfidf_vectorizer.pkl'
    saved_model_dir = 'saved_model'

    # Create the saved_model directory if it does not exist
    os.makedirs(saved_model_dir, exist_ok=True)

    # Define the filepath to save the pickled model
    model_filepath = os.path.join(saved_model_dir, filename)

    # Pickle dump the model into the saved_model directory
    with open(model_filepath, 'wb') as f:
        pickle.dump(vectorizer, f)


    X_train_tfidf = vectorizer.transform(X_train['DescriptionCleaned'])
    X_train_tfidf = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    X_test_tfidf = vectorizer.transform(X_test['DescriptionCleaned'])
    X_test_tfidf = pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    # Combine text features with other features
    X_train_other = X_train.drop('DescriptionCleaned', axis=1)
    X_test_other = X_test.drop('DescriptionCleaned', axis=1)
    X_train_final = pd.concat([X_train_other.reset_index(drop=True), X_train_tfidf.reset_index(drop=True)], axis=1)
    X_test_final = pd.concat([X_test_other.reset_index(drop=True), X_test_tfidf.reset_index(drop=True)], axis=1)

    return X_train_final, X_test_final

def main():
    # Load data (Modifiable Path)
    pets_prepared = pd.read_csv('../Technical Assessment/data/pets_prepared.csv')

    # Create new features for the df
    df = create_new_features(pets_prepared)

    # Preprocess data
    df_preprocessed = preprocess_data(df)

    # Split data into train and test sets
    X = df_preprocessed.drop('AdoptionSpeed', axis=1)
    y = df_preprocessed['AdoptionSpeed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

    #Vectorize text data
    X_train_final, X_test_final = vectorize_text(X_train, X_test)

    # Save datasets into files
    X_train_final.to_csv('X_train.csv', index=False)
    X_test_final.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

if __name__ == "__main__":
    main()
