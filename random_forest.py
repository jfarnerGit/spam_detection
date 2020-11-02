import string
import sklearn
import pandas as pd
import random
import nltk
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
import collections
from nltk.metrics import (precision, recall, f_measure)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#format csv files - search through and replace "ham ," with "ham<>"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score

text = open("spam2.csv", "r", encoding = "ISO-8859-1")

# clean the raw csv based on characters
#creating custom delimiter because there are commas in text messages
text = ''.join([i for i in text]).replace("ham,", "ham<>")
text = ''.join([i for i in text]).replace("spam,", "spam<>")
text = ''.join([i for i in text]).replace("v1,", "v1<>")
text = ''.join([i for i in text]).replace('"', '')
text = ''.join([i for i in text]).replace(',,,', '')
#accounting for some entries where there is a '.'
text = ''.join([i for i in text]).replace("spam.", "spam<>")
x = open("delspam2.csv","w")
x.writelines(text)
x.close
#print(text)
#reading
df = pd.read_csv('delspam2.csv', delimiter='<>', engine = 'python')
df.columns
df = df.sort_index()

hams = df.loc[df['v1'] == 'ham']
spams = df.loc[df['v1']=='spam']
#print(hams)
#print(spams)

labeledData = []
# handle null entries
df["v2"].fillna("No text", inplace = True)


AllWords = []
text = ""
for i in hams['v2']:
    labeledData.append((i, 'ham'))
    text+= str(i) ;
for i in spams['v2']:
    labeledData.append((i, 'spam'))
    text+= str(i) ;
random.shuffle(labeledData)
porter = nltk.PorterStemmer()
def clean_text(text):
    text = re.sub(',,,', ',', text)
    text = re.sub('"', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word

    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in stripped if word not in stop_words]
    return ' '.join(porter.stem(word) for word in words)

df.dropna(inplace=True,axis=1)

def preprocessing(current):
    tokens = nltk.word_tokenize(current)
    tokens = nltk.pos_tag(tokens)
    cleaned_row = [' '.join(t) for t in tokens]
    cleaned_row = str(cleaned_row)
    return cleaned_row
print('Currently pre-processing and cleaning text data...')
print('')
for index, row in df.iterrows():
    #print(row['v2'])
    current = row['v2']
    row['v2'] = preprocessing(current)

X = df['v2']
Y = df['v1']
print('')
print('Preparing cleaned data using TF-IDF for features...')
tf = TfidfVectorizer(min_df=0.5, stop_words='english')
# split into test and training
trainingMessages, testMessages, trainingLabels, testLabels = train_test_split(X, Y, test_size=0.2, random_state=4)
trainingMessagesTfIdf = tf.fit_transform(trainingMessages)
#print(trainingMessagesTfIdf.toarray())
#print(trainingMessagesTfIdf)
testMessagesCV = tf.fit_transform(testMessages)
# need to open the files and read at every labeled review
#featuresets = [(document_features(d), c) for (d,c) in labeledReviews]
# loop and append to featuresets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

target_labels = ['ham', 'spam']
print('')
print('Building Random Forest Model:')

from sklearn.linear_model import LogisticRegression

lg = RandomForestClassifier()
lg.fit(trainingMessagesTfIdf.todense(),trainingLabels)
lgpred = lg.predict(testMessagesCV.todense())
print(classification_report(testLabels, lgpred, target_names=target_labels))
print('accuracy Score: ', accuracy_score(testLabels,lgpred))
