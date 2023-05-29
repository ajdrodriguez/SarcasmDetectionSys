import streamlit as st
import tensorflow as tf
import numpy as np
import nltk
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import h5py

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

voc_size=5000
max_sent_length = 20
prediction1 = 0
prediction2 = 0
stop_words = set(nltk.corpus.stopwords.words('english'))


def clean_text(text):
    # Remove HTML tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', text)
    
    # Remove special characters and punctuation
    clean_text = re.sub('[^a-zA-Z]', ' ', cleantext)
    
    # Convert to lowercase
    clean_text = clean_text.lower()
    
    return clean_text
  
def remove_stopwords(tokens):
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def get_wordnet_pos(treebank_tag):
    """Map POS tag to first character used by WordNetLemmatizer"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    tagged = nltk.pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in tagged]
    return lemmatized_tokens

def preprocess_text(text):
    cleaned_text = clean_text(text)
    tokenized_text = nltk.word_tokenize(cleaned_text)
    filtered_tokens = remove_stopwords(tokenized_text)
    lemmatized_tokens = lemmatize_tokens(filtered_tokens)
    return " ".join(lemmatized_tokens)

def load_LSTM_model():
  LSTMmodel=tf.keras.models.load_model('models/LSTMModel.hdf5')
  return LSTMmodel

def load_CNN_model():
  CNNModel=tf.keras.models.load_model('models/cnnmodel.hdf5')
  return CNNModel

def load_LR_model():
  filename = 'models/logistic_regression_model.h5'
  LRModel = LogisticRegression()
  
  with h5py.File(filename, 'r') as f:
    # Access the 'model' group
    grp = f['classifier']
    
    # Load the model's parameters
    LRModel.coef_ = grp['coef_'][:]
    LRModel.intercept_ = grp['intercept_'][:]

  return LRModel

model1=load_LSTM_model()
model2=load_CNN_model()
model3=load_LR_model()

st.write("""# Sarcasm Detection System""")
st.write("""### Developed by CPE32S4 - Group 4""")
sentence = st.text_input('Please enter a sentence.')

if sentence == '':
    st.text("No sentence found. Please enter.")
    prediction1 = 0
    prediction2 = 0
    prediction3 = 0
else:
    st.write('Inputted Sentence: ', sentence)
    predcorpus = [sentence]
    onehot_=[one_hot(words,voc_size)for words in predcorpus] 
    embedded_docs=pad_sequences(onehot_,padding='pre',maxlen=max_sent_length)
    # Get labels based on probability 1 if p>= 0.01 else 0
    prediction1 = model1.predict(embedded_docs)
    pred_labels1 = []
    if prediction1 >= 0.5:
        pred_labels1.append(1)
    else:
        pred_labels1.append(0)
    if pred_labels1[0] == 1:
        output1 = 'Sarcasm Detected'
    else:
        output1 = 'No Sarcasm Detected'
    prediction1 = (str(prediction1[0][0]*100))+'%'
    
    prediction2 = model2.predict(embedded_docs)
    pred_labels2 = []
    if prediction2 >= 0.5:
        pred_labels2.append(1)
    else:
        pred_labels2.append(0)
    if pred_labels2[0] == 1:
        output2 = 'Sarcasm Detected'
    else:
        output2 = 'No Sarcasm Detected'
    prediction2 = (str(prediction2[0][0]*100))+'%'
    
    vectorizer = TfidfVectorizer()
    LR_data = preprocess_text(sentence)
    LR_data = nltk.word_tokenize(LR_data)
    LR_data = remove_stopwords(LR_data)
    vectorizer.fit(LR_data)
    LR_data = vectorizer.transform(" ".join(LR_data))
    prediction3 = model3.predict(LR_data)
    
    for output3, label in zip(predcorpus, prediction3):
      if label == 0:
          output3 = 'Sarcasm Detected'
      else:
          output3 = 'No Sarcasm Detected'
    
    st.write("Prediction Accuracy (LSTM): ", prediction1)
    st.write("Prediction Accuracy (CNN): ", prediction2)
    string1="OUTPUT OF LSTM: "+output1
    string2="OUTPUT OF CNN: "+output2
    string3="OUTPUT OF LR: "+output3
    st.success(string1)
    st.success(string2)
    st.success(string3)
    
    
st.write("")
st.write("")
st.markdown(
"""
Members:
- Jhonlix Dave Estrañero
- Christian John Leste Jaraula
- Jefferson Luis Langbid
- Andrew John Rodriguez
"""
)
