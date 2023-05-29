import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences


voc_size=5000
max_sent_length = 20
prediction1 = 0
prediction2 = 0
  
def load_LSTM_model():
  LSTMmodel=tf.keras.models.load_model('models/LSTMModel.hdf5')
  return LSTMmodel

def load_CNN_model():
  CNNModel=tf.keras.models.load_model('models/cnnmodel.hdf5')
  return CNNModel


model1=load_LSTM_model()
model2=load_CNN_model()

st.write("""# Sarcasm Detection System""")
st.write("""### Developed by CPE32S4 - Group 4""")
sentence = st.text_input('Please enter a sentence.')

if sentence == '':
    st.text("No sentence found. Please enter.")
    prediction1 = 0
    prediction2 = 0
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
    prediction1 = "{:.2f} %".format(prediction1[0][0]*100)
    
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
    prediction2 = "{:.2f} %".format(prediction2[0][0]*100)
    
    st.write("Prediction Accuracy (LSTM): ", prediction1)
    st.write("Prediction Accuracy (CNN): ", prediction2)
    string1="OUTPUT OF LSTM: "+output1
    string2="OUTPUT OF CNN: "+output2
    st.success(string1)
    st.success(string2)
    
    
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
