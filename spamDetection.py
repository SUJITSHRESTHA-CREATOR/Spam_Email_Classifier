#importing libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data = pd.read_csv(r"C:\Users\ECs\Desktop\AI\spam-email-classifier\dataset\spam.csv")
# print(data.head())  
# print(data.shape)

data.drop_duplicates(inplace = True)
# print(data.shape)

data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
# print(data.head())


#splitting task

msg = data['Message']
cat = data['Category']

(msg_train, msg_test, cat_train, cat_test) = train_test_split(msg, cat, test_size = 0.2)


#vectorization or cleaning or removal of stopwords

cv = CountVectorizer(stop_words = 'english')
features = cv.fit_transform(msg_train)

#model creation

model = MultinomialNB()
model.fit(features, cat_train)

#testing the model

features_test = cv.transform(msg_test)
#print(model.score(features_test, cat_test))

#data prediction

def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

st.header('Spam Detection')

input_msg = st.text_input('Enter your Message')

if st.button('Validate'):
    output = predict(input_msg)
    if output[0] == 'Spam':
        st.error("🚫 This message is Spam.")
    else:
        st.success("✅ This message is Not Spam.")




