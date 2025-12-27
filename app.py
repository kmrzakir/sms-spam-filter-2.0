import streamlit as st
import sklearn
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('stopwords')
ps = PorterStemmer()


model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.title("Sms spam filter classifier")

input_sms = st.text_area("Enter an sms")


# function for transforimg text
def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y =[]
  # removes the special characters
  for i in text:
    if i.isalnum():
      y.append(i)

  # remove stopwords
  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words("english") and i not in string.punctuation:
      y.append(i)

  # now we will do Stemming
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))


  text = y
  return " ".join(text)



predection = ""


if st.button("click"):
  if not input_sms or len(input_sms.strip()) < 5:
    st.warning("⚠️ Please enter at least 5 characters")
  else:
    # 1 preprocess text
    transformed_text = transform_text(input_sms)

    # 2 vectorize text
    vectorized_text = vectorizer.transform([transformed_text])

    # 3 predict output
    predection = model.predict(vectorized_text)[0]

# 4 display result
if predection == 1:
  st.error("🚨 Spam Message")
elif predection == 0:
  st.success("✅ Not Spam")
