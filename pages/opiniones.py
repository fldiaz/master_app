import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict
from gensim.parsing.preprocessing import preprocess_string

from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import STOPWORDS, strip_tags, strip_numeric, remove_stopwords, strip_short, stem_text, strip_punctuation, strip_multiple_whitespaces

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words=set(stopwords.words())

custom_filters=[lambda x:x.lower(),  strip_tags, strip_numeric, remove_stopwords, strip_short, strip_punctuation, strip_multiple_whitespaces]

from PIL import Image
from io import BytesIO
def show_wordcloud(data):
    alice_mask = np.array(Image.open("otro.png"))
    wordcloud = WordCloud(background_color='white',
           stopwords=stop_words,mask=alice_mask,
           max_words=100, margin=10, max_font_size=100# chosen at random by flipping a coin; it was heads
       ).generate_from_frequencies(data)
    fig = plt.figure(figsize=(6,10))
    plt.axis('off')
    plt.imshow(wordcloud)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return fig

def app():
    #st.set_page_config(layout="wide", page_title='Análisis de opiniones', page_icon="📙" )
    st.markdown('')
    container = st.container()
    container.markdown('Análisis de opiniones: ⚡')



    reviews_df = pd.read_json('all_reviews.json', dtype={'rating': int})
    reviews_df['rating'].replace(r'^\s*$', np.nan, regex=True, inplace=True)

    #reviews_df=reviews_df.astype({'rating': int})
    titulos=(reviews_df['book_title'].unique())
    libro_seleccion = st.selectbox("Seleccionar libro", titulos)
    positive_reviews=reviews_df.loc[(reviews_df.book_title==libro_seleccion) & (reviews_df.rating>=4)].sort_values(by='num_likes', ascending=False)
    positive_reviews_list=positive_reviews['text'].to_list()


    #preprocessing
    tokenized_docs=[preprocess_string(doc,custom_filters ) for doc in positive_reviews_list]
    # Remove words that are only 3 character.
    tokenized_docs = [[token for token in doc if len(token) > 3] for doc in tokenized_docs]
    tokenized_docs = [[token for token in text if token not in stop_words] for text in tokenized_docs]

    frequency = defaultdict(int)
    for text in tokenized_docs:
        for token in text:
            frequency[token] += 1
    # Only keep words that appear more than one
    tokenized_docs = [[token for token in text if frequency[token] > 1] for text in tokenized_docs]
    tokenized_docs = [[token for token in text if frequency[token] < 200] for text in tokenized_docs]
    fdist = FreqDist()
    for text in tokenized_docs:
        for token in text:
            fdist[token.lower()] += 1


    #-------------------------------
    st.markdown('### Principales frases')
    st.markdown('### 1. Adverbio-Adjetivo')
    from keyphrase_vectorizers import KeyphraseCountVectorizer
    # Init default vectorizer.
    vectorizer = KeyphraseCountVectorizer(pos_pattern='<RB.*>*<JJ.*>+')
    vectorizer.fit(positive_reviews_list)
    keyphrases = vectorizer.get_feature_names_out()
    fdist = FreqDist()
    for text in keyphrases:
      #print(text)
      fdist[text.lower()] += 1

    st.pyplot(show_wordcloud(fdist))

    all_reviews= ''.join([reviews for reviews in positive_reviews_list])
    #sKeybert: prioridad
    st.markdown('### Principales, Adverbio-Adjetivo')
    from keybert import KeyBERT
    kw_model = KeyBERT()
    keyphrases_bert=kw_model.extract_keywords(docs=all_reviews, vectorizer=KeyphraseCountVectorizer(pos_pattern='<RB.*>*<JJ.*>+'), top_n=100)
    st.dataframe(keyphrases_bert[0:20])

    st.markdown('### ----------------------')

    st.markdown('### 2. Adjetivo-Sustantivo')
    from keyphrase_vectorizers import KeyphraseCountVectorizer
    # Init default vectorizer.
    vectorizer = KeyphraseCountVectorizer(pos_pattern='<J.*>*<N.*>+')
    vectorizer.fit(positive_reviews_list)
    keyphrases = vectorizer.get_feature_names_out()
    fdist = FreqDist()
    for text in keyphrases:
      #print(text)
      fdist[text.lower()] += 1

    st.pyplot(show_wordcloud(fdist))

    all_reviews= ''.join([reviews for reviews in positive_reviews_list])
    #sKeybert: prioridad
    st.markdown('### Principales, Adverbio-Adjetivo')
    from keybert import KeyBERT
    kw_model = KeyBERT()
    keyphrases_bert=kw_model.extract_keywords(docs=all_reviews, vectorizer=KeyphraseCountVectorizer(pos_pattern='<RB.*>*<JJ.*>+'), top_n=100)
    st.dataframe(keyphrases_bert[0:20])
