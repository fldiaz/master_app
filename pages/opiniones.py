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
from sentence_transformers import SentenceTransformer, util
from keyphrase_vectorizers import KeyphraseTfidfVectorizer


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
    reviews_df['genero']= ''
    reviews_df.loc[reviews_df.book_title=='El último deseo', 'genero']='fantasy'
    reviews_df.loc[reviews_df.book_title=='El misterio del lobo blanco', 'genero']='fantasy'
    reviews_df.loc[reviews_df.book_title=='Conan (#1) El cimerio', 'genero']='fantasy'
    reviews_df.loc[reviews_df.book_title=="Miss Peregrine's Home for Peculiar Children", 'genero']='fantasy2'
    reviews_df.loc[reviews_df.book_title=="Harry Potter and the Sorcerer's Stone", 'genero']='fantasy2'
    reviews_df.loc[reviews_df.book_title=="The Golden Compass", 'genero']='fantasy2'
    reviews_df.loc[reviews_df.book_title=="La música del adiós", 'genero']='detectives'
    reviews_df.loc[reviews_df.book_title=="La habitación cerrada", 'genero']='detectives'
    reviews_df.loc[reviews_df.book_title=="Por el camino difícil", 'genero']='detectives'
    reviews_df.loc[reviews_df.book_title=="Rosas muertas", 'genero']='detectives'


    #reviews_df=reviews_df.astype({'rating': int})
    generos=(reviews_df['genero'].unique())
    genero_seleccion = st.selectbox("Seleccionar libro", generos)

    st.markdown(reviews_df.loc[reviews_df.genero==genero_seleccion]['book_title'].unique())
    opiniones=reviews_df.loc[(reviews_df.genero ==genero_seleccion)]
    opiniones=reviews_df.loc[(reviews_df.rating >= 4)]
    positive_reviews_list=opiniones.text.to_list()



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
    keyphrases_bert=kw_model.extract_keywords(docs=all_reviews, vectorizer=KeyphraseCountVectorizer(pos_pattern='<RB.*>*<JJ.*>+'),use_maxsum=True, diversity=0.7, nr_candidates=200, top_n=20)
    st.dataframe(keyphrases_bert)



    st.markdown('### ----------------------')

    st.markdown('### 2. Adjetivo-Sustantivo')
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
    st.markdown('### Principales, Adjetivo-Sustantivo')
    from keybert import KeyBERT
    kw_model = KeyBERT()
    keyphrases_bert=kw_model.extract_keywords(docs=all_reviews, vectorizer=KeyphraseCountVectorizer(pos_pattern='<J.*>*<N.*>+'), use_maxsum=True, diversity=0.7)
    st.dataframe(keyphrases_bert[0:20])

    st.markdown('### ----------------------')
    st.markdown('### Comparación con otros libros, Adverbio-Adjetivo')

    matriz=pd.read_csv('cosine_matriz.csv', index_col=0)
    print(matriz.head())
    def analisis(matriz, titulo):
        orden = matriz.loc[matriz.título==titulo].set_index('título').T.sort_values(by=titulo, ascending=False)
        return orden

    #st.dataframe(analisis(matriz, libro_seleccion))
