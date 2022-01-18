import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

import plotly.offline as pyo
import plotly.express as px

warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
pd.options.display.max_rows = 999





def tf(listados, ngram_range=(1,2)):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words="english").fit(listados)
    vector = vectorizer.transform(listados).toarray()
    return vector, vectorizer


def extract_top_n_words_per_topic(tf_idf, count, n=20):
    words = count.get_feature_names()
    tf_idf_transposed = tf_idf.T
    top_n_words={word: ( tf_idf_transposed[j][0]) for j, word in enumerate(words)}
    sort_words = sorted(top_n_words.items(), key=lambda x: x[1], reverse=True)[:n]
    return sort_words



def palabras_principales(df):
    #función para aplicar en todo el dataset, trae sólo la primer palabra
    docs_per_topic = df.groupby(['labels'], as_index = False).agg({'listado': ' '.join})
    docs_per_topic.rename(columns={'listado': 'palabras_listado'}, inplace=True)
    docs_per_topic['word']=0
    for i in range(len(docs_per_topic)):
        text=[docs_per_topic['palabras_listado'][i]]
        tf_idf, count = tf(text)
        docs_per_topic['word'][i]=extract_top_n_words_per_topic(tf_idf, count)[0][0]
    df=df.merge(docs_per_topic,on='labels', how='left')
    return df


def grafico(df):
    df=df.loc[df.labels != -1]
    fig = px.histogram(
                data_frame=df,
                x='count',
                y="word",orientation='h')
    return fig



@st.cache
def load_data():
    bert_clustering=pd.read_excel('bert_clustering_20_5_con guion.xlsx')
    print(bert_clustering.info())
    df=palabras_principales(bert_clustering)
    listados_all=pd.read_excel('listados_all.xlsx', index_col=0)
    listados_all=listados_all.drop_duplicates(keep='first')
    listados_all=listados_all.merge(df, left_on='list', right_on='listado')
    clasificacion=listados_all.groupby(['isbn13', 'word', 'labels', 'titulo', 'palabras_listado']).agg({'count': 'sum' }).sort_values(by='count', ascending=False).reset_index()
    catalogo=pd.read_csv('catalogo_actualizado.csv', usecols={'ean', 'sello', 'categoriapadre',  'categoria'},dtype={'ean':'str'}, sep=';')
    listados_originales=listados_all.groupby(['isbn13'], as_index = False).agg({'list': ','.join })
    listados_originales.rename(columns={'list': 'listados_originales'}, inplace=True)
    clasificacion=clasificacion.merge(listados_originales, on='isbn13', how='left')
    clasificacion=clasificacion.merge(catalogo, left_on='isbn13', right_on='ean')
    clasificacion.drop('ean', axis=1, inplace=True)
    return clasificacion

st.title('Análisis de clustering por temas')

clasificacion=load_data()
sellos = (clasificacion['sello'].unique())
# usuario selecciona sello
sello_seleccion = st.selectbox("Seleccionar sello", sellos)
st.subheader(f"Libros del sello seleccionado:  {sello_seleccion}")
seleccion=clasificacion.loc[clasificacion.sello==sello_seleccion]
# usuario selecciona titulo
titulos=(seleccion['titulo'].unique())
libro_seleccion = st.selectbox("Seleccionar libro", titulos)
xlibro=clasificacion.loc[clasificacion.titulo==libro_seleccion]
xlibro=xlibro.sort_values(by='count', ascending=False)
xlibro=xlibro.loc[xlibro.labels != -1]

#plot = grafico(xlibro)
st.plotly_chart(grafico(xlibro))
st.dataframe(xlibro)
#st.write(barh_variable(xlibro))

# usuario selecciona label
labels=(xlibro['labels'].unique())
labels_seleccion = st.selectbox("Seleccionar similares por label", labels)
st.subheader(f"Libros que pertenecen al mismo label:  {labels_seleccion}")
label_=clasificacion.loc[clasificacion.labels==labels_seleccion]
label_.sort_values(by=['count'], ascending=False)
st.dataframe(label_)





#isbn = st.text_area("ISBN para analizar", height=100)
#xlibro=clasificacion.loc[clasificacion.isbn13==isbn]

#clasificacion.loc[clasificacion.labels==55].sort_values(by='count', ascending=False)
