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
import s3fs
import os
import boto3
import time

import plotly.graph_objs as go
#grafico de barras
def barra(df, y, title):
    x= df.rating#['Administración', 'Gustavo Esquivel', 'Ismael Rimoldi','Juan Carlos Ferreira']
    y = df[y]
    fig_m_prog = go.Figure([go.Bar(x=x, y=y, text=y, textposition='auto',orientation='h', marker_color='rgb(26, 118, 255)')])
    fig_m_prog.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                                 font={'color': "#111111", 'size': 14}, title=title)
    fig_m_prog.update_yaxes(title='y', visible=False, showticklabels=False)
    return fig_m_prog


def grafico(df):
    df=df.loc[df.labels != -1]
    fig = px.histogram(
                data_frame=df,
                x='count',
                y="word",orientation='h')
    return fig

#grafico de barras dimension
def barra_dimension(df, y, title):
    df=df.groupby(y).sum()
    df['proporcion']=(df['count'] / df['count'].sum())*100
    df.sort_values(by='proporcion', inplace=True)
    x= df.proporcion#['Administración', 'Gustavo Esquivel', 'Ismael Rimoldi','Juan Carlos Ferreira']
    y = df.index
    fig_m_prog = go.Figure([go.Bar(x=x, y=y, text=np.round(x), textposition='auto',orientation='h', marker_color='rgb(26, 118, 255)',texttemplate='%{text}%')])
    fig_m_prog.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                                 font={'color': "#111111", 'size': 14}, title=title)
    fig_m_prog.update_yaxes( visible=True, showticklabels=True)
    return fig_m_prog

# Create connection object.
# `anon=False` means not anonymous, i.e. it uses access keys to pull data.
fs = s3fs.S3FileSystem(anon=False)

@st.cache
def load_data(filename):
    with fs.open(filename, mode="rb") as f:
        df=pd.read_excel(f, index_col=0)
        return df

@st.experimental_memo(ttl=600)
def load_csv(file):
    with fs.open(file, mode="rb") as f:
        df=pd.read_csv(f, index_col=0)
        return df

#-----------------------------------------------------------------------
#APP
st.set_page_config(layout="wide", page_title='Análisis clustering')
st.markdown('')
container = st.container()
container.markdown('## 📙 Análisis de los temas de los libros, según Goodreads')

st.set_page_config(layout="wide", page_title='Análisis clustering', page_icon="📙" )
st.markdown('')
container = st.container()
container.markdown('## 📙 Análisis de los temas de los libros, según Goodreads')


st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
### SEASON RANGE ###
st.sidebar.markdown("**Análisis de opiniones:** ⚡")

#-------------------------------------------------------------------------------
#DATOS
file1='s3://datos-riverside/clasificacion_libros_62022.xlsx'
clasificacion=load_data(file1)
print(clasificacion.columns)
# usuario selecciona titulo
titulos=(clasificacion['titulo'].unique())
libro_seleccion = st.selectbox("Seleccionar libro", titulos)
xlibro=clasificacion.loc[clasificacion.titulo==libro_seleccion]
xlibro=xlibro.sort_values(by='count', ascending=False)
trace1 = go.Bar(
y=xlibro['rating'], # NOC stands for National Olympic Committee
x=xlibro['publico_ojetivo'],
name = 'Público_Ojetivo',text=np.round(xlibro['rating']),
marker=dict(color='#FFD700') # set the marker color to gold
)
trace2 = go.Bar(
y=xlibro['rating'],
x=xlibro['popularidad'],
name='popularidad',text=np.round(xlibro['rating']),
marker=dict(color='#9EA0A1') # set the marker color to silver
)
trace3 = go.Bar(
y=xlibro['rating'],
x=xlibro['tema_literatura'],
name='tema_literatura',text=np.round(xlibro['rating']),
marker_color='crimson' # set the marker color to bronze
)
trace4 = go.Bar(
y=xlibro['rating'], # NOC stands for National Olympic Committee
x=xlibro['como_esta_escrito'],
name = 'como_esta_escrito',text=np.round(xlibro['rating']),
#marker=dict(color='#FFD700') # set the marker color to gold
)
trace5 = go.Bar(
y=xlibro['rating'],
x=xlibro['donde_esta_escrito'],
name='donde_esta_escrito',text=np.round(xlibro['rating']),
marker_color='rgb(26, 118, 255)' # set the marker color to silver
)
trace6 = go.Bar(
y=xlibro['rating'],
x=xlibro['cuando_esta_escrito'],
name='cuando_esta_escrito',text=np.round(xlibro['rating']),
marker=dict(color='#CD7F32') # set the marker color to bronze
)
data = [trace1, trace2, trace3 ,trace4, trace5, trace6]
layout = go.Layout( height=800, width=1000,
title='Participación de los temas sobre el total de votaciones',paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",  yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        domain=[0, 0.85]
    ),xaxis_tickangle=-90)
fig = go.Figure(data=data, layout=layout)
fig=fig.update_traces(texttemplate='%{text}%')
fig=fig.update_layout(uniformtext_minsize=6, uniformtext_mode='hide')

st.plotly_chart(fig)

col1, col2= st.columns([4, 4])
with col1:
    st.plotly_chart(barra_dimension(xlibro, 'popularidad', 'Popularidad'))
with col2:
    st.plotly_chart(barra_dimension(xlibro, 'tema_literatura', 'Tema del Libro'))
col1, col2= st.columns((2))
with col1:
    st.plotly_chart(barra_dimension(xlibro, 'cuando_esta_escrito', 'Cuándo está escrito'))
with col2:
    st.plotly_chart(barra_dimension(xlibro, 'donde_esta_escrito', 'Dónde está escrito'))
col1, col2= st.columns((2))
with col1:
    st.plotly_chart(barra_dimension(xlibro, 'como_esta_escrito', 'Cómo está escrito'))
with col2:
    st.plotly_chart(barra_dimension(xlibro, 'publico_ojetivo', 'Público Objetivo'))

#st.write(barh_variable(xlibro))
xlibro.fillna('-', inplace=True)
st.dataframe(xlibro)
#---------------------------------------------------
st.markdown('--------------------------------------')
st.title('Libros similares')
libros_similares=pd.read_excel('similar_books_62022.xlsx')
print(xlibro.iloc[0,2])
isbn=xlibro.iloc[0,2]
similares_eleccion=libros_similares.loc[libros_similares.isbn14==isbn]
st.dataframe(similares_eleccion[[ 'similar_books_titulo', 'similar_books_isbn13', 'similar_books_link']])

#---------------------------------------------------
st.markdown('--------------------------------------')
st.title('Análisis palabras y libros de un mismo cluster')
file2='s3://datos-riverside/clusters.xlsx'
file3='s3://datos-riverside/catalogo_actualizado.csv'
file4='s3://datos-riverside/listados_all.xlsx'
clusters=pd.read_excel(file2)
clusters=load_data(file2)
#seleccion por clustering
catalogo=load_csv(file3)
catalogo=catalogo.astype({'ean':'str'})
listados_all=load_data(file4)
clusters=listados_all.merge(clusters, left_on='list', right_on='listado')
clusters=clusters.merge(catalogo, left_on='isbn13', right_on='ean')
labels=(clusters['predicted_labels'].unique())
labels_seleccion = st.selectbox("Seleccionar Cluster", labels)
xlabels=clusters.loc[clusters.predicted_labels==labels_seleccion]
xlabels=xlabels.sort_values(by='listado', ascending=False)
xlabels.fillna('-', inplace=True)
st.dataframe(xlabels[['listado', 'titulo_x', 'isbn13', 'count', 'popularidad', 'publico_ojetivo', 'tema_literatura',
       'como_esta_escrito', 'donde_esta_escrito', 'cuando_esta_escrito']])
