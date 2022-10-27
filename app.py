import streamlit as st

# Custom imports
from menu import MultiPage
from pages import funciones_app, opiniones # import your pages here

# Create an instance of the app
app = MultiPage()

# Add all your applications (pages) here
app.add_page("📙 Análisis de los temas de los libros, según Goodreads", funciones_app.app)
app.add_page("Análisis de opiniones", opiniones.app)

# The main app
app.run()
