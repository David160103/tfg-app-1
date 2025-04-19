import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar los modelos y transformadores
modelo = joblib.load("modelo_entrenado.pkl")
escalador = joblib.load("escalador.pkl")
codificador = joblib.load("codificador.pkl")
columnas_numericas = joblib.load("columnas_numericas.pkl")
columnas_codificadas = joblib.load("columnas_categoricas.pkl")

st.set_page_config(page_title="Predicción de Precios de Vivienda", layout="centered")
st.title("🧠 Valoriza AI - Estimador de precios inmobiliarios")
st.markdown("Introduce los datos de tu vivienda para estimar su precio de venta en Madrid.")

# Campos de entrada
neighborhood = st.selectbox("Barrio", [
    "El Plantío", "Aravaca", "Valdemarín", "Casa de Campo", "Ciudad Universitaria",
    "Argüelles", "Valdezarza", "Castilla", "Ciudad Jardín", "El Viso", "Hispanoamérica",
    "Nueva España", "Prosperidad", "Chamartín"])

district = st.selectbox("Distrito", ["Moncloa-Aravaca", "Chamartín"])

surface = st.number_input("Superficie (m²)", min_value=10, max_value=1000, value=100)
rooms = st.number_input("Número de habitaciones", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Número de baños", min_value=1, max_value=5, value=2)

air_conditioner = st.selectbox("Aire acondicionado", ["Sí", "No"])
elevator = st.selectbox("Ascensor", ["Sí", "No"])
swimming_pool = st.selectbox("Piscina", ["Sí", "No"])
terrace = st.selectbox("Terraza", ["Sí", "No"])
parking = st.selectbox("Plaza de garaje", ["Sí", "No"])

orientation = st.selectbox("Orientación", ["Norte", "Sur", "Este", "Oeste"])

price_per_m2 = st.number_input("Precio estimado por m² (opcional, puedes dejarlo en blanco si no lo sabes)", value=0)

# Botón para predecir
if st.button("Estimar precio"):
    # Codificar orientación
    orientaciones = {
        "is_orientation_north": 1 if orientation == "Norte" else 0,
        "is_orientation_west": 1 if orientation == "Oeste" else 0,
        "is_orientation_south": 1 if orientation == "Sur" else 0,
        "is_orientation_east": 1 if orientation == "Este" else 0,
    }

    # Datos numéricos
    datos_numericos = pd.DataFrame([{
        "Surface": surface,
        "Rooms": rooms,
        "Bathrooms": bathrooms,
        "Air_Conditioner": 1 if air_conditioner == "Sí" else 0,
        "Elevator": 1 if elevator == "Sí" else 0,
        "Swimming_Pool": 1 if swimming_pool == "Sí" else 0,
        "Terrace": 1 if terrace == "Sí" else 0,
        "Parking": 1 if parking == "Sí" else 0,
        **orientaciones,
    }])

    # Normalizar datos numéricos
    datos_numericos = pd.DataFrame(escalador.transform(datos_numericos), columns=columnas_numericas)

    # Codificar barrio y distrito
    datos_categoricos = pd.DataFrame([{"neighborhood": neighborhood, "district": district}])
    datos_codificados = codificador.transform(datos_categoricos)
    datos_codificados = pd.DataFrame(datos_codificados, columns=columnas_codificadas)

    # Unir ambas partes
    input_final = pd.concat([datos_numericos, datos_codificados], axis=1)

    # Predecir
    prediccion = modelo.predict(input_final)[0]
    st.success(f"💰 El precio estimado de venta es: {prediccion:,.0f} €")
