import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelos
modelo = joblib.load("modelo_entrenado.pkl")
escalador = joblib.load("escalador.pkl")
codificador = joblib.load("codificador.pkl")

st.title("Predicción de precio de vivienda")
st.write("Introduce las características del inmueble para obtener una estimación del precio de venta.")

# Formulario
surface = st.number_input("Superficie (m²)", min_value=10, max_value=1000, value=80)
rooms = st.slider("Número de habitaciones", 1, 10, 3)
bathrooms = st.slider("Número de baños", 1, 5, 2)
air = st.checkbox("Aire acondicionado")
elevator = st.checkbox("Ascensor")
pool = st.checkbox("Piscina")
terrace = st.checkbox("Terraza")
parking = st.checkbox("Plaza de garaje")

orientaciones = ["Norte", "Sur", "Este", "Oeste"]
orientacion = st.selectbox("Orientación", orientaciones)

neighborhood = st.selectbox("Barrio", ["Ciudad Universitaria", "Valdemarín", "El Viso", "Nueva España", "Castilla"])
district = st.selectbox("Distrito", ["Moncloa-Aravaca", "Chamartín"])

# Codificar orientación
orient_dict = {
    "is_orientation_north": 1 if orientacion == "Norte" else 0,
    "is_orientation_south": 1 if orientacion == "Sur" else 0,
    "is_orientation_east": 1 if orientacion == "Este" else 0,
    "is_orientation_west": 1 if orientacion == "Oeste" else 0,
}

# Crear input del usuario como DataFrame
input_data = pd.DataFrame([{
    "Surface": surface,
    "Rooms": rooms,
    "Bathrooms": bathrooms,
    "Air_Conditioner": int(air),
    "Elevator": int(elevator),
    "Swimming_Pool": int(pool),
    "Terrace": int(terrace),
    "Parking": int(parking),
    **orient_dict,
    "Price_per_m2": 0  # Campo dummy si lo necesitas
}])

# Normalizar las variables numéricas
input_scaled = escalador.transform(input_data)

# Codificar barrio y distrito
categoricas = pd.DataFrame(codificador.transform([[neighborhood, district]]), columns=codificador.get_feature_names_out())

# Unir todo
X_final = pd.concat([pd.DataFrame(input_scaled, columns=input_data.columns), categoricas], axis=1)

# Hacer predicción
prediccion = modelo.predict(X_final)[0]

# Mostrar resultado
st.subheader("Precio estimado de la vivienda:")
st.success(f"{int(prediccion):,} €".replace(",", "."))  # Formato español
