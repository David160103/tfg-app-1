import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelos
modelo = joblib.load("modelo_xgboost.pkl")
escalador = joblib.load("scaler.pkl")
codificador = joblib.load("encoder.pkl")

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

# Datos introducidos por el usuario
input_data = {
    "Surface": surface,
    "Rooms": rooms,
    "Bathrooms": bathrooms,
    "Air_Conditioner": air_conditioner,
    "Elevator": elevator,
    "Swimming_Pool": swimming_pool,
    "Terrace": terrace,
    "Parking": parking,
    "is_orientation_north": 1 if orientation == "Norte" else 0,
    "is_orientation_west": 1 if orientation == "Oeste" else 0,
    "is_orientation_south": 1 if orientation == "Sur" else 0,
    "is_orientation_east": 1 if orientation == "Este" else 0,
    "Price_per_m2": price_per_m2
}

input_data = pd.DataFrame([input_data])

# 👉 Asegurarse de que el orden de las columnas es el mismo que en el entrenamiento
columnas_esperadas = ['Surface', 'Rooms', 'Bathrooms', 'Air_Conditioner', 'Elevator',
                      'Swimming_Pool', 'Terrace', 'Parking', 'is_orientation_north',
                      'is_orientation_west', 'is_orientation_south', 'is_orientation_east', 'Price_per_m2']
input_data = input_data[columnas_esperadas]  # Reordenar y evitar errores de validación

# Aplicar el escalador
input_scaled = escalador.transform(input_data)


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
