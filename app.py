import streamlit as st
import joblib
import pandas as pd

# Cargar modelo y transformadores
modelo = joblib.load("modelo_xgboost.pkl")
escalador = joblib.load("scaler.pkl")
codificador = joblib.load("encoder.pkl")

st.title("Predicción del Precio de Viviendas en Madrid")
st.write("Introduce las características de la vivienda para estimar su precio de venta")

# Entradas de usuario
surface = st.number_input("Superficie (m2)", min_value=10, max_value=1000, step=1)
rooms = st.slider("Nº de habitaciones", 1, 10, 3)
bathrooms = st.slider("Nº de baños", 1, 5, 2)
air_conditioner = st.checkbox("Aire acondicionado")
elevator = st.checkbox("Ascensor")
swimming_pool = st.checkbox("Piscina")
terrace = st.checkbox("Terraza")
parking = st.checkbox("Plaza de garaje")
orientation = st.selectbox("Orientación", ["Norte", "Sur", "Este", "Oeste"])
price_per_m2 = st.number_input("Precio por m²", min_value=1000, max_value=20000, step=100)
district = st.selectbox("Distrito", ["District 5: Chamartín", "District 11: Moncloa"])
neighborhood = st.selectbox("Barrio", [
    "El Viso", "Castilla", "Ciudad Jardín", "Nueva España", "Hispanoamérica", "Prosperidad", "Chamartín",
    "Aravaca", "Argüelles", "Casa de Campo", "Ciudad Universitaria", "El Plantío", "Valdemarín", "Valdezarza", "Moncloa"])

# Codificar orientación
is_orientation_north = int(orientation == "Norte")
is_orientation_south = int(orientation == "Sur")
is_orientation_east = int(orientation == "Este")
is_orientation_west = int(orientation == "Oeste")

# Crear DataFrame con datos del usuario
input_data = pd.DataFrame([{ 
    "Surface": surface,
    "Rooms": rooms,
    "Bathrooms": bathrooms,
    "Air_Conditioner": int(air_conditioner),
    "Elevator": int(elevator),
    "Swimming_Pool": int(swimming_pool),
    "Terrace": int(terrace),
    "Parking": int(parking),
    "is_orientation_north": is_orientation_north,
    "is_orientation_west": is_orientation_west,
    "is_orientation_south": is_orientation_south,
    "is_orientation_east": is_orientation_east,
    "Price_per_m2": price_per_m2,
    "neighborhood": neighborhood,
    "district": district
}])

# Separar columnas categóricas y numéricas
categorical = input_data[["neighborhood", "district"]]
numerical = input_data.drop(columns=["neighborhood", "district"])

# Codificar categóricas y escalar numéricas
categorical_encoded = pd.DataFrame(codificador.transform(categorical),
                                    columns=codificador.get_feature_names_out(["neighborhood", "district"]))
numerical_scaled = pd.DataFrame(escalador.transform(numerical), columns=numerical.columns)

# Unir todo el input transformado
input_scaled = pd.concat([numerical_scaled, categorical_encoded], axis=1)

# Predicción
if st.button("Predecir precio"):
    prediccion = modelo.predict(input_scaled)[0]
    st.success(f"El precio estimado de la vivienda es: {prediccion:,.0f} €")
