from pathlib import Path

import pandas as pd

# Cargar el dataset.
archivo = (
    Path(__file__).parent.parent
    / "Practica 1"
    / "Week 23 - Atlantic Hurricane Tracks 1851 to 2017 clean.csv"
)

df = pd.read_csv(archivo)

# Mostrar información general.
print("Dataset cargado desde:", archivo)

print("Número de registros:", len(df))
print("Número de columnas:", len(df.columns))
print("Número de tormentas:", df["Storm_ID"].nunique())

variables = [
    "Observation_Latitude",
    "Observation_Longitude",
    "Observation_Max_Wind",
    "Observation_Minimum_Pressure"
]

# Convertir fechas y variables numericas al formato adecuado.
df["Observation_Date"] = pd.to_datetime(
    df["Observation_Date"],
    format="%Y-%m-%d",
    errors="coerce"
)

df["Observation_Max_Wind"] = pd.to_numeric(df["Observation_Max_Wind"], errors="coerce")
df["Observation_Minimum_Pressure"] = pd.to_numeric(
    df["Observation_Minimum_Pressure"], errors="coerce"
)

print("\n Estadisticas descriptivas de las variables numéricas:")

estadistica = df[variables].describe()

print(estadistica)

# Calcular medidas de tendencia central
print("\n Medidas de tendencia central:")

for variable in variables:

    print("\n", variable)

    print("Media:",
          df[variable].mean())

    print("Mediana:",
          df[variable].median())

    modas = df[variable].mode()
    if len(modas) > 1:
        print(f"Moda: hay {len(modas)} valores empatados -> {modas.tolist()}")
    else:
        print("Moda:", modas.iloc[0])

# Calcular medidas de dispersion
print("\n Medidas de dispersión:")

for variable in variables:

    print("\n", variable)

    print("Varianza:",
          df[variable].var())

    print("Desviación estándar:",
          df[variable].std())

    print("Mínimo:",
          df[variable].min())

    print("Máximo:",
          df[variable].max())

    print("Rango:",
          df[variable].max() -
          df[variable].min())

    print("Q1:",
          df[variable].quantile(0.25))

    print("Q2 (Mediana):",
          df[variable].quantile(0.50))

    print("Q3:",
          df[variable].quantile(0.75))


# Calcular frecuencias y porcentajes por tipo de tormenta
print("\n Frecuencia por tipo de tormenta (por observación):")

frecuencia_tipo = (
    df["Observation_Storm_Type"]
    .value_counts()
)

print(frecuencia_tipo)

print("\nPorcentaje por tipo de tormenta (por observación):")

porcentaje_tipo = (
    df["Observation_Storm_Type"]
    .value_counts(normalize=True) * 100
)

print(porcentaje_tipo.round(2))

# Calcular metricas agrupadas por tipo de tormenta
print("\nMetricas agrupadas por tipo de tormenta (por observación):")

grupo_tipo = df.groupby(
    "Observation_Storm_Type"
).agg(

    Observaciones=(
        "Storm_ID",
        "count"
    ),

    Tormentas_que_pasaron_por_este_tipo=(
        "Storm_ID",
        "nunique"
    ),

    Viento_Promedio=(
        "Observation_Max_Wind",
        "mean"
    ),

    Viento_Mediana=(
        "Observation_Max_Wind",
        "median"
    ),

    Viento_Maximo=(
        "Observation_Max_Wind",
        "max"
    ),

    Presion_Promedio=(
        "Observation_Minimum_Pressure",
        "mean"
    ),

    Presion_Minima=(
        "Observation_Minimum_Pressure",
        "min"
    )
)

print(grupo_tipo.round(2))

# Identificar el tipo mas intenso alcanzado por cada tormenta.
orden_intensidad = {
    "DB": 0, "WV": 0, "LO": 1, "SD": 2, "TD": 2,
    "SS": 3, "TS": 3, "ET": 4, "EX": 4, "HU": 5,
}
df["_rango_intensidad"] = df["Observation_Storm_Type"].map(orden_intensidad).fillna(-1)

tipo_mas_intenso_por_tormenta = (
    df.sort_values("_rango_intensidad")
    .groupby("Storm_ID")["Observation_Storm_Type"]
    .last()
)

print("\nClasificación de tormentas por su tipo más intenso:")
print(tipo_mas_intenso_por_tormenta.value_counts())

df = df.drop(columns="_rango_intensidad")


inicio_tormenta = df.groupby("Storm_ID")["Observation_Date"].transform("min")
df["Año"] = inicio_tormenta.dt.year
df["Decada"] = (df["Año"] // 10) * 10

# Calcular metricas agrupadas por decada.
print("\nMetricas agrupadas por década:")

grupo_decada = df.groupby("Decada").agg(

    Observaciones=(
        "Storm_ID",
        "count"
    ),

    Tormentas=(
        "Storm_ID",
        "nunique"
    ),

    Viento_Promedio=(
        "Observation_Max_Wind",
        "mean"
    ),

    Viento_Maximo=(
        "Observation_Max_Wind",
        "max"
    ),

    Presion_Promedio=(
        "Observation_Minimum_Pressure",
        "mean"
    ),

    Presion_Minima=(
        "Observation_Minimum_Pressure",
        "min"
    )
)

print(grupo_decada.round(2))

# Resumen
print("\nResumen de métricas generales:")

print("Total de observaciones:",
      len(df))

print("Total de tormentas:",
      df["Storm_ID"].nunique())

print(
    "Viento promedio:",
    round(df["Observation_Max_Wind"].mean(), 2)
)

print(
    "Viento máximo:",
    df["Observation_Max_Wind"].max()
)

print(
    "Presión promedio:",
    round(
        df["Observation_Minimum_Pressure"].mean(),
        2
    )
)

print(
    "Presión mínima:",
    df["Observation_Minimum_Pressure"].min()
)

print(
    "Tipo de tormenta más frecuente (por observación):",
    df["Observation_Storm_Type"]
    .value_counts()
    .idxmax()
)

print(
    "Tipo de tormenta más frecuente (tormentas que llegaron a ese nivel, sin doble conteo):",
    tipo_mas_intenso_por_tormenta.value_counts().idxmax()
)