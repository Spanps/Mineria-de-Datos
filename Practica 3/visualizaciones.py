from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Cargar el dataset limpio
archivo = (
    Path(__file__).parent.parent
    / "Practica 1"
    / "Week 23 - Atlantic Hurricane Tracks 1851 to 2017 clean.csv"
)
df = pd.read_csv(archivo)

# Preparar las columnas necesarias para las graficas
df["Observation_Date"] = pd.to_datetime(
    df["Observation_Date"],
    errors="coerce"
)
df["Observation_Max_Wind"] = pd.to_numeric(
    df["Observation_Max_Wind"],
    errors="coerce"
)
df["Observation_Minimum_Pressure"] = pd.to_numeric(
    df["Observation_Minimum_Pressure"],
    errors="coerce"
)

df["Observation_Storm_Type"] = df["Observation_Storm_Type"].str.strip()

df["Año"] = df["Observation_Date"].dt.year
df["Decada"] = (df["Año"] // 10) * 10

salida = Path(__file__).with_name("graficas")
salida.mkdir(exist_ok=True)

frecuencia_tipo = df["Observation_Storm_Type"].value_counts()
umbral_otros = frecuencia_tipo.sum() * 0.02
frecuencia_pastel = frecuencia_tipo[frecuencia_tipo >= umbral_otros].copy()
resto = frecuencia_tipo[frecuencia_tipo < umbral_otros].sum()
if resto > 0:
    frecuencia_pastel["Otros"] = resto

#Diagrama de pastel.
plt.figure(figsize=(9, 7))
wedges, _, autotexts = plt.pie(
    frecuencia_pastel.values,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.8
)
plt.setp(autotexts, size=9, color="white")
plt.legend(
    wedges,
    frecuencia_pastel.index,
    title="Tipo de tormenta",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)
plt.title("Distribucion de observaciones por tipo de tormenta")
plt.tight_layout()
plt.savefig(salida / "pastel_tipos_tormenta.png", dpi=150)
plt.close()

#Histogramas.
variables_histograma = {
    "Observation_Max_Wind": "Velocidad maxima del viento",
    "Observation_Minimum_Pressure": "Presion minima"
}

for variable, titulo in variables_histograma.items():
    valores = df[variable].dropna()
    plt.figure(figsize=(9, 6))
    plt.hist(valores, bins=25, edgecolor="black")
    plt.title(f"Histograma de {titulo}")
    plt.xlabel(titulo)
    plt.ylabel("Frecuencia")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida / f"histograma_{variable}.png", dpi=150)
    plt.close()

#Diagramas de caja
variables_caja = {
    "Observation_Max_Wind": "Viento maximo",
    "Observation_Minimum_Pressure": "Presion minima"
}

for variable, titulo in variables_caja.items():
    datos_caja = [
        grupo[variable].dropna()
        for _, grupo in df.groupby("Observation_Storm_Type")
    ]
    etiquetas = sorted(df["Observation_Storm_Type"].dropna().unique())

    plt.figure(figsize=(12, 7))
    plt.boxplot(datos_caja)
    plt.xticks(range(1, len(etiquetas) + 1), etiquetas)
    plt.title(f"Diagrama de caja de {titulo} por tipo de tormenta")
    plt.xlabel("Tipo de tormenta")
    plt.ylabel(titulo)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida / f"caja_{variable}.png", dpi=150)
    plt.close()

# Diagrama de dispersion.
dispersion = df.dropna(
    subset=["Observation_Longitude", "Observation_Latitude"]
)
plt.figure(figsize=(10, 7))
plt.scatter(
    dispersion["Observation_Longitude"],
    dispersion["Observation_Latitude"],
    s=4,
    alpha=0.25
)
plt.title("Distribucion geografica de las observaciones")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(salida / "dispersion_ubicaciones.png", dpi=150)
plt.close()

#Diagrama de barras.
plt.figure(figsize=(10, 6))
frecuencia_tipo.sort_values().plot(kind="barh")
plt.title("Numero de observaciones por tipo de tormenta")
plt.xlabel("Observaciones")
plt.ylabel("Tipo de tormenta")
plt.tight_layout()
plt.savefig(salida / "barras_tipos_tormenta.png", dpi=150)
plt.close()

#Grafica de linea.
observaciones_decada = df.groupby("Decada")["Storm_ID"].count()
plt.figure(figsize=(11, 6))
plt.plot(
    observaciones_decada.index,
    observaciones_decada.values,
    marker="o"
)
plt.title("Observaciones registradas por decada")
plt.xlabel("Decada")
plt.ylabel("Numero de observaciones")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(salida / "linea_observaciones_decada.png", dpi=150)
plt.close()

print("Dataset utilizado:", archivo)
print("Graficas generadas en:", salida)