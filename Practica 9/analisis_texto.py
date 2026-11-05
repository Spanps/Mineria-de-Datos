from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA = (
    RAIZ_PROYECTO
    / "Practica 1"
    / "Week 23 - Atlantic Hurricane Tracks 1851 to 2017 clean.csv"
)
CARPETA = Path(__file__).resolve().parent / "salidas"
CARPETA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RUTA)

df["Year"] = df["Storm_ID"].str[-4:].astype(int)
df["Nombre"] = df["Storm_Name"].str.strip().str.title()

tormentas = df.drop_duplicates("Storm_ID").copy()

con_nombre = tormentas[tormentas["Nombre"] != "Unnamed"]
sin_nombre = tormentas[tormentas["Nombre"] == "Unnamed"]

print("Tormentas en total:", len(tormentas))
print("Tormentas con nombre:", len(con_nombre))
print("Tormentas sin nombre:", len(sin_nombre))

# Frecuencias de los nombres
frecuencias = con_nombre["Nombre"].value_counts()

print("\nNombres distintos:", len(frecuencias))
print("Usados una sola vez:", sum(frecuencias == 1))
print("Usados varias veces:", sum(frecuencias > 1))

print("\n10 nombres mas frecuentes:")
print(frecuencias.head(10).to_string())

# Longitud de los nombres 

longitudes = frecuencias.index.str.len()

print("\nNombre mas corto:", frecuencias.index[longitudes.argmin()])
print("Nombre mas largo:", frecuencias.index[longitudes.argmax()])

# Graficas
# Nube de palabras
nube = WordCloud(width=1600, height=900, background_color="white", max_words=len(frecuencias))
nube.generate_from_frequencies(frecuencias.to_dict())
plt.figure(figsize=(16, 9))
plt.imshow(nube)
plt.axis("off")
plt.title("Nombres de tormentas")
plt.tight_layout()
plt.savefig(CARPETA / "nube_nombres.png", dpi=150)
plt.close()

# Nombres mas frecuentes
plt.figure(figsize=(8, 6))
top = frecuencias.head(15).sort_values()
plt.barh(top.index, top.values, color="steelblue")
plt.title("Nombres de tormentas mas frecuentes")
plt.xlabel("Numero de tormentas")
plt.tight_layout()
plt.savefig(CARPETA / "top_nombres.png", dpi=150)
plt.close()

# Tormentas con nombre y sin nombre por decada
tormentas["Decada"] = (tormentas["Year"] // 10) * 10
tormentas["Con nombre"] = tormentas["Nombre"] != "Unnamed"

tabla = tormentas.groupby(["Decada", "Con nombre"]).size().unstack(fill_value=0)

tabla.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title("Tormentas con nombre y sin nombre por decada")
plt.xlabel("Decada")
plt.ylabel("Numero de tormentas")
plt.legend(["Sin nombre", "Con nombre"])
plt.tight_layout()
plt.savefig(CARPETA / "nombres_por_decada.png", dpi=150)
plt.close()