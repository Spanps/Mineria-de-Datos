
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score
)

ruta = Path(__file__).resolve().parent.parent / "Practica 1" / "Week 23 - Atlantic Hurricane Tracks 1851 to 2017 clean.csv"
carpeta = Path(__file__).resolve().parent / "salidas"
carpeta.mkdir(exist_ok=True)

# Cargar datos
variables = [
    "Observation_Max_Wind",
    "Observation_Minimum_Pressure",
    "Observation_Latitude",
    "Observation_Longitude"
]

df = pd.read_csv(ruta).dropna(subset=variables).reset_index(drop=True)
X = StandardScaler().fit_transform(df[variables])

print("Observaciones:", len(df))

# Probar modelos con distintos valores de K
valores_k = range(2, 11)
inercias = []
siluetas = []

print("\nK | Inercia | Silueta | Davies-Bouldin | Calinski-Harabasz")

for k in valores_k:
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    grupos = modelo.fit_predict(X)

    inercias.append(modelo.inertia_)
    silueta = silhouette_score(
        X, grupos, sample_size=5000, random_state=42
    )
    siluetas.append(silueta)

    print(
        f"{k} | {modelo.inertia_:.0f}   | {silueta:.3f}   | "
        f"{davies_bouldin_score(X, grupos):.3f}          | "
        f"{calinski_harabasz_score(X, grupos):.0f}"
    )

# Modelo
K = 4
modelo = KMeans(n_clusters=K, random_state=42, n_init=10)
df["Cluster"] = modelo.fit_predict(X)

orden = df.groupby("Cluster")["Observation_Max_Wind"].mean().sort_values().index
numeros = {grupo: i for i, grupo in enumerate(orden)}
df["Cluster"] = df["Cluster"].map(numeros)

print("\nEvaluación del modelo:")
print("Silueta:", round(silhouette_score( X, modelo.labels_, sample_size=5000, random_state=42), 3))
print("Davies-Bouldin:", round(davies_bouldin_score(X, modelo.labels_), 3))
print("Calinski-Harabasz:", round(calinski_harabasz_score(X, modelo.labels_), 0))

# Promedios de cada grupo
print("\nPerfil de los clusters:")
print(df.groupby("Cluster")[variables].mean().round(1))

# Comparar con el tipo real de tormenta
tipos = df["Observation_Storm_Type"].str.strip()
seleccion = tipos.isin(["TD", "TS", "HU"])

print("\nClusters vs tipo de tormenta:")
print(pd.crosstab(df.loc[seleccion, "Cluster"], tipos[seleccion]))

print("Indice de Rand ajustado:", round(adjusted_rand_score(tipos[seleccion], df.loc[seleccion, "Cluster"]), 3))

# Grafica del codo
plt.plot(valores_k, inercias, marker="o")
plt.axvline(K, color="red", linestyle="--")
plt.title("Metodo del codo")
plt.xlabel("Numero de clusters")
plt.ylabel("Inercia")
plt.savefig(carpeta / "codo.png")
plt.close()

# Grafica de silueta
plt.plot(valores_k, siluetas, marker="o")
plt.axvline(K, color="red", linestyle="--")
plt.title("Coeficiente de silueta")
plt.xlabel("Numero de clusters")
plt.ylabel("Silueta")
plt.savefig(carpeta / "silueta.png")
plt.close()

# Grafica de viento y presion
for c in range(K):
    grupo = df[df["Cluster"] == c]
    plt.scatter(grupo["Observation_Max_Wind"], grupo["Observation_Minimum_Pressure"], s=6, alpha=0.3, label=f"Cluster {c}")

centros = df.groupby("Cluster")[["Observation_Max_Wind", "Observation_Minimum_Pressure"]].mean()

plt.scatter(centros.iloc[:, 0], centros.iloc[:, 1],color="black", marker="X", s=150, label="Centros")
plt.xlabel("Viento maximo (kt)")
plt.ylabel("Presion minima (mb)")
plt.title("Clusters: viento vs presion")
plt.legend()
plt.savefig(carpeta / "clusters_viento_presion.png")
plt.close()

# Grafica geografica
for c in range(K):
    grupo = df[df["Cluster"] == c]
    plt.scatter(grupo["Observation_Longitude"], grupo["Observation_Latitude"], s=6, alpha=0.3, label=f"Cluster {c}")

plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Clusters")
plt.legend()
plt.savefig(carpeta / "clusters.png")
plt.close()