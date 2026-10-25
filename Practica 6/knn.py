from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA = (
    RAIZ_PROYECTO
    / "Practica 1"
    / "Week 23 - Atlantic Hurricane Tracks 1851 to 2017 clean.csv"
)
CARPETA = Path(__file__).resolve().parent / "salidas"
CARPETA.mkdir(parents=True, exist_ok=True)

# Cargar y preparar datos
df = pd.read_csv(RUTA)
df["Tipo"] = df["Observation_Storm_Type"].str.strip()
df["Mes"] = pd.to_datetime(df["Observation_Date"]).dt.month
df = df[df["Tipo"].isin(["TD", "TS", "HU"])]

variables = [
    "Observation_Minimum_Pressure",
    "Observation_Latitude",
    "Observation_Longitude",
    "Mes"
]
df = df.dropna(subset=variables + ["Storm_ID"])

X = df[variables]
y = df["Tipo"]
grupos = df["Storm_ID"]

print("Observaciones:", len(df))

# Separar tormentas para entrenamiento y prueba
tormentas = np.random.default_rng(42).permutation(grupos.unique())
prueba = grupos.isin(tormentas[:int(len(tormentas) * 0.2)])

X_train, X_test = X[~prueba], X[prueba]
y_train, y_test = y[~prueba], y[prueba]
grupos_train = grupos[~prueba]

print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

# Comparar valores de K mediante validacion cruzada
valores_k = range(1, 31)
exactitudes = []

for k in valores_k:
    modelo = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    puntajes = cross_val_score(modelo, X_train, y_train, cv=GroupKFold(n_splits=5),groups=grupos_train)
    exactitudes.append(puntajes.mean())

mejor_k = list(valores_k)[np.argmax(exactitudes)]
print("Mejor K:", mejor_k)
print("Exactitud de validación:", round(max(exactitudes), 4))

# Entrenar el modelo final
modelo = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=mejor_k)
)
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)

exactitud = accuracy_score(y_test, predicciones)
print("Exactitud de prueba:", round(exactitud, 4))

# Comparacion de exactitud segun K
plt.plot(valores_k, exactitudes, marker="o")
plt.axvline(mejor_k, color="red", linestyle="--", label=f"Mejor K = {mejor_k}")
plt.title("Exactitud segun K")
plt.xlabel("Numero de vecinos (K)")
plt.ylabel("Exactitud promedio")
plt.legend()
plt.tight_layout()
plt.savefig(CARPETA / "exactitud_segun_k.png")
plt.close()

# Matriz de confusion
ConfusionMatrixDisplay.from_predictions(
    y_test, predicciones, labels=["TD", "TS", "HU"], cmap="Blues"
)
plt.title(f"Matriz de confusion (K = {mejor_k})")
plt.tight_layout()
plt.savefig(CARPETA / "matriz_confusion.png")
plt.close()

# Tipos de tormenta
for tipo in ["TD", "TS", "HU"]:
    grupo = df[df["Tipo"] == tipo]
    plt.scatter(
        grupo["Observation_Minimum_Pressure"],
        grupo["Observation_Latitude"],
        label=tipo, alpha=0.4, s=10
    )

plt.title("Tipos de tormenta: presion vs latitud")
plt.xlabel("Presion minima (mb)")
plt.ylabel("Latitud")
plt.legend()
plt.tight_layout()
plt.savefig(CARPETA / "tipos_tormenta.png")
plt.close()