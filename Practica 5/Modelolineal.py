from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA = (
    RAIZ_PROYECTO
    / "Practica 1"
    / "Week 23 - Atlantic Hurricane Tracks 1851 to 2017 clean.csv"
)
CARPETA = Path(__file__).resolve().parent / "salidas"
CARPETA.mkdir(parents=True, exist_ok=True)

# Cargar datos y quitar filas con valores faltantes
df = pd.read_csv(RUTA)
df = df.dropna(subset=["Observation_Max_Wind", "Observation_Minimum_Pressure"])

viento = df["Observation_Max_Wind"].to_numpy()           # x
presion = df["Observation_Minimum_Pressure"].to_numpy()  # y
print("Observaciones usadas:", len(df))

# Correlacion de Pearson
dx = viento - viento.mean()
dy = presion - presion.mean()
r = np.sum(dx * dy) / np.sqrt(np.sum(dx**2) * np.sum(dy**2))
print(f"Correlacion de Pearson (r): {r:.4f}")

# Ajustar la recta: Presion = m · Viento + b
m = np.sum(dx * dy) / np.sum(dx**2)        # pendiente
b = presion.mean() - m * viento.mean()     # intercepto
print(f"Modelo: Presion = {m:.4f} · Viento + {b:.4f}")

# Predicciones, residuos y R cuadrado
prediccion = m * viento + b
residuos = presion - prediccion

suma_residual = np.sum(residuos**2)        # error que deja el modelo
suma_total = np.sum(dy**2)                 # variación total de la presión
r2 = 1 - suma_residual / suma_total
print(f"R²: {r2:.4f}")

# Graficas 
# Dispersión y recta de regresion
plt.figure(figsize=(8, 6))
x_linea = np.linspace(viento.min(), viento.max(), 100)
plt.scatter(viento, presion, s=6, alpha=0.2, label="Observaciones")
plt.plot(x_linea, m * x_linea + b, color="red", linewidth=2, label=f"Presion = {m:.3f}·Viento + {b:.1f}\nR² = {r2:.3f}")
plt.title("Dispersion y recta de regresion")
plt.xlabel("Viento máximo (kt)")
plt.ylabel("Presion mínima (mb)")
plt.legend()
plt.tight_layout()
plt.savefig(CARPETA / "dispersion_recta.png", dpi=150)
plt.close()

# Presion observada contra presion predicha
plt.figure(figsize=(8, 6))
plt.scatter(presion, prediccion, s=6, alpha=0.2, color="green")
limites = [presion.min(), presion.max()]
plt.plot(limites, limites, "k--", label="Predicción perfecta")
plt.title("Observado vs predicho")
plt.xlabel("Presion observada (mb)")
plt.ylabel("Presion predicha (mb)")
plt.legend()
plt.tight_layout()
plt.savefig(CARPETA / "observado_vs_predicho.png", dpi=150)
plt.close()

# Residuos contra valores predichos
plt.figure(figsize=(8, 6))
plt.scatter(prediccion, residuos, s=6, alpha=0.2, color="orange")
plt.axhline(0, color="black")
plt.title("Residuos vs valores predichos")
plt.xlabel("Presion predicha (mb)")
plt.ylabel("Residuo (mb)")
plt.tight_layout()
plt.savefig(CARPETA / "residuos.png", dpi=150)
plt.close()