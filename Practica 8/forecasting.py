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

YEARS_PRUEBA = 15      # ultimos años que se reservan para evaluar
YEARS_FUTUROS = 10     # cuantos años se pronostican

# numero de tormentas por año
df = pd.read_csv(RUTA)
df["Year"] = df["Storm_ID"].str[-4:].astype(int)         
serie = df.drop_duplicates("Storm_ID").groupby("Year").size()
years = serie.index.to_numpy()
tormentas = serie.to_numpy()
print(f"Serie: {years[0]}-{years[-1]} ({len(serie)} años)")
print(f"Promedio: {tormentas.mean():.1f} tormentas por año")

years_train, y_train = years[:-YEARS_PRUEBA], tormentas[:-YEARS_PRUEBA]
years_test, y_test = years[-YEARS_PRUEBA:], tormentas[-YEARS_PRUEBA:]
print(f"Entrenamiento: {years_train[0]}-{years_train[-1]} | Prueba: {years_test[0]}-{years_test[-1]}")

# Funciones para medir el error
def mae(real, pred):
    return np.mean(np.abs(real - pred))

def rmse(real, pred):
    return np.sqrt(np.mean((real - pred) ** 2))

def r_cuadrado(real, pred):
    return 1 - np.sum((real - pred) ** 2) / np.sum((real - real.mean()) ** 2)

# regresion lineal con el año como variable (tendencia)
# Tormentas = m · (Año - 1900) + b
m, b = np.polyfit(years_train - 1900, y_train, 1)
pred_tendencia = m * (years_test - 1900) + b

# promedio historico
pred_promedio = np.full(len(y_test), y_train.mean())

# promedio movil ultimos 10 años
pred_movil = np.full(len(y_test), y_train[-10:].mean())

modelos = {
    "Tendencia lineal": pred_tendencia,
    "Promedio historico": pred_promedio,
    "Promedio movil": pred_movil,
}
print("\nEvaluacion en los años de prueba:")
print(f"{'Modelo':27s} | {'MAE':>5s} | {'RMSE':>5s} | {'R²':>6s}")
for nombre, pred in modelos.items():
    print(f"{nombre:27s} | {mae(y_test, pred):5.2f} | {rmse(y_test, pred):5.2f} | {r_cuadrado(y_test, pred):6.3f}")

m, b = np.polyfit(years - 1900, tormentas, 1)
ajuste = m * (years - 1900) + b
residuos = tormentas - ajuste
sigma = residuos.std(ddof=2)
print(f"\nModelo: Tormentas = {m:.4f} · (Año - 1900) + {b:.2f}")
print(f"Tendencia: {m:.3f} tormentas mas por año")
print(f"R²: {r_cuadrado(tormentas, ajuste):.3f}")

# Pronostico de los años futuros
years_futuros = np.arange(years[-1] + 1, years[-1] + 1 + YEARS_FUTUROS)
pronostico = m * (years_futuros - 1900) + b
inferior = np.maximum(pronostico - 1.96 * sigma, 0)
superior = pronostico + 1.96 * sigma

tabla = pd.DataFrame({
    "Año": years_futuros,
    "Pronostico": pronostico.round(1),
    "Limite inferior": inferior.round(1),
    "Limite superior": superior.round(1),
})
print("\nPronostico:")
print(tabla.to_string(index=False))

# Graficas
plt.figure(figsize=(10, 6))
plt.plot(years, tormentas, marker="o", markersize=3, linewidth=1, label="Tormentas por año")
plt.plot(years, ajuste, color="red", linewidth=2,
         label=f"Tendencia lineal ")
plt.title("Serie historica de tormentas del Atlantico")
plt.xlabel("Año")
plt.ylabel("Numero de tormentas")
plt.legend()
plt.tight_layout()
plt.savefig(CARPETA / "tendencia.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(years_test, y_test, marker="o", color="black", linewidth=2, label="Real")
plt.plot(years_test, pred_tendencia, marker="s", label="Tendencia lineal")
plt.plot(years_test, pred_promedio, linestyle="--", label="Promedio historico")
plt.plot(years_test, pred_movil, linestyle=":", label="Promedio movil ")
plt.title("Real y pronosticado")
plt.xlabel("Año")
plt.ylabel("Numero de tormentas")
plt.legend()
plt.tight_layout()
plt.savefig(CARPETA / "validacion.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(years[-40:], tormentas[-40:], marker="o", markersize=3, label="Datos historicos")
plt.plot(years_futuros, pronostico, marker="o", color="red", label="Pronostico")
plt.fill_between(years_futuros, inferior, superior, color="red", alpha=0.2,
                 label="Intervalo de 95 %")
plt.title(f"Pronostico de tormentas {years_futuros[0]}-{years_futuros[-1]}")
plt.xlabel("Año")
plt.ylabel("Numero de tormentas")
plt.legend()
plt.tight_layout()
plt.savefig(CARPETA / "pronostico.png", dpi=150)
plt.close()