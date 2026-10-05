import itertools
import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent
PROYECTO_DIR = BASE_DIR.parent
RUTA = PROYECTO_DIR / "Practica 1" / "Week 23 - Atlantic Hurricane Tracks 1851 to 2017 clean.csv"
DIR_SALIDA = BASE_DIR / "salidas"
ALPHA = 0.05
RNG = np.random.default_rng(42)


# Utilidades de formato
def fmt_p(p: float) -> str:
    return "~0 (menor que 1e-300)" if p == 0 else f"{p:.4g}"


def fmt_p_corto(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


# Carga de datos
def cargar_datos(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    # Los códigos categóricos vienen con espacios al inicio (" HU", " TS"...)
    for col in ["Observation_Storm_Type", "Observation_Event"]:
        df[col] = df[col].str.strip()
    df["Anio"] = df["Storm_ID"].str[-4:].astype(int)
    return df


# Funciones de contraste
def verificar_supuestos(grupos: dict) -> dict:
    res = {"normalidad": {}, "levene_p": None}
    for nombre, x in grupos.items():
        # Shapiro pierde fiabilidad con n muy grande: se usa una submuestra (<=5000)
        muestra = x if len(x) <= 5000 else RNG.choice(x, 5000, replace=False)
        res["normalidad"][nombre] = stats.shapiro(muestra).pvalue
    res["levene_p"] = stats.levene(*grupos.values(), center="median").pvalue
    res["normal_todos"] = all(p > ALPHA for p in res["normalidad"].values())
    res["homogeneas"] = res["levene_p"] > ALPHA
    return res


def eta_cuadrado(grupos: dict) -> float:
    todos = np.concatenate(list(grupos.values()))
    media = todos.mean()
    sc_entre = sum(len(x) * (x.mean() - media) ** 2 for x in grupos.values())
    sc_total = ((todos - media) ** 2).sum()
    return sc_entre / sc_total


def epsilon_cuadrado(h: float, n: int) -> float:
    return h / (n - 1)


def interpretar_efecto(v: float) -> str:
    return "pequeño" if v < 0.06 else "mediano" if v < 0.14 else "grande"


def comparar_grupos(grupos: dict, variable: str, etiqueta: str):

    salida = []

    def log(texto: str = ""):
        print(texto)
        salida.append(texto)
        
    log(f"{variable} según {etiqueta}")
    

    # Descriptivos
    desc = pd.DataFrame(
        {k: {"n": len(v), "media": v.mean(), "mediana": np.median(v), "desv.est.": v.std(ddof=1)}
         for k, v in grupos.items()}
    ).T
    desc["n"] = desc["n"].astype(int)
    log("\nEstadísticos descriptivos:")
    log(desc.round(2).to_string())

    # Supuestos
    sup = verificar_supuestos(grupos)
    log("\nSupuestos")
    for k, p in sup["normalidad"].items():
        log(f"  Shapiro-Wilk [{k}]: p = {fmt_p(p)}  -> " + ("normal" if p > ALPHA else "NO normal"))
    log(f"  Levene: p = {fmt_p(sup['levene_p'])}  -> "
        + ("varianzas homogéneas" if sup["homogeneas"] else "varianzas NO homogéneas"))

    usar_anova = sup["normal_todos"] and sup["homogeneas"]
    pares = list(itertools.combinations(grupos.keys(), 2))
    m = len(pares)  # número de comparaciones para la corrección de Bonferroni
    filas = []

    if usar_anova:
        log("\n>> Se cumplen los supuestos: ANOVA de un factor + pruebas t post hoc")
        f, p = stats.f_oneway(*grupos.values())
        ef = eta_cuadrado(grupos)
        log(f"  ANOVA: F = {f:.3f}, p = {fmt_p(p)}, eta² = {ef:.3f} ({interpretar_efecto(ef)})")
        prueba, stat, p_global = "ANOVA", f, p
        for a, b in pares:
            t, pp = stats.ttest_ind(grupos[a], grupos[b], equal_var=True)
            filas.append((a, b, "t de Student", t, min(pp * m, 1.0)))
    else:
        log("\n>> NO se cumplen los supuestos: prueba de Kruskal-Wallis + Mann-Whitney post hoc")
        h, p = stats.kruskal(*grupos.values())
        n = sum(len(x) for x in grupos.values())
        ef = epsilon_cuadrado(h, n)
        log(f"  Kruskal-Wallis: H = {h:.3f}, p = {fmt_p(p)}, epsilon² = {ef:.3f} ({interpretar_efecto(ef)})")
        prueba, stat, p_global = "Kruskal-Wallis", h, p
        for a, b in pares:
            u, pp = stats.mannwhitneyu(grupos[a], grupos[b], alternative="two-sided")
            filas.append((a, b, "Mann-Whitney U", u, min(pp * m, 1.0)))

    # Post hoc (solo tiene sentido si la prueba global es significativa)
    post = pd.DataFrame(filas, columns=["Grupo A", "Grupo B", "Prueba", "Estadístico", "p (Bonferroni)"])
    post["Significativo"] = post["p (Bonferroni)"] < ALPHA
    if p_global < ALPHA:
        log(f"\n  Prueba global significativa -> comparaciones por pares (Bonferroni, {m} pares):")
        post_imp = post.copy()
        post_imp["Estadístico"] = post_imp["Estadístico"].round(2)
        post_imp["p (Bonferroni)"] = post_imp["p (Bonferroni)"].map(fmt_p_corto)
        post_imp["Significativo"] = post_imp["Significativo"].map({True: "Sí", False: "No"})
        log(post_imp.to_string(index=False))
        concl = "Existen diferencias estadísticamente significativas entre los grupos."
    else:
        log("\n  Prueba global NO significativa: no se interpreta el post hoc.")
        concl = "No hay evidencia de diferencias entre los grupos."
    log(f"\n  Conclusión (alfa = {ALPHA}): {concl}\n")

    return {"prueba": prueba, "estadistico": stat, "p": p_global, "efecto": ef,
            "efecto_nombre": "eta²" if usar_anova else "epsilon²",
            "post_hoc": post, "descriptivos": desc, "supuestos": sup,
            "usar_anova": usar_anova, "texto": "\n".join(salida)}


# Informe de texto
def interpretacion_en_palabras(res: dict, unidad_var: str, nombre_grupos: str) -> str:
    desc = res["descriptivos"]
    post = res["post_hoc"]
    lineas = []
    ef_txt = interpretar_efecto(res["efecto"])
    if res["p"] < ALPHA:
        lineas.append(
            f"- La prueba global ({res['prueba']}) es significativa (p = {fmt_p(res['p'])}): "
            f"al menos un grupo se distingue de los demás.")
        lineas.append(
            f"- El tamaño del efecto es {ef_txt} ({res['efecto_nombre']} = {res['efecto']:.3f}). "
            f"Esto indica qué tan relevante es la diferencia más allá de su significancia.")
        orden = desc["mediana"].sort_values()
        lineas.append(
            f"- Medianas de menor a mayor ({unidad_var}): "
            + "; ".join(f"{g} = {v:.1f}" for g, v in orden.items()) + ".")
        sig = post[post["Significativo"]]
        nosig = post[~post["Significativo"]]
        lineas.append(
            f"- Comparaciones por pares (Bonferroni): {len(sig)} de {len(post)} son significativas.")
        if len(nosig):
            lineas.append(
                "  No significativas: "
                + ", ".join(f"{a} vs {b}" for a, b in zip(nosig["Grupo A"], nosig["Grupo B"])) + ".")
        justos = sig[sig["p (Bonferroni)"] > 0.01]
        if len(justos):
            lineas.append(
                "  Significativas por margen estrecho (p entre 0.01 y 0.05): "
                + ", ".join(f"{a} vs {b} (p = {p:.4f})"
                            for a, b, p in zip(justos["Grupo A"], justos["Grupo B"],
                                               justos["p (Bonferroni)"])) + ".")
    else:
        lineas.append(
            f"- La prueba global ({res['prueba']}) NO es significativa (p = {fmt_p(res['p'])}): "
            f"no hay evidencia de diferencias entre {nombre_grupos}.")
    return "\n".join(lineas)


# Programa principal
def main():
    os.makedirs(DIR_SALIDA, exist_ok=True)
    df = cargar_datos(RUTA)

    # Análisis de presión mínima por tipo de tormenta
    tipos = ["TD", "TS", "HU"]
    a = df[df["Observation_Storm_Type"].isin(tipos)].dropna(subset=["Observation_Minimum_Pressure"])
    grupos_a = {t: a.loc[a["Observation_Storm_Type"] == t, "Observation_Minimum_Pressure"].to_numpy()
                for t in tipos}
    res_a = comparar_grupos(grupos_a, "Presión mínima (mb)", "tipo de tormenta (TD / TS / HU)")

    # Análisis de viento máximo por tormenta y época
    storms = (df.groupby("Storm_ID")
                .agg(Anio=("Anio", "first"), Viento_max=("Observation_Max_Wind", "max"))
                .dropna())
    bins = [1850, 1899, 1949, 1999, 2017]
    labels = ["1851-1899", "1900-1949", "1950-1999", "2000-2017"]
    storms["Epoca"] = pd.cut(storms["Anio"], bins=bins, labels=labels)
    grupos_b = {e: storms.loc[storms["Epoca"] == e, "Viento_max"].to_numpy() for e in labels}
    res_b = comparar_grupos(grupos_b, "Viento máximo por tormenta (kt)", "época")

    # Gráficos
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=a, x="Observation_Storm_Type", y="Observation_Minimum_Pressure",
                order=tipos, palette="Set2", ax=ax[0])
    ax[0].set(title=f"A) Presión mínima por tipo de tormenta\n"
                    f"{res_a['prueba']}: p = {res_a['p']:.2e}",
              xlabel="Tipo de tormenta", ylabel="Presión mínima (mb)")
    sns.boxplot(data=storms, x="Epoca", y="Viento_max", order=labels, palette="Set2", ax=ax[1])
    ax[1].set(title=f"B) Viento máximo por tormenta según época\n"
                    f"{res_b['prueba']}: p = {res_b['p']:.2e}",
              xlabel="Época", ylabel="Viento máximo (kt)")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_SALIDA, "boxplots_pruebas.png"), dpi=150)

    # Distribuciones para revisar la normalidad visualmente
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for t in tipos:
        sns.kdeplot(grupos_a[t], label=t, fill=True, alpha=0.3, ax=ax[0])
    ax[0].set(title="Distribución de la presión mínima por tipo", xlabel="mb"); ax[0].legend()
    for e in labels:
        sns.kdeplot(grupos_b[e], label=e, fill=True, alpha=0.2, ax=ax[1])
    ax[1].set(title="Distribución del viento máximo por época", xlabel="kt"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_SALIDA, "distribuciones.png"), dpi=150)

if __name__ == "__main__":
    main()