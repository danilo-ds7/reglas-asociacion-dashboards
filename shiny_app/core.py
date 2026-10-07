"""
core.py — Lógica compartida de los dashboards de Reglas de Asociación.
Reproduce las funciones de la guía "Reglas de Asociación y Sistemas de Recomendación en Python"
(Danilo Gómez Correa). La usan por igual la app de Streamlit y la de Shiny for Python.
"""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from pyvis.network import Network
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Paleta del curso
AZUL = "#0b1f45"
GRIS = "#52607a"
ROJO = "#c0392b"
CLARO = "#e6e9ef"

DATA_PATH = Path(__file__).parent / "data" / "groceries.csv"
URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv"

CANASTAS_CLASE = [
    ["leche", "huevo"],
    ["pan", "queso"],
    ["leche", "pan", "huevo"],
    ["pan", "queso"],
    ["leche", "queso", "huevo"],
    ["pan", "queso", "huevo"],
    ["leche", "pan", "queso", "huevo"],
]


# ---------------------------------------------------------------- datos
def leer_tickets() -> list[list[str]]:
    if DATA_PATH.exists():
        texto = DATA_PATH.read_text(encoding="utf-8")
    else:  # respaldo: descarga directa
        import urllib.request
        texto = urllib.request.urlopen(URL).read().decode("utf-8")
    tickets = []
    for linea in texto.splitlines():
        productos = [p.strip() for p in linea.split(",") if p.strip()]
        if productos:
            tickets.append(productos)
    return tickets


def matriz_binaria(transacciones: list[list[str]], indice=None) -> pd.DataFrame:
    cod = TransactionEncoder()
    m = cod.fit_transform(transacciones)
    return pd.DataFrame(m, columns=cod.columns_, index=indice)


def cargar_compras() -> pd.DataFrame:
    return matriz_binaria(leer_tickets())


# ---------------------------------------------------------------- formato
def texto_itemset(itemset) -> str:
    return "{" + ", ".join(sorted(itemset)) + "}"


def tabla_reglas(reglas: pd.DataFrame) -> pd.DataFrame:
    if reglas.empty:
        return pd.DataFrame(columns=["regla", "soporte", "confianza", "lift", "leverage", "conviccion"])
    tabla = pd.DataFrame({
        "regla": reglas["antecedents"].apply(texto_itemset) + " ⇒ " + reglas["consequents"].apply(texto_itemset),
        "soporte": reglas["support"],
        "confianza": reglas["confidence"],
        "lift": reglas["lift"],
        "leverage": reglas["leverage"],
        "conviccion": reglas["conviction"],
    })
    for col in ["p_ajustado", "significativa", "redundante"]:
        if col in reglas:
            tabla[col] = reglas[col].values
    return tabla.sort_values("lift", ascending=False).reset_index(drop=True)


def redondear(df: pd.DataFrame, dec: int = 3) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes("number").columns:
        if c == "p_ajustado":
            out[c] = out[c].apply(lambda p: f"{p:.1e}")
        else:
            out[c] = out[c].round(dec)
    return out


# ---------------------------------------------------------------- minería
def minar(datos: pd.DataFrame, min_sop: float, min_conf: float, algoritmo: str = "fpgrowth",
          max_len: int | None = None):
    f = fpgrowth if algoritmo == "fpgrowth" else apriori
    itemsets = f(datos, min_support=min_sop, use_colnames=True, max_len=max_len)
    if itemsets.empty:
        return itemsets, pd.DataFrame()
    reglas = association_rules(itemsets, metric="confidence", min_threshold=min_conf)
    return itemsets, reglas


def agregar_fisher(reglas: pd.DataFrame, datos: pd.DataFrame, alpha: float = 0.01) -> pd.DataFrame:
    """Test exacto de Fisher unilateral + corrección de Holm (sección 2.6)."""
    if reglas.empty:
        return reglas
    reglas = reglas.copy()
    cols = {c: datos[c].values for c in datos.columns}
    n = len(datos)

    def p_valor(x, z):
        tx = np.logical_and.reduce([cols[c] for c in x])
        tz = np.logical_and.reduce([cols[c] for c in z])
        a = int((tx & tz).sum()); b = int(tx.sum()) - a
        c = int(tz.sum()) - a; d = n - a - b - c
        return fisher_exact([[a, b], [c, d]], alternative="greater")[1]

    reglas["p_valor"] = [p_valor(x, z) for x, z in zip(reglas["antecedents"], reglas["consequents"])]
    sig, p_adj, _, _ = multipletests(reglas["p_valor"], alpha=alpha, method="holm")
    reglas["significativa"] = sig
    reglas["p_ajustado"] = p_adj
    return reglas


def marcar_redundantes(reglas: pd.DataFrame) -> np.ndarray:
    """X ⇒ Z es redundante si existe X' ⊂ X con el mismo Z y confianza ≥ (sección 2.7)."""
    if reglas.empty:
        return np.array([], dtype=bool)
    redundante = np.zeros(len(reglas), dtype=bool)
    for z, grupo in reglas.groupby(reglas["consequents"]):
        ants = list(grupo["antecedents"]); confs = list(grupo["confidence"]); idx = list(grupo.index)
        for i in range(len(ants)):
            for j in range(len(ants)):
                if ants[j] < ants[i] and confs[j] >= confs[i]:
                    redundante[reglas.index.get_loc(idx[i])] = True
                    break
    return redundante


# ---------------------------------------------------------------- ejemplo de clase
def ejemplo_clase(min_sop=0.4, min_conf=0.84):
    datos = matriz_binaria(CANASTAS_CLASE, indice=[f"T{i}00" for i in range(1, 8)])
    n = len(datos)
    items = apriori(datos, min_support=min_sop, use_colnames=True)
    if not items.empty:
        items["tamaño"] = items["itemsets"].apply(len)
        items["conteo (σ)"] = (items["support"] * n).round().astype(int)
        items = items.sort_values(["tamaño", "support"], ascending=[True, False])
    if items.empty:
        todas = pd.DataFrame()
    else:
        todas = association_rules(items.drop(columns=["tamaño", "conteo (σ)"]), metric="confidence", min_threshold=0)
    tabla = tabla_reglas(todas)
    if not tabla.empty:
        tabla[f"¿conf ≥ {min_conf}?"] = np.where(tabla["confianza"] >= min_conf - 1e-12, "✅", "—")
    items_txt = items.copy()
    if not items_txt.empty:
        items_txt["itemsets"] = items_txt["itemsets"].apply(texto_itemset)
    return datos.astype(int), items_txt.reset_index(drop=True), tabla


def fig_matriz_clase(datos):
    fig = px.imshow(datos, color_continuous_scale=["white", AZUL], aspect="auto",
                    title="Matriz transacción × ítem", text_auto=True)
    fig.update_layout(coloraxis_showscale=False, height=330, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_soporte_clase(datos, min_sop):
    s = datos.mean().sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=s.index, y=s.values, marker_color=[AZUL if v >= min_sop else "#b6bdca" for v in s.values]))
    fig.add_hline(y=min_sop, line_dash="dash", line_color=ROJO, annotation_text=f"minsop = {min_sop}")
    fig.update_layout(title="Soporte de cada producto", yaxis_title="soporte", height=330,
                      margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white")
    return fig


# ---------------------------------------------------------------- exploración
def resumen(compras: pd.DataFrame) -> dict:
    ppt = compras.sum(axis=1)
    return {
        "tickets": len(compras),
        "productos": compras.shape[1],
        "densidad": compras.values.mean(),
        "promedio": ppt.mean(),
        "mediana": ppt.median(),
        "maximo": int(ppt.max()),
    }


def fig_productos_por_ticket(compras):
    vc = compras.sum(axis=1).clip(upper=20).value_counts().sort_index()
    fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=AZUL,
                           hovertemplate="%{x} productos: %{y} tickets<extra></extra>"))
    fig.update_layout(title="Productos por ticket (20 = 20 o más)", xaxis_title="productos",
                      yaxis_title="tickets", height=400, plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_top_productos(soporte_producto, top=20, min_sop=0.01):
    s = soporte_producto.head(top).sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=GRIS,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.add_vline(x=min_sop, line_dash="dash", line_color=ROJO, annotation_text=f"minsop = {min_sop}")
    fig.update_layout(title=f"{top} productos más frecuentes", xaxis_title="soporte",
                      height=max(400, 22 * top), plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_coocurrencias(compras, soporte_producto, top=12):
    top_items = soporte_producto.head(top).index
    sub = compras[top_items].astype(int)
    co = sub.T @ sub
    fig = px.imshow(co, color_continuous_scale="Blues", text_auto=True, aspect="auto",
                    title=f"Co-ocurrencias entre los {top} productos más frecuentes")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_traces(textfont_size=9)
    return fig


def evaluar_regla(compras, antecedente, consecuente) -> dict:
    tx = compras[antecedente]; tz = compras[consecuente]
    p_x, p_z, p_xz = tx.mean(), tz.mean(), (tx & tz).mean()
    return {"regla": f"{{{antecedente}}} ⇒ {{{consecuente}}}", "soporte": p_xz,
            "confianza": p_xz / p_x if p_x else np.nan, "P(consecuente)": p_z,
            "lift": p_xz / (p_x * p_z) if p_x and p_z else np.nan}


# ---------------------------------------------------------------- visualización de reglas
def fig_dispersion(reglas: pd.DataFrame, titulo=""):
    t = tabla_reglas(reglas)
    fig = px.scatter(t, x="soporte", y="confianza", color="lift", hover_name="regla",
                     hover_data={"soporte": ":.4f", "confianza": ":.3f", "lift": ":.2f"},
                     color_continuous_scale="Reds", title=titulo or f"{len(t)} reglas")
    fig.update_traces(marker=dict(size=8, line=dict(width=0.3, color="white")))
    fig.update_layout(height=480, plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def color_lift(valor, minimo, maximo):
    escala = plt.get_cmap("Reds")
    p = (valor - minimo) / (maximo - minimo) if maximo > minimo else 1
    return mcolors.to_hex(escala(0.25 + 0.75 * p))


def _html_red(red: Network) -> str:
    return red.generate_html()


def grafo_reglas_html(reglas: pd.DataFrame, soporte_producto, alto=620, max_reglas=60) -> str:
    reglas = reglas.sort_values("lift", ascending=False).head(max_reglas).reset_index(drop=True)
    red = Network(height=f"{alto}px", width="100%", directed=True, cdn_resources="in_line")
    if reglas.empty:
        return "<p style='font-family:sans-serif'>Sin reglas para graficar.</p>"
    lmin, lmax = reglas["lift"].min(), reglas["lift"].max()
    smax = reglas["support"].max()
    for i, r in reglas.iterrows():
        nombre = f"regla {i + 1}"
        det = (f"{texto_itemset(r['antecedents'])} ⇒ {texto_itemset(r['consequents'])}\n"
               f"soporte:   {r['support']:.3f}\nconfianza: {r['confidence']:.3f}\nlift:      {r['lift']:.2f}")
        red.add_node(nombre, label=" ", title=det, shape="dot", size=10 + 25 * r["support"] / smax,
                     color=color_lift(r["lift"], lmin, lmax))
        for p in r["antecedents"] | r["consequents"]:
            red.add_node(p, label=p, shape="box", color=CLARO, font={"color": AZUL, "size": 18},
                         title=f"soporte: {soporte_producto[p]:.3f}")
        for p in r["antecedents"]:
            red.add_edge(p, nombre, color=GRIS)
        for p in r["consequents"]:
            red.add_edge(nombre, p, color=GRIS)
    red.repulsion(node_distance=140, spring_length=120)
    return _html_red(red)


def reglas_pares(compras, min_sop=0.005, min_conf=0.15, lift_min=1.2):
    pares = fpgrowth(compras, min_support=min_sop, use_colnames=True, max_len=2)
    rp = association_rules(pares, metric="confidence", min_threshold=min_conf)
    rp["desde"] = rp["antecedents"].apply(lambda x: next(iter(x)))
    rp["hacia"] = rp["consequents"].apply(lambda x: next(iter(x)))
    return rp[rp["lift"] > lift_min].reset_index(drop=True)


def mapa_flujo_html(rp, soporte_producto, lift_minimo=2.0, alto=650) -> tuple[str, int]:
    sel = rp[rp["lift"] >= lift_minimo]
    if sel.empty:
        return "<p style='font-family:sans-serif'>Ninguna regla supera ese lift.</p>", 0
    red = Network(height=f"{alto}px", width="100%", directed=True, cdn_resources="in_line")
    lmin, lmax = sel["lift"].min(), sel["lift"].max()
    for p in set(sel["desde"]) | set(sel["hacia"]):
        red.add_node(p, label=p, shape="dot", color=AZUL, size=10 + 80 * soporte_producto[p],
                     font={"size": 30, "color": AZUL, "strokeWidth": 6, "strokeColor": "white"},
                     title=f"{p}\nsoporte: {soporte_producto[p]:.3f}")
    for _, r in sel.iterrows():
        red.add_edge(r["desde"], r["hacia"], width=1 + 8 * r["confidence"],
                     color=color_lift(r["lift"], lmin, lmax),
                     title=(f"{r['desde']} ⇒ {r['hacia']}\nsoporte:   {r['support']:.3f}\n"
                            f"confianza: {r['confidence']:.3f}\nlift:      {r['lift']:.2f}"))
    red.barnes_hut(gravity=-9000, spring_length=160)
    red.options.interaction.hover = True
    return _html_red(red), len(sel)


def cadena_de_compras(rp, inicio, pasos=6, criterio="lift"):
    camino, tramos = [inicio], []
    while len(camino) < pasos:
        salidas = rp[(rp["desde"] == camino[-1]) & ~rp["hacia"].isin(camino)]
        if salidas.empty:
            break
        mejor = salidas.sort_values(criterio, ascending=False).iloc[0]
        tramos.append(mejor); camino.append(mejor["hacia"])
    regreso = rp[(rp["desde"] == camino[-1]) & rp["hacia"].isin(camino[:-1])]
    regreso = regreso.sort_values(criterio, ascending=False).head(1)
    tabla = pd.DataFrame({
        "paso": range(1, len(tramos) + 1),
        "regla": [f"{t['desde']} ⇒ {t['hacia']}" for t in tramos],
        "soporte": [t["support"] for t in tramos],
        "confianza": [t["confidence"] for t in tramos],
        "lift": [t["lift"] for t in tramos],
    })
    texto = " → ".join(camino)
    if not regreso.empty:
        texto += f"  ↺ vuelve a {regreso.iloc[0]['hacia']}"
    return camino, tabla, regreso, texto


def cadena_html(camino, tabla, regreso, alto=300) -> str:
    red = Network(height=f"{alto}px", width="100%", directed=True, cdn_resources="in_line")
    for nivel, p in enumerate(camino):
        red.add_node(p, label=p, x=230 * nivel, y=0, shape="box",
                     color=AZUL if nivel == 0 else CLARO,
                     font={"color": "white" if nivel == 0 else AZUL, "size": 16})
    for _, t in tabla.iterrows():
        a, b = t["regla"].split(" ⇒ ")
        red.add_edge(a, b, label=f"lift {t['lift']:.2f}", color=GRIS, width=1 + 6 * t["confianza"],
                     title=f"conf {t['confianza']:.3f} | lift {t['lift']:.2f}")
    if not regreso.empty:
        r = regreso.iloc[0]
        red.add_edge(r["desde"], r["hacia"], color=ROJO, dashes=True, label="↺",
                     smooth={"type": "curvedCW", "roundness": 0.4})
    red.toggle_physics(False)
    return _html_red(red)


# ---------------------------------------------------------------- recomendador
def minar_reglas_rec(datos, soporte=0.005, confianza=0.3):
    frec = fpgrowth(datos, min_support=soporte, use_colnames=True)
    reglas = association_rules(frec, metric="confidence", min_threshold=confianza)
    reglas = reglas[reglas["consequents"].apply(len) == 1].copy()
    reglas["producto"] = reglas["consequents"].apply(lambda z: next(iter(z)))
    return reglas.reset_index(drop=True)


def recomendar(canasta, reglas, populares, top=3, orden="lift", lift_hibrido=None):
    canasta = set(canasta)
    activa = reglas["antecedents"].apply(lambda x: x <= canasta)
    nueva = ~reglas["producto"].isin(canasta)
    cand = reglas[activa & nueva]
    if lift_hibrido is not None and not (cand["lift"] > lift_hibrido).any():
        cand = cand.iloc[0:0]  # híbrido conmutado: sin regla fuerte → más vendidos
    cand = cand.sort_values([orden, "confidence"], ascending=False).drop_duplicates("producto").head(top)
    sug = [{"producto": r["producto"],
            "motivo": texto_itemset(r["antecedents"]) + " ⇒ " + texto_itemset(r["consequents"]),
            "confianza": r["confidence"], "lift": r["lift"]} for _, r in cand.iterrows()]
    ya = {s["producto"] for s in sug}
    for p in populares:
        if len(sug) >= top:
            break
        if p not in canasta and p not in ya:
            sug.append({"producto": p, "motivo": "más vendido", "confianza": np.nan, "lift": np.nan})
    return pd.DataFrame(sug, columns=["producto", "motivo", "confianza", "lift"])


# ---------------------------------------------------------------- evaluación
def preparar_evaluacion(compras, semilla=42, prop_train=0.8, soporte=0.005, confianza=0.3):
    gen = np.random.default_rng(semilla)
    orden = gen.permutation(len(compras))
    corte = int(prop_train * len(compras))
    train, test = compras.iloc[orden[:corte]], compras.iloc[orden[corte:]]
    reglas_train = minar_reglas_rec(train, soporte, confianza)
    populares = train.mean().sort_values(ascending=False).index.tolist()
    casos = []
    cols = np.array(compras.columns)
    for fila in test.values:
        productos = cols[fila].tolist()
        if len(productos) >= 2:
            oculto = gen.choice(productos)
            casos.append(([p for p in productos if p != oculto], oculto))
    return reglas_train, populares, casos, len(train), len(test)


def hit_rate(N, casos, reglas_train, populares, orden="lift", lift_hibrido=None):
    """HitRate@N leave-one-out. Misma lógica que recomendar(), pero con máscaras de bits (rápido)."""
    idx = {p: i for i, p in enumerate(populares)}
    mask = lambda items: sum(1 << idx[p] for p in items)
    ordenadas = reglas_train.sort_values([orden, "confidence"], ascending=False)
    lista = [(mask(a), p, l) for a, p, l in zip(ordenadas["antecedents"], ordenadas["producto"], ordenadas["lift"])]
    ar = ap = cr = 0
    for conocidos, oculto in casos:
        cm = mask(conocidos)
        activas = [(p, l) for am, p, l in lista if am & ~cm == 0 and not (cm >> idx[p]) & 1]
        if lift_hibrido is not None and not any(l > lift_hibrido for _, l in activas):
            activas = []
        sug = []
        for p, _ in activas:
            if p not in sug:
                sug.append(p)
                if len(sug) == N:
                    break
        cr += len(sug) > 0
        for p in populares:
            if len(sug) >= N:
                break
            if not (cm >> idx[p]) & 1 and p not in sug:
                sug.append(p)
        ar += oculto in sug
        ap += oculto in [p for p in populares if not (cm >> idx[p]) & 1][:N]
    t = len(casos)
    return {"N": N, "HitRate reglas": ar / t, "HitRate más vendidos": ap / t,
            "% casos con regla activa": float(cr / t)}


def fig_hitrate(res: pd.DataFrame, etiqueta="Reglas"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["N"], y=res["HitRate reglas"], mode="lines+markers",
                             name=etiqueta, line=dict(color=AZUL, width=3)))
    fig.add_trace(go.Scatter(x=res["N"], y=res["HitRate más vendidos"], mode="lines+markers",
                             name="Más vendidos", line=dict(color=ROJO, width=3)))
    fig.update_layout(title="¿Acierta el recomendador? HitRate@N", xaxis_title="N (tamaño de la lista)",
                      yaxis_title="HitRate@N", yaxis_range=[0, max(0.6, res[["HitRate reglas", "HitRate más vendidos"]].max().max() + 0.05)],
                      height=420, plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor="#eee"); fig.update_yaxes(gridcolor="#eee")
    return fig


# ---------------------------------------------------------------- utilidades HTML
def iframe(html_pagina: str, alto: int) -> str:
    """Envuelve una página completa (pyvis / plotly) en un iframe srcdoc."""
    return (f'<iframe srcdoc="{html.escape(html_pagina)}" width="100%" height="{alto}" '
            f'style="border:1px solid #dde1e8;border-radius:8px;background:white"></iframe>')


def plotly_pagina(fig) -> str:
    return fig.to_html(include_plotlyjs="cdn", full_html=True, config={"displaylogo": False})
