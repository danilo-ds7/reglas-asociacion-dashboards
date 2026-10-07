"""
Dashboard de Reglas de Asociación — versión STREAMLIT
Basado en la guía de clase de Danilo Gómez Correa.
Ejecutar:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

import core as c

st.set_page_config(page_title="Reglas de Asociación · Streamlit", page_icon="🛒", layout="wide")

st.markdown(f"""
<style>
h1, h2, h3 {{ color: {c.AZUL}; }}
[data-testid="stMetricValue"] {{ color: {c.AZUL}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
.caja {{ background:#f4f6fa; border-left:4px solid {c.ROJO}; padding:.7rem 1rem; border-radius:6px; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ caché
@st.cache_data(show_spinner="Cargando 9.835 tickets…")
def datos():
    compras = c.cargar_compras()
    return compras, compras.mean().sort_values(ascending=False)


@st.cache_data(show_spinner="Minando reglas…")
def reglas_cache(min_sop, min_conf, algoritmo, max_len):
    compras, _ = datos()
    itemsets, reglas = c.minar(compras, min_sop, min_conf, algoritmo, max_len)
    if not reglas.empty:
        reglas = c.agregar_fisher(reglas, compras)
        reglas["redundante"] = c.marcar_redundantes(reglas)
    return itemsets, reglas


@st.cache_data
def pares_cache():
    compras, _ = datos()
    return c.reglas_pares(compras)


@st.cache_data
def reglas_rec_cache(sop, conf):
    compras, _ = datos()
    return c.minar_reglas_rec(compras, sop, conf)


@st.cache_data(show_spinner="Preparando evaluación leave-one-out…")
def eval_cache(sop, conf):
    compras, _ = datos()
    return c.preparar_evaluacion(compras, soporte=sop, confianza=conf)


compras, soporte_producto = datos()


def html_interactivo(pagina, height):
    # Grafos pyvis (vis.js embebido en la página) dentro de un iframe
    if hasattr(st, "iframe"):
        st.iframe(pagina, height=height)
    else:  # versiones anteriores de Streamlit
        import streamlit.components.v1 as components
        components.html(pagina, height=height)
productos = sorted(compras.columns)

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.title("🛒 Reglas de Asociación")
    st.caption("Supermercado *Groceries* · 9.835 tickets · 169 productos")
    st.subheader("Parámetros de minería")
    algoritmo = st.radio("Algoritmo", ["fpgrowth", "apriori"], horizontal=True,
                         help="Encuentran los mismos itemsets; FP-Growth es más rápido con soportes bajos.")
    min_sop = st.select_slider("Soporte mínimo (minsop)",
                               options=[0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05], value=0.01)
    min_conf = st.slider("Confianza mínima (minconf)", 0.05, 1.0, 0.5, 0.05)
    max_len = st.select_slider("Largo máximo del itemset", options=[2, 3, 4, 5, "sin límite"], value="sin límite")
    max_len = None if max_len == "sin límite" else max_len
    st.subheader("Filtros de reglas")
    lift_min = st.slider("Lift mínimo", 0.0, 10.0, 1.0, 0.1)
    consecuente = st.selectbox("Consecuente (rhs)", ["(todos)"] + productos)
    contiene = st.multiselect("El antecedente contiene", productos)
    solo_sig = st.checkbox("Solo significativas (Fisher + Holm, α = 0,01)")
    sin_red = st.checkbox("Ocultar reglas redundantes")
    st.divider()
    st.caption("Docente: Danilo Gómez Correa · DSDATA")

itemsets, reglas = reglas_cache(min_sop, min_conf, algoritmo, max_len)

filtradas = reglas
if not filtradas.empty:
    filtradas = filtradas[filtradas["lift"] >= lift_min]
    if consecuente != "(todos)":
        filtradas = filtradas[filtradas["consequents"] == frozenset({consecuente})]
    if contiene:
        filtradas = filtradas[filtradas["antecedents"].apply(lambda a: set(contiene) <= a)]
    if solo_sig:
        filtradas = filtradas[filtradas["significativa"]]
    if sin_red:
        filtradas = filtradas[~filtradas["redundante"]]

st.title("Reglas de Asociación y Sistemas de Recomendación")
tabs = st.tabs(["🧺 Ejemplo de clase", "🔎 Exploración", "📏 Reglas", "🕸️ Grafos",
                "🔗 Cadena de compras", "🛍️ Recomendador", "📊 Evaluación"])

# ================================================================== 1. EJEMPLO DE CLASE
with tabs[0]:
    st.header("Las 7 canastas, paso a paso")
    a, b = st.columns(2)
    sop_ej = a.slider("minsop (ejemplo)", 0.1, 0.8, 0.4, 0.05, key="sop_ej")
    conf_ej = b.slider("minconf (ejemplo)", 0.1, 1.0, 0.84, 0.01, key="conf_ej")
    datos_ej, items_ej, tabla_ej = c.ejemplo_clase(sop_ej, conf_ej)
    a, b = st.columns(2)
    a.plotly_chart(c.fig_matriz_clase(datos_ej), width="stretch")
    b.plotly_chart(c.fig_soporte_clase(datos_ej, sop_ej), width="stretch")

    st.subheader("Cálculo a mano de una regla")
    x1, x2 = st.columns(2)
    ant = x1.selectbox("Antecedente", datos_ej.columns, index=list(datos_ej.columns).index("leche"))
    con = x2.selectbox("Consecuente", datos_ej.columns, index=list(datos_ej.columns).index("huevo"))
    r = c.evaluar_regla(datos_ej.astype(bool), ant, con)
    m = st.columns(4)
    m[0].metric("Soporte  P(X∩Z)", f"{r['soporte']:.3f}")
    m[1].metric("Confianza  P(Z|X)", f"{r['confianza']:.3f}")
    m[2].metric("P(Z)", f"{r['P(consecuente)']:.3f}")
    m[3].metric("Lift", f"{r['lift']:.3f}")

    a, b = st.columns([1, 2])
    a.markdown(f"**Itemsets frecuentes** (σ ≥ {sop_ej * 7:.1f})")
    a.dataframe(c.redondear(items_ej), hide_index=True, width="stretch")
    b.markdown("**Todas las reglas** generadas desde los itemsets frecuentes")
    b.dataframe(c.redondear(tabla_ej), hide_index=True, width="stretch")

# ================================================================== 2. EXPLORACIÓN
with tabs[1]:
    res = c.resumen(compras)
    m = st.columns(6)
    m[0].metric("Tickets", f"{res['tickets']:,}".replace(",", "."))
    m[1].metric("Productos", res["productos"])
    m[2].metric("Densidad", f"{res['densidad']:.2%}")
    m[3].metric("Promedio / ticket", f"{res['promedio']:.2f}")
    m[4].metric("Mediana / ticket", f"{res['mediana']:.0f}")
    m[5].metric("Máximo / ticket", res["maximo"])

    a, b = st.columns(2)
    a.plotly_chart(c.fig_productos_por_ticket(compras), width="stretch")
    top_n = b.slider("Productos a mostrar", 10, 40, 20, key="topn")
    b.plotly_chart(c.fig_top_productos(soporte_producto, top_n, min_sop), width="stretch")

    top_co = st.slider("Tamaño de la matriz de co-ocurrencias", 6, 20, 12)
    st.plotly_chart(c.fig_coocurrencias(compras, soporte_producto, top_co), width="stretch")

    st.subheader("¿Por qué no basta la confianza?")
    st.markdown('<div class="caja"><i>Whole milk</i> aparece en 1 de cada 4 tickets: casi cualquier regla '
                '<code>X ⇒ {whole milk}</code> tiene confianza "razonable". El <b>lift</b> compara con lo esperado '
                'bajo independencia.</div>', unsafe_allow_html=True)
    comparar = st.multiselect("Antecedentes a comparar", productos,
                              default=["bottled beer", "bottled water", "butter"])
    cons_cmp = st.selectbox("Consecuente", productos, index=productos.index("whole milk"))
    if comparar:
        st.dataframe(c.redondear(pd.DataFrame([c.evaluar_regla(compras, x, cons_cmp) for x in comparar if x != cons_cmp])),
                     hide_index=True, width="stretch")

# ================================================================== 3. REGLAS
with tabs[2]:
    m = st.columns(5)
    m[0].metric("Itemsets frecuentes", f"{len(itemsets):,}".replace(",", "."))
    m[1].metric("Reglas minadas", len(reglas))
    m[2].metric("Reglas tras filtros", len(filtradas))
    m[3].metric("Significativas", int(reglas["significativa"].sum()) if not reglas.empty else 0)
    m[4].metric("Redundantes", int(reglas["redundante"].sum()) if not reglas.empty else 0)

    if not itemsets.empty:
        tam = itemsets["itemsets"].apply(len).value_counts().sort_index()
        st.caption("Itemsets por tamaño: " + " · ".join(f"**{k}**: {v}" for k, v in tam.items()))

    if filtradas.empty:
        st.warning("No hay reglas con estos parámetros. Baje el soporte, la confianza o el lift mínimo.")
    else:
        st.plotly_chart(c.fig_dispersion(filtradas, f"{len(filtradas)} reglas (minsop = {min_sop}; minconf = {min_conf})"),
                        width="stretch")
        tabla = c.tabla_reglas(filtradas)
        st.dataframe(c.redondear(tabla), hide_index=True, width="stretch", height=420,
                     column_config={"lift": st.column_config.ProgressColumn(
                         "lift", format="%.2f", min_value=0, max_value=float(tabla["lift"].max()))})
        st.download_button("⬇️ Descargar reglas (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                           "reglas.csv", "text/csv")

# ================================================================== 4. GRAFOS
with tabs[3]:
    st.subheader("a) Grafo de reglas")
    st.caption("Cuadros: productos · Círculos: reglas (tamaño = soporte, color = lift). Use los filtros de la barra lateral.")
    max_r = st.slider("Máximo de reglas a dibujar (mayor lift primero)", 5, 150, 40)
    if filtradas.empty:
        st.info("Sin reglas para graficar con los filtros actuales.")
    else:
        html_interactivo(c.grafo_reglas_html(filtradas, soporte_producto, 620, max_r), height=640)

    st.subheader("b) Mapa de flujo producto → producto")
    st.caption("Grosor = confianza · Color = lift · Tamaño del nodo = soporte (minsop 0,005; minconf 0,15).")
    lift_flujo = st.slider("Lift mínimo del mapa", 1.2, 4.0, 2.0, 0.1)
    html_flujo, n_flujo = c.mapa_flujo_html(pares_cache(), soporte_producto, lift_flujo, 650)
    st.write(f"Mostrando **{n_flujo}** reglas con lift ≥ {lift_flujo}")
    html_interactivo(html_flujo, height=670)

# ================================================================== 5. CADENA
with tabs[4]:
    st.subheader('"Esto lleva a esto, que lleva a esto…"')
    rp = pares_cache()
    opciones = sorted(rp["desde"].unique())
    a, b, d = st.columns(3)
    inicio = a.selectbox("Producto de partida", opciones, index=opciones.index("sausage"))
    pasos = b.slider("Pasos", 2, 10, 6)
    criterio = d.radio("Criterio", ["lift", "confidence"], horizontal=True)
    camino, tabla_c, regreso, texto = c.cadena_de_compras(rp, inicio, pasos, criterio)
    st.success(texto)
    html_interactivo(c.cadena_html(camino, tabla_c, regreso), height=320)
    st.dataframe(c.redondear(tabla_c), hide_index=True, width="stretch")
    with st.expander("Comparar ambos criterios lado a lado"):
        a, b = st.columns(2)
        for col, cr in [(a, "lift"), (b, "confidence")]:
            _, t, _, tx = c.cadena_de_compras(rp, inicio, pasos, cr)
            col.markdown(f"**{cr}:** {tx}")
            col.dataframe(c.redondear(t), hide_index=True, width="stretch")

# ================================================================== 6. RECOMENDADOR
with tabs[5]:
    st.subheader("Clientes que compraron esto también compraron…")
    a, b, d, e = st.columns(4)
    sop_rec = a.select_slider("minsop", [0.002, 0.003, 0.005, 0.01, 0.02], 0.005, key="sop_rec")
    conf_rec = b.slider("minconf", 0.1, 0.8, 0.3, 0.05, key="conf_rec")
    top_rec = d.slider("Top N", 1, 10, 3, key="top_rec")
    orden_rec = e.radio("Ordenar por", ["lift", "confidence"], horizontal=True, key="ord_rec")
    reglas_rec = reglas_rec_cache(sop_rec, conf_rec)
    st.caption(f"Reglas disponibles (un consecuente): {len(reglas_rec)}")

    ejemplos = {"(personalizada)": None,
                "root vegetables + tropical fruit": ["root vegetables", "tropical fruit"],
                "yogurt + curd": ["yogurt", "curd"],
                "sausage + rolls/buns": ["sausage", "rolls/buns"],
                "bottled beer (arranque en frío)": ["bottled beer"],
                "citrus fruit + root vegetables": ["citrus fruit", "root vegetables"]}
    ej = st.selectbox("Canastas de ejemplo", list(ejemplos))
    canasta = st.multiselect("Canasta del cliente", productos,
                             default=ejemplos[ej] or ["root vegetables", "tropical fruit"])
    if canasta:
        rec = c.recomendar(canasta, reglas_rec, soporte_producto.index.tolist(), top_rec, orden_rec)
        n_reglas = (rec["motivo"] != "más vendido").sum()
        if n_reglas == 0:
            st.warning("Ninguna regla se activa (arranque en frío): se completan con los más vendidos.")
        cols = st.columns(min(len(rec), 5))
        for i, (_, r) in enumerate(rec.head(5).iterrows()):
            cols[i].metric(f"#{i + 1}", r["producto"],
                           f"lift {r['lift']:.2f}" if pd.notna(r["lift"]) else "más vendido",
                           delta_color="normal" if pd.notna(r["lift"]) else "off")
        st.dataframe(c.redondear(rec), hide_index=True, width="stretch")

# ================================================================== 7. EVALUACIÓN
with tabs[6]:
    st.subheader("Evaluación offline leave-one-out (80 % / 20 %)")
    a, b, d = st.columns(3)
    orden_ev = a.radio("Ordenar recomendaciones por", ["lift", "confidence"], horizontal=True, key="ord_ev")
    hibrido = b.checkbox("Híbrido conmutado (reglas solo si alguna activa tiene lift > umbral)")
    umbral_h = d.slider("Umbral de lift del híbrido", 1.0, 4.0, 2.0, 0.1, disabled=not hibrido)
    reglas_tr, pop_tr, casos, n_tr, n_te = eval_cache(0.005, 0.3)
    st.caption(f"Entrenamiento: {n_tr} tickets · Prueba: {n_te} · Casos (≥ 2 productos): {len(casos)} · "
               f"Reglas de entrenamiento: {len(reglas_tr)}")
    Ns = [1, 2, 3, 5, 7, 10]
    res = pd.DataFrame([c.hit_rate(N, casos, reglas_tr, pop_tr, orden_ev, umbral_h if hibrido else None) for N in Ns])
    etiqueta = f"Híbrido (lift > {umbral_h})" if hibrido else f"Reglas ({orden_ev})"
    a, b = st.columns([3, 2])
    a.plotly_chart(c.fig_hitrate(res, etiqueta), width="stretch")
    res["Diferencia"] = res["HitRate reglas"] - res["HitRate más vendidos"]
    b.dataframe(c.redondear(res), hide_index=True, width="stretch")
    b.markdown('<div class="caja"><b>Precisión vs. novedad:</b> los más vendidos aciertan mucho por ser frecuentes; '
               'las reglas sugieren productos menos obvios. Compare el criterio <i>lift</i> con <i>confidence</i>.</div>',
               unsafe_allow_html=True)
