"""
Dashboard de Reglas de Asociación — versión SHINY for Python
Basado en la guía de clase de Danilo Gómez Correa.
Ejecutar:  shiny run app.py
"""
from functools import lru_cache
from pathlib import Path

import plotly

import pandas as pd
from shiny import App, reactive, render, ui

import core as c

# ------------------------------------------------------------------ datos (en memoria, una vez)
compras = c.cargar_compras()
soporte_producto = compras.mean().sort_values(ascending=False)
productos = sorted(compras.columns)
RP = c.reglas_pares(compras)
OPC_CADENA = sorted(RP["desde"].unique())
SOPORTES = ["0.001", "0.002", "0.003", "0.005", "0.01", "0.02", "0.03", "0.05"]


@lru_cache(maxsize=16)
def reglas_cache(min_sop, min_conf, algoritmo, max_len):
    itemsets, reglas = c.minar(compras, min_sop, min_conf, algoritmo, max_len)
    if not reglas.empty:
        reglas = c.agregar_fisher(reglas, compras)
        reglas["redundante"] = c.marcar_redundantes(reglas)
    return itemsets, reglas


@lru_cache(maxsize=8)
def reglas_rec_cache(sop, conf):
    return c.minar_reglas_rec(compras, sop, conf)


@lru_cache(maxsize=1)
def eval_cache():
    return c.preparar_evaluacion(compras, soporte=0.005, confianza=0.3)


def plot(fig):
    # plotly.js se sirve una sola vez como recurso estático (ver App(..., static_assets))
    return ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False,
                               config={"displaylogo": False, "responsive": True}))


def tabla(df):
    return render.DataGrid(c.redondear(df), width="100%", height="420px", filters=False)


def metrica(titulo, valor, tema="primary", nota=None):
    return ui.value_box(titulo, valor, *([nota] if nota else []), theme=tema)


CSS = f"""
:root {{ --bs-primary: {c.AZUL}; }}
.navbar {{ background:{c.AZUL} !important; }}
.navbar .nav-link, .navbar-brand {{ color:white !important; }}
.navbar .nav-link.active {{ border-bottom:3px solid {c.ROJO}; }}
h3, h4 {{ color:{c.AZUL}; }}
.caja {{ background:#f4f6fa; border-left:4px solid {c.ROJO}; padding:.7rem 1rem; border-radius:6px; margin:.5rem 0; }}
.bslib-value-box .value-box-value {{ font-size:1.5rem; }}
.bslib-value-box.bg-primary {{ background:{c.AZUL} !important; }}
.bslib-value-box.bg-danger {{ background:{c.ROJO} !important; }}
.bslib-value-box.bg-secondary {{ background:{c.GRIS} !important; }}
.irs--shiny .irs-bar, .irs--shiny .irs-single, .irs--shiny .irs-handle {{ background:{c.AZUL} !important; border-color:{c.AZUL} !important; }}
.form-check-input:checked {{ background-color:{c.ROJO}; border-color:{c.ROJO}; }}
"""

# ------------------------------------------------------------------ UI
sidebar = ui.sidebar(
    ui.h5("Parámetros de minería"),
    ui.input_radio_buttons("algoritmo", "Algoritmo", ["fpgrowth", "apriori"], inline=True),
    ui.input_select("min_sop", "Soporte mínimo (minsop)", SOPORTES, selected="0.01"),
    ui.input_slider("min_conf", "Confianza mínima (minconf)", 0.05, 1.0, 0.5, step=0.05),
    ui.input_select("max_len", "Largo máximo del itemset", ["2", "3", "4", "5", "sin límite"], selected="sin límite"),
    ui.h5("Filtros de reglas"),
    ui.input_slider("lift_min", "Lift mínimo", 0, 10, 1.0, step=0.1),
    ui.input_selectize("consecuente", "Consecuente (rhs)", ["(todos)"] + productos),
    ui.input_selectize("contiene", "El antecedente contiene", productos, multiple=True),
    ui.input_checkbox("solo_sig", "Solo significativas (Fisher + Holm, α = 0,01)"),
    ui.input_checkbox("sin_red", "Ocultar reglas redundantes"),
    ui.hr(),
    ui.p("Docente: Danilo Gómez Correa · DSDATA", class_="text-muted small"),
    width=320, title="🛒 Groceries · 9.835 tickets",
)

tab_ejemplo = ui.nav_panel(
    "🧺 Ejemplo de clase",
    ui.h3("Las 7 canastas, paso a paso"),
    ui.layout_columns(
        ui.input_slider("sop_ej", "minsop (ejemplo)", 0.1, 0.8, 0.4, step=0.05),
        ui.input_slider("conf_ej", "minconf (ejemplo)", 0.1, 1.0, 0.84, step=0.01),
    ),
    ui.layout_columns(ui.output_ui("fig_matriz_ej"), ui.output_ui("fig_sop_ej")),
    ui.h4("Cálculo a mano de una regla"),
    ui.layout_columns(
        ui.input_select("ant_ej", "Antecedente", ["huevo", "leche", "pan", "queso"], selected="leche"),
        ui.input_select("con_ej", "Consecuente", ["huevo", "leche", "pan", "queso"], selected="huevo"),
    ),
    ui.output_ui("metricas_ej"),
    ui.layout_columns(
        ui.card(ui.card_header("Itemsets frecuentes"), ui.output_data_frame("items_ej")),
        ui.card(ui.card_header("Todas las reglas"), ui.output_data_frame("reglas_ej")),
        col_widths=[4, 8],
    ),
)

tab_explora = ui.nav_panel(
    "🔎 Exploración",
    ui.output_ui("metricas_exp"),
    ui.layout_columns(
        ui.card(ui.output_ui("fig_ppt")),
        ui.card(ui.input_slider("topn", "Productos a mostrar", 10, 40, 20), ui.output_ui("fig_top")),
    ),
    ui.card(ui.input_slider("top_co", "Tamaño de la matriz de co-ocurrencias", 6, 20, 12), ui.output_ui("fig_co")),
    ui.h4("¿Por qué no basta la confianza?"),
    ui.div(ui.HTML("<i>Whole milk</i> aparece en 1 de cada 4 tickets: casi cualquier regla "
                   "<code>X ⇒ {whole milk}</code> tiene confianza “razonable”. El <b>lift</b> compara con lo "
                   "esperado bajo independencia."), class_="caja"),
    ui.layout_columns(
        ui.input_selectize("comparar", "Antecedentes a comparar", productos, multiple=True,
                           selected=["bottled beer", "bottled water", "butter"]),
        ui.input_selectize("cons_cmp", "Consecuente", productos, selected="whole milk"),
    ),
    ui.output_data_frame("tabla_cmp"),
)

tab_reglas = ui.nav_panel(
    "📏 Reglas",
    ui.output_ui("metricas_reglas"),
    ui.output_ui("tamanos"),
    ui.output_ui("fig_disp"),
    ui.output_data_frame("tabla_reglas"),
    ui.download_button("descargar", "⬇️ Descargar reglas (CSV)", class_="mt-2"),
)

tab_grafos = ui.nav_panel(
    "🕸️ Grafos",
    ui.h4("a) Grafo de reglas"),
    ui.p("Cuadros: productos · Círculos: reglas (tamaño = soporte, color = lift). Use los filtros del panel lateral.",
         class_="text-muted"),
    ui.input_slider("max_r", "Máximo de reglas a dibujar (mayor lift primero)", 5, 150, 40),
    ui.output_ui("grafo"),
    ui.h4("b) Mapa de flujo producto → producto"),
    ui.p("Grosor = confianza · Color = lift · Tamaño del nodo = soporte (minsop 0,005; minconf 0,15).",
         class_="text-muted"),
    ui.input_slider("lift_flujo", "Lift mínimo del mapa", 1.2, 4.0, 2.0, step=0.1),
    ui.output_ui("flujo"),
)

tab_cadena = ui.nav_panel(
    "🔗 Cadena de compras",
    ui.h4("“Esto lleva a esto, que lleva a esto…”"),
    ui.layout_columns(
        ui.input_selectize("inicio", "Producto de partida", OPC_CADENA, selected="sausage"),
        ui.input_slider("pasos", "Pasos", 2, 10, 6),
        ui.input_radio_buttons("criterio", "Criterio", ["lift", "confidence"], inline=True),
    ),
    ui.output_ui("cadena_texto"),
    ui.output_ui("cadena_grafo"),
    ui.output_data_frame("cadena_tabla"),
    ui.accordion(ui.accordion_panel("Comparar ambos criterios lado a lado", ui.output_ui("cadena_comp")), open=False),
)

EJEMPLOS = {"(personalizada)": None,
            "root vegetables + tropical fruit": ["root vegetables", "tropical fruit"],
            "yogurt + curd": ["yogurt", "curd"],
            "sausage + rolls/buns": ["sausage", "rolls/buns"],
            "bottled beer (arranque en frío)": ["bottled beer"],
            "citrus fruit + root vegetables": ["citrus fruit", "root vegetables"]}

tab_rec = ui.nav_panel(
    "🛍️ Recomendador",
    ui.h4("Clientes que compraron esto también compraron…"),
    ui.layout_columns(
        ui.input_select("sop_rec", "minsop", ["0.002", "0.003", "0.005", "0.01", "0.02"], selected="0.005"),
        ui.input_slider("conf_rec", "minconf", 0.1, 0.8, 0.3, step=0.05),
        ui.input_slider("top_rec", "Top N", 1, 10, 3),
        ui.input_radio_buttons("ord_rec", "Ordenar por", ["lift", "confidence"], inline=True),
    ),
    ui.output_ui("n_reglas_rec"),
    ui.layout_columns(
        ui.input_select("ejemplo", "Canastas de ejemplo", list(EJEMPLOS)),
        ui.input_selectize("canasta", "Canasta del cliente", productos, multiple=True,
                           selected=["root vegetables", "tropical fruit"]),
        col_widths=[4, 8],
    ),
    ui.output_ui("rec_cards"),
    ui.output_data_frame("rec_tabla"),
)

tab_eval = ui.nav_panel(
    "📊 Evaluación",
    ui.h4("Evaluación offline leave-one-out (80 % / 20 %)"),
    ui.layout_columns(
        ui.input_radio_buttons("ord_ev", "Ordenar recomendaciones por", ["lift", "confidence"], inline=True),
        ui.input_checkbox("hibrido", "Híbrido conmutado (reglas solo si alguna activa tiene lift > umbral)"),
        ui.input_slider("umbral_h", "Umbral de lift del híbrido", 1.0, 4.0, 2.0, step=0.1),
    ),
    ui.output_ui("eval_info"),
    ui.layout_columns(
        ui.output_ui("fig_eval"),
        ui.div(ui.output_data_frame("tabla_eval"),
               ui.div(ui.HTML("<b>Precisión vs. novedad:</b> los más vendidos aciertan mucho por ser frecuentes; "
                              "las reglas sugieren productos menos obvios."), class_="caja")),
        col_widths=[7, 5],
    ),
)

app_ui = ui.page_navbar(
    tab_ejemplo, tab_explora, tab_reglas, tab_grafos, tab_cadena, tab_rec, tab_eval,
    title="Reglas de Asociación · Shiny",
    sidebar=sidebar,
    header=ui.TagList(ui.tags.style(CSS), ui.tags.script(src="plotly/plotly.min.js")),
    fillable=False,
    window_title="Reglas de Asociación · Shiny",
)


# ------------------------------------------------------------------ servidor
def server(input, output, session):

    # ---------- 1. Ejemplo de clase
    @reactive.calc
    def ej():
        return c.ejemplo_clase(input.sop_ej(), input.conf_ej())

    @render.ui
    def fig_matriz_ej():
        return plot(c.fig_matriz_clase(ej()[0]))

    @render.ui
    def fig_sop_ej():
        return plot(c.fig_soporte_clase(ej()[0], input.sop_ej()))

    @render.ui
    def metricas_ej():
        r = c.evaluar_regla(ej()[0].astype(bool), input.ant_ej(), input.con_ej())
        return ui.layout_columns(
            metrica("Soporte  P(X∩Z)", f"{r['soporte']:.3f}"),
            metrica("Confianza  P(Z|X)", f"{r['confianza']:.3f}"),
            metrica("P(Z)", f"{r['P(consecuente)']:.3f}"),
            metrica("Lift", f"{r['lift']:.3f}", "danger"),
        )

    @render.data_frame
    def items_ej():
        return tabla(ej()[1])

    @render.data_frame
    def reglas_ej():
        return tabla(ej()[2])

    # ---------- 2. Exploración
    @render.ui
    def metricas_exp():
        r = c.resumen(compras)
        return ui.layout_columns(
            metrica("Tickets", f"{r['tickets']:,}".replace(",", ".")),
            metrica("Productos", r["productos"]),
            metrica("Densidad", f"{r['densidad']:.2%}"),
            metrica("Promedio / ticket", f"{r['promedio']:.2f}"),
            metrica("Mediana / ticket", f"{r['mediana']:.0f}"),
            metrica("Máximo / ticket", r["maximo"]),
        )

    @render.ui
    def fig_ppt():
        return plot(c.fig_productos_por_ticket(compras))

    @render.ui
    def fig_top():
        return plot(c.fig_top_productos(soporte_producto, input.topn(), float(input.min_sop())))

    @render.ui
    def fig_co():
        return plot(c.fig_coocurrencias(compras, soporte_producto, input.top_co()))

    @render.data_frame
    def tabla_cmp():
        filas = [c.evaluar_regla(compras, x, input.cons_cmp()) for x in input.comparar() if x != input.cons_cmp()]
        return tabla(pd.DataFrame(filas))

    # ---------- 3. Reglas
    @reactive.calc
    def minado():
        ml = None if input.max_len() == "sin límite" else int(input.max_len())
        with ui.Progress() as p:
            p.set(message="Minando reglas…")
            return reglas_cache(float(input.min_sop()), float(input.min_conf()), input.algoritmo(), ml)

    @reactive.calc
    def filtradas():
        _, r = minado()
        if r.empty:
            return r
        r = r[r["lift"] >= input.lift_min()]
        if input.consecuente() != "(todos)":
            r = r[r["consequents"] == frozenset({input.consecuente()})]
        if input.contiene():
            req = set(input.contiene())
            r = r[r["antecedents"].apply(lambda a: req <= a)]
        if input.solo_sig():
            r = r[r["significativa"]]
        if input.sin_red():
            r = r[~r["redundante"]]
        return r

    @render.ui
    def metricas_reglas():
        it, r = minado()
        return ui.layout_columns(
            metrica("Itemsets frecuentes", f"{len(it):,}".replace(",", ".")),
            metrica("Reglas minadas", len(r)),
            metrica("Reglas tras filtros", len(filtradas()), "danger"),
            metrica("Significativas", int(r["significativa"].sum()) if not r.empty else 0),
            metrica("Redundantes", int(r["redundante"].sum()) if not r.empty else 0),
        )

    @render.ui
    def tamanos():
        it, _ = minado()
        if it.empty:
            return None
        tam = it["itemsets"].apply(len).value_counts().sort_index()
        return ui.p(ui.HTML("Itemsets por tamaño: " + " · ".join(f"<b>{k}</b>: {v}" for k, v in tam.items())),
                    class_="text-muted")

    @render.ui
    def fig_disp():
        f = filtradas()
        if f.empty:
            return ui.div("No hay reglas con estos parámetros. Baje el soporte, la confianza o el lift mínimo.",
                          class_="alert alert-warning")
        return plot(c.fig_dispersion(f, f"{len(f)} reglas (minsop = {input.min_sop()}; minconf = {input.min_conf()})"))

    @render.data_frame
    def tabla_reglas():
        return tabla(c.tabla_reglas(filtradas()))

    @render.download(filename="reglas.csv")
    def descargar():
        yield c.tabla_reglas(filtradas()).to_csv(index=False)

    # ---------- 4. Grafos
    @render.ui
    def grafo():
        f = filtradas()
        if f.empty:
            return ui.div("Sin reglas para graficar con los filtros actuales.", class_="alert alert-info")
        return ui.HTML(c.iframe(c.grafo_reglas_html(f, soporte_producto, 620, input.max_r()), 640))

    @render.ui
    def flujo():
        h, n = c.mapa_flujo_html(RP, soporte_producto, input.lift_flujo(), 650)
        return ui.TagList(ui.p(ui.HTML(f"Mostrando <b>{n}</b> reglas con lift ≥ {input.lift_flujo()}")),
                          ui.HTML(c.iframe(h, 670)))

    # ---------- 5. Cadena
    @reactive.calc
    def cadena():
        return c.cadena_de_compras(RP, input.inicio(), input.pasos(), input.criterio())

    @render.ui
    def cadena_texto():
        return ui.div(cadena()[3], class_="alert alert-success")

    @render.ui
    def cadena_grafo():
        camino, t, reg, _ = cadena()
        return ui.HTML(c.iframe(c.cadena_html(camino, t, reg), 320))

    @render.data_frame
    def cadena_tabla():
        return tabla(cadena()[1])

    @render.ui
    def cadena_comp():
        cols = []
        for cr in ["lift", "confidence"]:
            _, t, _, tx = c.cadena_de_compras(RP, input.inicio(), input.pasos(), cr)
            cols.append(ui.div(ui.p(ui.tags.b(f"{cr}: "), tx),
                               ui.HTML(c.redondear(t).to_html(index=False, classes="table table-sm"))))
        return ui.layout_columns(*cols)

    # ---------- 6. Recomendador
    @reactive.effect
    @reactive.event(input.ejemplo)
    def _ejemplo():
        if EJEMPLOS[input.ejemplo()]:
            ui.update_selectize("canasta", selected=EJEMPLOS[input.ejemplo()])

    @reactive.calc
    def reglas_rec():
        return reglas_rec_cache(float(input.sop_rec()), input.conf_rec())

    @reactive.calc
    def rec():
        if not input.canasta():
            return pd.DataFrame(columns=["producto", "motivo", "confianza", "lift"])
        return c.recomendar(list(input.canasta()), reglas_rec(), soporte_producto.index.tolist(),
                            input.top_rec(), input.ord_rec())

    @render.ui
    def n_reglas_rec():
        return ui.p(f"Reglas disponibles (un consecuente): {len(reglas_rec())}", class_="text-muted")

    @render.ui
    def rec_cards():
        r = rec()
        if r.empty:
            return None
        aviso = None
        if (r["motivo"] != "más vendido").sum() == 0:
            aviso = ui.div("Ninguna regla se activa (arranque en frío): se completan con los más vendidos.",
                           class_="alert alert-warning")
        cajas = [metrica(f"#{i + 1}", fila["producto"], "primary" if pd.notna(fila["lift"]) else "secondary",
                         f"lift {fila['lift']:.2f} · conf {fila['confianza']:.2f}" if pd.notna(fila["lift"]) else "más vendido")
                 for i, (_, fila) in enumerate(r.head(5).iterrows())]
        return ui.TagList(aviso, ui.layout_columns(*cajas))

    @render.data_frame
    def rec_tabla():
        return tabla(rec())

    # ---------- 7. Evaluación
    @reactive.calc
    def evaluacion():
        rt, pop, casos, ntr, nte = eval_cache()
        umbral = input.umbral_h() if input.hibrido() else None
        res = pd.DataFrame([c.hit_rate(N, casos, rt, pop, input.ord_ev(), umbral) for N in [1, 2, 3, 5, 7, 10]])
        res["Diferencia"] = res["HitRate reglas"] - res["HitRate más vendidos"]
        return res, (rt, casos, ntr, nte)

    @render.ui
    def eval_info():
        _, (rt, casos, ntr, nte) = evaluacion()
        return ui.p(f"Entrenamiento: {ntr} tickets · Prueba: {nte} · Casos (≥ 2 productos): {len(casos)} · "
                    f"Reglas de entrenamiento: {len(rt)}", class_="text-muted")

    @render.ui
    def fig_eval():
        res, _ = evaluacion()
        et = f"Híbrido (lift > {input.umbral_h()})" if input.hibrido() else f"Reglas ({input.ord_ev()})"
        return plot(c.fig_hitrate(res, et))

    @render.data_frame
    def tabla_eval():
        return tabla(evaluacion()[0])


PLOTLY_DIR = Path(plotly.__file__).parent / "package_data"
app = App(app_ui, server, static_assets={"/plotly": PLOTLY_DIR})
