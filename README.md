# Dashboard de Reglas de Asociación: Streamlit vs. Shiny for Python

El mismo dashboard (basado en la guía *Reglas de Asociación y Sistemas de Recomendación en Python*)
está construido dos veces, con la misma lógica (`core.py`), para comparar los dos frameworks.
Las dos versiones quedan listas para publicarse en **Hugging Face Spaces**.

```
dashboard_reglas/
├── streamlit_app/        → Space 1 (Docker, puerto 7860)
├── shiny_app/            → Space 2 (Docker, puerto 7860)
└── publicar_en_hf.py     → sube ambas carpetas como Spaces
```

## Pestañas (iguales en ambas apps)

| Pestaña | Sección de la guía | Qué se puede mover |
|---|---|---|
| 🧺 Ejemplo de clase | 1. Las 7 canastas | minsop, minconf, regla a calcular a mano |
| 🔎 Exploración | 2.1–2.3 | top-N productos, tamaño de la matriz de co-ocurrencias, comparar `X ⇒ {whole milk}` |
| 📏 Reglas | 2.4–2.8 | Apriori / FP-Growth, minsop, minconf, largo máximo, lift mínimo, consecuente, antecedente, Fisher + Holm, redundantes, descarga CSV |
| 🕸️ Grafos | 2.9–2.10 | grafo de reglas (pyvis), mapa de flujo producto → producto con lift mínimo |
| 🔗 Cadena de compras | 2.10 d | producto inicial, pasos, criterio lift / confidence |
| 🛍️ Recomendador | 3 | canasta del cliente, top N, orden, umbrales de minería |
| 📊 Evaluación | 4 + ejercicio 5 | HitRate@N por lift o confianza, híbrido conmutado con umbral de lift |

## Correr localmente

```bash
cd streamlit_app && pip install -r requirements.txt && streamlit run app.py
cd shiny_app     && pip install -r requirements.txt && shiny run app.py
```

## Publicar en Hugging Face

**Opción A, con el script** (crea los dos Spaces y sube los archivos):
```bash
pip install -U huggingface_hub
huggingface-cli login                 # token con permiso "write" (huggingface.co/settings/tokens)
python publicar_en_hf.py TU_USUARIO
```
Quedan en `huggingface.co/spaces/TU_USUARIO/reglas-asociacion-streamlit` y `.../reglas-asociacion-shiny`.

**Opción B, desde la web:** *New Space* → SDK **Docker** (Blank) → *Files* → *Upload files* →
arrastra el contenido de `streamlit_app/` (o `shiny_app/`), incluida la carpeta `data/`.
El `README.md` de cada carpeta ya trae la configuración del Space (`sdk: docker`, `app_port: 7860`).

La primera construcción tarda unos 3–5 minutos. El dataset va incluido en `data/`, así que no se depende de GitHub.

## Comparación

| Criterio | Streamlit | Shiny for Python |
|---|---|---|
| Modelo de ejecución | Re-ejecuta **todo el script** en cada interacción; se controla con `@st.cache_data` | **Reactivo**: solo se recalculan los outputs que dependen del input que cambió |
| Líneas de la app (sin `core.py`) | ~280 | ~460 (UI y servidor separados) |
| Curva de aprendizaje | Muy baja: se escribe como un notebook | Media: hay que entender `@reactive.calc`, `@render.*` e ids de inputs |
| Rendimiento con muchos controles | Puede sentirse más lento: cada clic re-ejecuta el script (mitigado por la caché) | Más fino: mover el slider del mapa de flujo no vuelve a minar reglas |
| Layout | Columnas y pestañas simples; poco control fino | Bootstrap completo: navbar, sidebar compartida, cards, value boxes |
| Tablas | `st.dataframe` con barras de progreso en la columna lift | `DataGrid` ordenable |
| Gráficos interactivos | `st.plotly_chart` nativo; pyvis vía `st.iframe` | Plotly embebido en HTML; pyvis vía iframe |
| Estado entre interacciones | `st.session_state` | Natural, por el grafo reactivo |
| Despliegue en HF Spaces | Docker (plantilla oficial) | Docker (plantilla oficial) |
| Para estudiantes | **Mejor para empezar**: prototipo rápido desde el notebook | **Mejor para apps más grandes** o con muchos usuarios simultáneos |

**Recomendación para el curso:** Streamlit para que los estudiantes conviertan su notebook en una app en una clase;
Shiny for Python para mostrar programación reactiva y apps más eficientes (y como puente con quienes ya usan Shiny en R).
