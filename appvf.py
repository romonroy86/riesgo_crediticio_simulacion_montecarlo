"""
App web para predecir incumplimiento de tarjeta de crédito y simular riesgo.
Consume el artefacto guardado en el notebook: modelo_incumplimiento_final.joblib
  artefacto = {'modelo': Pipeline, 'nombre_modelo': str, 'umbral': float}

Ejecutar:  python -m streamlit run appvf.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
RUTA_MODELO = Path(__file__).parent / "modelo_incumplimiento_final.joblib"

# Mismos recortes (percentil 99) que se aplicaron ANTES de entrenar
TOPE_INGRESO = 577_901.51
TOPE_LIMITE = 145_102.30

COLUMNAS = [
    "edad", "genero", "tiene_auto", "tiene_casa", "numero_hijos", "ingreso_anual_neto",
    "dias_empleado", "tipo_ocupacion", "total_miembros_familia", "trabajador_migrante",
    "pagos_anuales_deuda", "limite_credito", "porcentaje_limite_credito_utilizado",
    "puntaje_crediticio", "incumplimientos_previos",
]

st.set_page_config(page_title="Riesgo de incumplimiento", page_icon="💳", layout="wide")


# --------------------------------------------------------------------------
# Carga del modelo (una sola vez)
# --------------------------------------------------------------------------
@st.cache_resource
def cargar_artefacto():
    art = joblib.load(RUTA_MODELO)
    pipe = art["modelo"]
    # Las categorías válidas se leen del OneHotEncoder ya entrenado
    enc = pipe.named_steps["preprocessor"].named_transformers_["cat"]
    cat_cols = pipe.named_steps["preprocessor"].transformers_[0][2]
    categorias = {c: list(v) for c, v in zip(cat_cols, enc.categories_)}
    return pipe, float(art["umbral"]), art.get("nombre_modelo", "Random Forest"), categorias


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el mismo preprocesamiento manual que en el entrenamiento."""
    df = df[COLUMNAS].copy()
    df["ingreso_anual_neto"] = df["ingreso_anual_neto"].clip(upper=TOPE_INGRESO)
    df["limite_credito"] = df["limite_credito"].clip(upper=TOPE_LIMITE)
    return df


def nivel_riesgo(prob: float, umbral: float) -> tuple[str, str]:
    if prob >= umbral:
        return "ALTO", "🔴"
    if prob >= umbral * 0.6:
        return "MEDIO", "🟠"
    return "BAJO", "🟢"


# --------------------------------------------------------------------------
# Interfaz
# --------------------------------------------------------------------------
st.title("💳 Predicción de incumplimiento de tarjeta de crédito")

if not RUTA_MODELO.exists():
    st.error(f"No encuentro el modelo en:\n\n`{RUTA_MODELO}`\n\n"
             "Copia `modelo_incumplimiento_final.joblib` en la misma carpeta que `appvf.py`.")
    st.stop()

pipe, umbral, nombre_modelo, categorias = cargar_artefacto()
st.caption(f"Modelo: **{nombre_modelo}** · Umbral de decisión: **{umbral:.2f}** "
           "(un cliente se marca como *incumplidor* si su probabilidad ≥ umbral)")

# SOLUCIÓN DEL ERROR: Creamos las 4 variables de pestañas con sus nombres correctos
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Cliente individual", 
    "📂 Lote de clientes (CSV)", 
    "ℹ️ Sobre el modelo",
    "🎲 Simulación de Montecarlo"
])

# ---------------------------- Cliente individual (TAB 1) ---------------------------
with tab1:
    with st.form("form_cliente"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Datos personales")
            edad = st.number_input("Edad", 18, 80, 39)
            genero = st.selectbox("Género", categorias["genero"])
            numero_hijos = st.number_input("Número de hijos", 0, 15, 0)
            total_miembros_familia = st.number_input("Miembros de la familia", 1, 15, 2)
            trabajador_migrante = st.selectbox("Trabajador migrante", categorias["trabajador_migrante"])
        with c2:
            st.subheader("Situación económica")
            ingreso_anual_neto = st.number_input("Ingreso anual neto", 0.0, 5_000_000.0, 170_000.0, step=5_000.0)
            dias_empleado = st.number_input("Días empleado", 0, 30_000, 1_700)
            tipo_ocupacion = st.selectbox("Tipo de ocupación", categorias["tipo_ocupacion"])
            tiene_auto = st.selectbox("Tiene auto", categorias["tiene_auto"])
            tiene_casa = st.selectbox("Tiene casa", categorias["tiene_casa"])
        with c3:
            st.subheader("Perfil crediticio")
            limite_credito = st.number_input("Límite de crédito", 0.0, 1_000_000.0, 35_000.0, step=1_000.0)
            pct_uso = st.slider("% del límite utilizado", 0, 100, 54)
            pagos_anuales_deuda = st.number_input("Pagos anuales de deuda", 0.0, 500_000.0, 29_000.0, step=1_000.0)
            puntaje_crediticio = st.slider("Puntaje crediticio", 300, 950, 786)
            incumplimientos_previos = st.number_input("Incumplimientos previos", 0, 10, 0)
        enviar = st.form_submit_button("Calcular riesgo", type="primary", use_container_width=True)

    if enviar:
        fila = pd.DataFrame([{
            "edad": edad, "genero": genero, "tiene_auto": tiene_auto, "tiene_casa": tiene_casa,
            "numero_hijos": numero_hijos, "ingreso_anual_neto": ingreso_anual_neto,
            "dias_empleado": dias_empleado, "tipo_ocupacion": tipo_ocupacion,
            "total_miembros_familia": total_miembros_familia, "trabajador_migrante": trabajador_migrante,
            "pagos_anuales_deuda": pagos_anuales_deuda, "limite_credito": limite_credito,
            "porcentaje_limite_credito_utilizado": pct_uso, "puntaje_crediticio": puntaje_crediticio,
            "incumplimientos_previos": incumplimientos_previos,
        }])
        prob = float(pipe.predict_proba(preparar(fila))[0, 1])
        nivel, icono = nivel_riesgo(prob, umbral)

        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilidad de incumplimiento", f"{prob:.1%}")
        m2.metric("Nivel de riesgo", f"{icono} {nivel}")
        m3.metric("Decisión según umbral", "Incumplidor" if prob >= umbral else "No incumplidor")
        st.progress(min(prob, 1.0), text=f"Probabilidad {prob:.1%} · umbral {umbral:.0%}")
        if prob >= umbral:
            st.warning("Cliente de **alto riesgo**: revisar límite de crédito, tasa o aprobación.")
        else:
            st.success("Cliente por debajo del umbral de riesgo.")
        st.caption("La probabilidad proviene de un modelo con `class_weight='balanced'`; sirve para **ordenar** "
                   "clientes por riesgo, no debe leerse como probabilidad real calibrada. "
                   "Es una herramienta de apoyo, no sustituye el criterio del analista.")

# ------------------------------- Lote CSV (TAB 2) ----------------------------------
with tab2:
    st.write("Sube un CSV con estas columnas (los nombres deben coincidir exactamente):")
    st.code(", ".join(COLUMNAS), language=None)
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        datos = pd.read_csv(archivo)
        faltan = [c for c in COLUMNAS if c not in datos.columns]
        if faltan:
            st.error(f"Faltan columnas: {faltan}")
        else:
            probs = pipe.predict_proba(preparar(datos))[:, 1]
            salida = datos.copy()
            salida["prob_incumplimiento"] = probs.round(4)
            salida["decision"] = np.where(probs >= umbral, "Incumplidor", "No incumplidor")
            salida["nivel_riesgo"] = [nivel_riesgo(p, umbral)[0] for p in probs]

            a, b, c = st.columns(3)
            a.metric("Clientes evaluados", f"{len(salida):,}")
            b.metric("Marcados como incumplidores", f"{(probs >= umbral).sum():,}")
            c.metric("% marcados", f"{(probs >= umbral).mean():.1%}")
            st.bar_chart(salida["nivel_riesgo"].value_counts().reindex(["BAJO", "MEDIO", "ALTO"]).fillna(0))
            st.dataframe(salida.sort_values("prob_incumplimiento", ascending=False), use_container_width=True)
            st.download_button("⬇️ Descargar resultados", salida.to_csv(index=False).encode("utf-8"),
                               "predicciones.csv", "text/csv")

# ------------------------------ Sobre el modelo (TAB 3) ----------------------------
with tab3:
    st.markdown(f"""
**Modelo:** {nombre_modelo} (`class_weight='balanced'`, 300 árboles, profundidad máx. 10).
**Umbral:** {umbral:.2f}, elegido para equilibrar precisión y recall de la clase *incumple*.

**Limitaciones que debes comunicar**
- Los datos de entrenamiento provienen de Kaggle; el desempeño real puede ser menor.
- Las probabilidades no están calibradas.
- No debe usarse como única base para negar un crédito (revisar sesgos por edad, género, etc.).
- Reentrenar periódicamente para detectar *drift*.
""")

# ----------------------- Simulación de Montecarlo (TAB 4) -----------------------
with tab4:
    st.header("🎲 Proyección Temporal y Simulación de Pérdidas")
    st.write("Esta sección ejecuta simulaciones de Montecarlo utilizando el archivo CSV cargado en la pestaña anterior para proyectar la evolución de las pérdidas monetarias día por día.")

    if archivo is None:
        st.info("💡 Por favor, primero sube un archivo CSV válido en la pestaña **'📂 Lote de clientes (CSV)'** para poder ejecutar la simulación.")
    else:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            dias_simular = st.number_input("Días a simular (Horizonte)", min_value=1, max_value=365, value=30, step=1)
        with col_ctrl2:
            num_simulaciones = st.select_slider("Número de escenarios (Simulaciones)", options=[1000, 2000, 5000, 10000], value=5000)
        with col_ctrl3:
            lgd_input = st.slider("Severidad de Pérdida (LGD)", min_value=0.0, max_value=1.0, value=0.45, step=0.05, help="Porcentaje de la deuda que el banco da por perdido permanentemente tras el impago.")

        # CORRECCIÓN DE SANGRÍA: Este botón ahora sí responderá correctamente dentro del TAB 4
        if st.button("🚀 Ejecutar Simulación de Montecarlo", type="primary", use_container_width=True):
            archivo.seek(0)   # el archivo ya fue leído en la pestaña 2: hay que volver al inicio
            datos_sim = pd.read_csv(archivo)
            faltan_sim = [c for c in COLUMNAS if c not in datos_sim.columns]
            if faltan_sim:
                st.error(f"Faltan columnas en el CSV: {faltan_sim}")
                st.stop()
            pd_anuales = pipe.predict_proba(preparar(datos_sim))[:, 1]
            
            proporcion_tiempo = dias_simular / 365.0
            pd_horizonte_base = 1 - (1 - pd_anuales) ** proporcion_tiempo
            
            pd_mitigada_anual = np.where(pd_anuales >= umbral, pd_anuales * 0.05, pd_anuales)
            pd_horizonte_umbral = 1 - (1 - pd_mitigada_anual) ** proporcion_tiempo

            ead = datos_sim['limite_credito'].values * (datos_sim['porcentaje_limite_credito_utilizado'].values / 100.0)
            perdida_si_incumple = ead * lgd_input
            num_clientes_sim = len(datos_sim)

            perdidas_base_diarias = np.zeros((num_simulaciones, dias_simular + 1))
            perdidas_umbral_diarias = np.zeros((num_simulaciones, dias_simular + 1))

            rng = np.random.default_rng(422)
            with st.spinner("Lanzando dados matemáticos y calculando escenarios..."):
                for i in range(num_simulaciones):
                    aleatorios = rng.random(num_clientes_sim)
                    
                    incumple_base = aleatorios <= pd_horizonte_base
                    if np.any(incumple_base):
                        dias_evento = rng.integers(1, dias_simular + 1, size=np.sum(incumple_base))
                        p_mes = np.bincount(dias_evento, weights=perdida_si_incumple[incumple_base], minlength=dias_simular + 1)
                        perdidas_base_diarias[i, :] = np.cumsum(p_mes)
                        
                    incumple_umbral = aleatorios <= pd_horizonte_umbral
                    if np.any(incumple_umbral):
                        dias_evento_u = rng.integers(1, dias_simular + 1, size=np.sum(incumple_umbral))
                        p_mes_u = np.bincount(dias_evento_u, weights=perdida_si_incumple[incumple_umbral], minlength=dias_simular + 1)
                        perdidas_umbral_diarias[i, :] = np.cumsum(p_mes_u)

            esperada_base = perdidas_base_diarias.mean(axis=0)
            var99_base = np.percentile(perdidas_base_diarias, 99, axis=0)

            esperada_umbral = perdidas_umbral_diarias.mean(axis=0)
            var99_umbral = np.percentile(perdidas_umbral_diarias, 99, axis=0)
            
            ahorro_neto = esperada_base[-1] - esperada_umbral[-1]

            st.subheader("📊 Resultados Financieros de la Cartera Simulada")
            m_sim1, m_sim2, m_sim3 = st.columns(3)
            m_sim1.metric("Pérdida Esperada (BASE)", f"Q{esperada_base[-1]:,.2f}")
            m_sim2.metric("Pérdida Esperada (CON UMBRAL 0.75)", f"Q{esperada_umbral[-1]:,.2f}")
            m_sim3.metric("Ahorro Estimado por Mitigación", f"Q{ahorro_neto:,.2f}", delta=f"-{(1-esperada_umbral[-1]/esperada_base[-1]):.1%} Pérdida")

            m_sim4, m_sim5 = st.columns(2)
            m_sim4.metric("Capital de Reserva en Crisis (VaR 99% Base)", f"Q{var99_base[-1]:,.2f}")
            m_sim5.metric("Capital de Reserva en Crisis (VaR 99% Umbral)", f"Q{var99_umbral[-1]:,.2f}")

            st.subheader("📈 Evolución Temporal de las Pérdidas Acumuladas")
            fig, ax = plt.subplots(figsize=(11, 5.5))
            dias_eje_x = np.arange(0, dias_simular + 1)

            muestra_escenarios = rng.choice(num_simulaciones, size=30, replace=False)
            for esc in muestra_escenarios:
                ax.plot(dias_eje_x, perdidas_base_diarias[esc, :], color='gray', alpha=0.1, linewidth=1)

            ax.plot(dias_eje_x, esperada_base, color='#C62828', linewidth=2.5, label=f'Pérdida Esperada BASE (Cierre: Q{esperada_base[-1]:,.0f})')
            ax.plot(dias_eje_x, var99_base, color='#5D4037', linewidth=1.5, linestyle='--', label=f'VaR 99% BASE (Cierre: Q{var99_base[-1]:,.0f})')
            
            ax.plot(dias_eje_x, esperada_umbral, color='#2E7D32', linewidth=2.5, label=f'Pérdida Esperada con Umbral 0.75 (Cierre: Q{esperada_umbral[-1]:,.0f})')
            ax.plot(dias_eje_x, var99_umbral, color='#1B5E20', linewidth=1.5, linestyle='-.', label=f'VaR 99% con Umbral 0.75 (Cierre: Q{var99_umbral[-1]:,.0f})')

            ax.fill_between(dias_eje_x, esperada_base, esperada_umbral, color='#2E7D32', alpha=0.15, label=f'Zona de Ahorro Neto (Q{ahorro_neto:,.0f})')

            ax.set_title(f'Simulación de Montecarlo: Pérdidas Acumuladas a {dias_simular} Días', fontsize=12, fontweight='bold')
            ax.set_xlabel('Días Proyectados', fontsize=10)
            ax.set_ylabel('Pérdida Acumulada del Portafolio (Q)', fontsize=10)
            ax.set_xlim(0, dias_simular)
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            ax.grid(True, alpha=0.25, linestyle=':')
            ax.legend(loc='upper left', fontsize=9)
            
            st.pyplot(fig)
