"""
================================================================================
APLICACIÓN WEB INTERACTIVA - CLASIFICADOR DE EXOPLANETAS
================================================================================

Aplicación desarrollada con Streamlit para desplegar el modelo de Machine
Learning de clasificación de exoplanetas candidatos.

Características:
- Interfaz intuitiva para ingresar datos
- Predicción en tiempo real
- Visualización de resultados
- Análisis de confianza
- Explicabilidad del modelo

Autor: Leonardo Daniel Aviña Neri
Framework: CRISP-ML(Q)
Dataset: NASA Kepler Exoplanet Search Results
================================================================================
"""

# ==============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================

st.set_page_config(
    page_title="Clasificador de Exoplanetas",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

@st.cache_resource
def load_model_and_artifacts():
    """
    Carga el modelo entrenado y los artefactos necesarios.
    Se cachea para no recargar en cada interacción.
    """
    try:
        # Obtener rutas - usar rutas relativas para compatibilidad con Streamlit Cloud
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Rutas relativas al directorio de la app
        models_dir = os.path.join(base_dir, 'models')
        processed_dir = os.path.join(base_dir, 'db', 'processed')
        
        # Cargar modelo
        model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
        
        # Cargar transformadores
        scaler = joblib.load(os.path.join(processed_dir, 'scaler.pkl'))
        label_encoder = joblib.load(os.path.join(processed_dir, 'label_encoder.pkl'))
        
        # Cargar metadata (el archivo se llama preprocessing_metadata.json)
        with open(os.path.join(processed_dir, 'preprocessing_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Cargar nombres de features desde X_train
        X_train_sample = pd.read_csv(os.path.join(processed_dir, 'X_train_processed.csv'), nrows=1)
        feature_names = X_train_sample.columns.tolist()
        metadata['feature_names'] = feature_names
        
        # Cargar resultados de evaluación
        eval_dir = os.path.join(models_dir, 'evaluation')
        with open(os.path.join(eval_dir, 'evaluation_report.json'), 'r') as f:
            evaluation = json.load(f)
        
        return {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'metadata': metadata,
            'evaluation': evaluation
        }
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.info("Asegúrate de haber ejecutado fase2.py, fase3.py y fase4.py primero.")
        return None

def create_input_features(user_input, expected_features):
    """
    Transforma los inputs del usuario al formato esperado por el modelo.
    Aplica la misma ingeniería de features que en entrenamiento.
    """
    # Crear diccionario con todas las features esperadas inicializadas en 0
    features = {col: 0.0 for col in expected_features}
    
    # Asignar valores directos de los inputs del usuario
    direct_features = [
        'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 
        'koi_fpflag_ec', 'koi_period', 'koi_time0bk', 'koi_impact', 
        'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 
        'koi_model_snr', 'koi_tce_plnt_num', 'koi_steff', 'koi_slogg', 
        'koi_srad', 'ra', 'dec', 'koi_kepmag'
    ]
    
    for feat in direct_features:
        if feat in user_input:
            features[feat] = float(user_input[feat])
    
    # Features calculadas - logaritmos
    if user_input.get('koi_period', 0) > 0:
        features['log_koi_period'] = np.log1p(user_input['koi_period'])
    
    if user_input.get('koi_depth', 0) > 0:
        features['log_koi_depth'] = np.log1p(user_input['koi_depth'])
    
    if user_input.get('koi_prad', 0) > 0:
        features['log_koi_prad'] = np.log1p(user_input['koi_prad'])
    
    if user_input.get('koi_insol', 0) > 0:
        features['log_koi_insol'] = np.log1p(user_input['koi_insol'])
    
    # Features calculadas - razones
    if user_input.get('koi_period', 0) > 0:
        features['duration_period_ratio'] = user_input.get('koi_duration', 0) / user_input['koi_period']
    
    if user_input.get('koi_period', 0) > 0 and user_input.get('koi_depth', 0) > 0:
        features['depth_period_ratio'] = user_input['koi_depth'] / user_input['koi_period']
    
    if user_input.get('koi_srad', 0) > 0 and user_input.get('koi_prad', 0) > 0:
        features['planet_star_radius_ratio'] = user_input['koi_prad'] / user_input['koi_srad']
    
    # Composite confidence score
    flags = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
    features['composite_confidence_score'] = sum(user_input.get(f, 0) for f in flags)
    
    # Extreme temperature
    if user_input.get('koi_teq', 0) > 500:
        features['extreme_temperature'] = 1.0
    
    # koi_pdisposition - one-hot encoding
    if user_input.get('koi_pdisposition') == 'CANDIDATE':
        features['koi_pdisposition_CANDIDATE'] = 1.0
        features['koi_pdisposition_FALSE POSITIVE'] = 0.0
    else:  # FALSE POSITIVE
        features['koi_pdisposition_CANDIDATE'] = 0.0
        features['koi_pdisposition_FALSE POSITIVE'] = 1.0
    
    # Planet size category - one-hot encoding
    koi_prad = user_input.get('koi_prad', 0)
    if koi_prad < 1.25:
        features['planet_size_category_Terrestre'] = 1.0
    elif koi_prad < 2.0:
        features['planet_size_category_SuperTierra'] = 1.0
    elif koi_prad < 6.0:
        features['planet_size_category_Neptuniano'] = 1.0
    else:
        features['planet_size_category_Joviano'] = 1.0
    
    # Habitable zone - one-hot encoding
    koi_insol = user_input.get('koi_insol', 0)
    if 0.25 <= koi_insol <= 2.0:
        features['habitable_zone_Yes'] = 1.0
        features['habitable_zone_No'] = 0.0
    else:
        features['habitable_zone_Yes'] = 0.0
        features['habitable_zone_No'] = 1.0
    
    # Orbital period category - one-hot encoding
    period = user_input.get('koi_period', 0)
    if period < 10:
        features['orbital_period_category_Muy_Corto'] = 1.0
    elif period < 50:
        features['orbital_period_category_Corto'] = 1.0
    elif period < 200:
        features['orbital_period_category_Medio'] = 1.0
    else:
        features['orbital_period_category_Largo'] = 1.0
    
    # Convertir a DataFrame con el orden correcto de columnas
    df = pd.DataFrame([features])[expected_features]
    
    return df

def make_prediction(artifacts, input_data):
    """
    Realiza la predicción usando el modelo cargado.
    """
    try:
        # Obtener features esperadas
        expected_features = artifacts['metadata']['feature_names']
        
        # Crear features
        features_df = create_input_features(input_data, expected_features)
        
        # Separar features numéricas y categóricas (one-hot)
        # Las one-hot son las que terminan con nombres de categorías
        onehot_patterns = ['_CANDIDATE', '_FALSE POSITIVE', '_Terrestre', '_SuperTierra', 
                          '_Neptuniano', '_Joviano', '_No', '_Yes', '_Muy_Corto', '_Corto', 
                          '_Medio', '_Largo']
        
        categorical_features = [col for col in expected_features 
                               if any(col.endswith(pattern) for pattern in onehot_patterns)]
        numeric_features = [col for col in expected_features if col not in categorical_features]
        
        # Crear copia para escalar
        features_scaled = features_df.copy()
        
        # Escalar solo las features numéricas
        features_scaled[numeric_features] = artifacts['scaler'].transform(
            features_df[numeric_features]
        )
        
        # Predecir
        prediction = artifacts['model'].predict(features_scaled)[0]
        probabilities = artifacts['model'].predict_proba(features_scaled)[0]
        
        # Decodificar la clase predicha
        predicted_class = artifacts['label_encoder'].inverse_transform([prediction])[0]
        
        # Crear diccionario de probabilidades por clase
        class_probabilities = {}
        for i, class_name in enumerate(artifacts['label_encoder'].classes_):
            class_probabilities[class_name] = float(probabilities[i])
        
        return {
            'predicted_class': predicted_class,
            'probabilities': class_probabilities,
            'confidence': float(np.max(probabilities))
        }
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# ==============================================================================
# INTERFAZ DE LA APLICACIÓN
# ==============================================================================

def main():
    """
    Función principal de la aplicación.
    """
    
    # Título principal
    st.title("🪐 Clasificador de Exoplanetas")
    st.markdown("### Sistema de Machine Learning para clasificación de candidatos a exoplanetas")
    st.markdown("---")
    
    # Cargar modelo y artefactos
    artifacts = load_model_and_artifacts()
    
    if artifacts is None:
        st.stop()
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información del Proyecto")
        st.markdown("""
        **Proyecto:** Clasificación de Exoplanetas
        
        **Framework:** CRISP-ML(Q)
        
        **Dataset:** NASA Kepler
        
        **Modelo:** XGBoost
        
        **Métricas:**
        - Accuracy: 91.53%
        - F1-Score: 88.59%
        - AUC-ROC: 98.12%
        """)
        
        st.markdown("---")
        
        st.header("📊 Clases")
        st.markdown("""
        - **CANDIDATE**: Candidato a exoplaneta
        - **CONFIRMED**: Exoplaneta confirmado
        - **FALSE POSITIVE**: Falso positivo
        """)
        
        st.markdown("---")
        
        st.header("🎯 Acerca de")
        st.markdown("""
        Esta aplicación utiliza Machine Learning
        para clasificar objetos detectados por el
        telescopio espacial Kepler.
        
        Desarrollado como parte del proyecto final
        de Estadística Inferencial.
        """)
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["🔮 Predicción", "📈 Rendimiento del Modelo", "📚 Documentación"])
    
    # ==============================================================================
    # TAB 1: PREDICCIÓN
    # ==============================================================================
    
    with tab1:
        st.header("Ingresa los datos del objeto celestial")
        
        # Crear columnas para organizar inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📏 Parámetros Orbitales")
            
            koi_period = st.number_input(
                "Período Orbital (días)",
                min_value=0.0,
                max_value=1000.0,
                value=10.0,
                help="Tiempo que tarda el objeto en completar una órbita"
            )
            
            koi_duration = st.number_input(
                "Duración del Tránsito (horas)",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                help="Duración del tránsito frente a la estrella"
            )
            
            koi_time0bk = st.number_input(
                "Tiempo de Tránsito (BJD)",
                min_value=0.0,
                max_value=200000.0,
                value=131.5,
                help="Tiempo del primer tránsito observado"
            )
            
            koi_impact = st.number_input(
                "Parámetro de Impacto",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                help="Distancia mínima al centro estelar durante el tránsito"
            )
            
            koi_depth = st.number_input(
                "Profundidad del Tránsito (ppm)",
                min_value=0.0,
                max_value=100000.0,
                value=100.0,
                help="Disminución de brillo de la estrella durante el tránsito"
            )
            
            koi_prad = st.number_input(
                "Radio del Planeta (radios terrestres)",
                min_value=0.0,
                max_value=50.0,
                value=2.0,
                help="Radio estimado del planeta en radios terrestres"
            )
            
            koi_insol = st.number_input(
                "Insolación (flux terrestres)",
                min_value=0.0,
                max_value=1000.0,
                value=1.0,
                help="Flujo de radiación recibido del sol"
            )
            
            koi_teq = st.number_input(
                "Temperatura de Equilibrio (K)",
                min_value=0,
                max_value=3000,
                value=288,
                help="Temperatura de equilibrio del planeta (Tierra ≈ 288K)"
            )
        
        with col2:
            st.subheader("⭐ Parámetros Estelares")
            
            koi_steff = st.number_input(
                "Temperatura Efectiva Estelar (K)",
                min_value=2000,
                max_value=10000,
                value=5778,
                help="Temperatura de la estrella anfitriona (Sol ≈ 5778K)"
            )
            
            koi_slogg = st.number_input(
                "Gravedad Superficial Estelar (log g)",
                min_value=2.0,
                max_value=5.0,
                value=4.4,
                help="Logaritmo de la gravedad superficial (Sol ≈ 4.4)"
            )
            
            koi_srad = st.number_input(
                "Radio Estelar (radios solares)",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                help="Radio de la estrella en radios solares"
            )
            
            koi_kepmag = st.number_input(
                "Magnitud Kepler",
                min_value=8.0,
                max_value=20.0,
                value=15.0,
                help="Magnitud aparente en banda Kepler"
            )
            
            st.subheader("📍 Coordenadas")
            
            ra = st.number_input(
                "Ascensión Recta (grados)",
                min_value=0.0,
                max_value=360.0,
                value=290.0,
                help="Coordenada de ascensión recta"
            )
            
            dec = st.number_input(
                "Declinación (grados)",
                min_value=-90.0,
                max_value=90.0,
                value=45.0,
                help="Coordenada de declinación"
            )
            
            st.subheader("🎯 Métricas de Calidad")
            
            koi_score = st.slider(
                "Score de Disposición",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="Puntuación de confianza del pipeline de Kepler"
            )
            
            koi_model_snr = st.number_input(
                "Relación Señal/Ruido",
                min_value=0.0,
                max_value=1000.0,
                value=10.0,
                help="Relación señal-ruido del modelo de tránsito"
            )
        
        st.markdown("---")
        
        # Sección de flags
        st.subheader("🚩 Flags de Falso Positivo")
        
        col_flag1, col_flag2, col_flag3, col_flag4 = st.columns(4)
        
        with col_flag1:
            koi_fpflag_nt = st.checkbox("Not Transit-Like", value=False, 
                                        help="Señal no es similar a un tránsito")
        
        with col_flag2:
            koi_fpflag_ss = st.checkbox("Stellar Eclipse", value=False,
                                        help="Posible eclipse estelar")
        
        with col_flag3:
            koi_fpflag_co = st.checkbox("Centroid Offset", value=False,
                                        help="Offset en el centroide")
        
        with col_flag4:
            koi_fpflag_ec = st.checkbox("Ephemeris Match", value=False,
                                        help="Coincidencia de efemérides")
        
        st.markdown("---")
        
        # Disposición preliminar del pipeline de Kepler
        st.subheader("🔍 Disposición Preliminar de Kepler")
        
        koi_pdisposition = st.selectbox(
            "Clasificación del pipeline de Kepler",
            options=["CANDIDATE", "FALSE POSITIVE"],
            help="Clasificación preliminar del pipeline automático de Kepler"
        )
        
        st.markdown("---")
        
        # Número de planetas en el sistema
        koi_tce_plnt_num = st.number_input(
            "Número de planeta en el sistema",
            min_value=1,
            max_value=10,
            value=1,
            help="Número de planeta en el sistema (1 = primer planeta detectado)"
        )
        
        st.markdown("---")
        
        # Botón de predicción
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn2:
            predict_button = st.button("🚀 CLASIFICAR OBJETO", use_container_width=True, type="primary")
        
        # Realizar predicción
        if predict_button:
            with st.spinner("Clasificando objeto celestial..."):
                # Preparar datos de entrada
                input_data = {
                    'koi_period': koi_period,
                    'koi_time0bk': koi_time0bk,
                    'koi_impact': koi_impact,
                    'koi_duration': koi_duration,
                    'koi_depth': koi_depth,
                    'koi_prad': koi_prad,
                    'koi_insol': koi_insol,
                    'koi_teq': koi_teq,
                    'koi_steff': koi_steff,
                    'koi_slogg': koi_slogg,
                    'koi_srad': koi_srad,
                    'koi_kepmag': koi_kepmag,
                    'ra': ra,
                    'dec': dec,
                    'koi_score': koi_score,
                    'koi_model_snr': koi_model_snr,
                    'koi_fpflag_nt': int(koi_fpflag_nt),
                    'koi_fpflag_ss': int(koi_fpflag_ss),
                    'koi_fpflag_co': int(koi_fpflag_co),
                    'koi_fpflag_ec': int(koi_fpflag_ec),
                    'koi_pdisposition': koi_pdisposition,
                    'koi_tce_plnt_num': koi_tce_plnt_num
                }
                
                # Hacer predicción
                result = make_prediction(artifacts, input_data)
                
                if result:
                    st.markdown("---")
                    st.success("✅ Clasificación completada!")
                    
                    # Mostrar resultado principal
                    st.markdown("### 🎯 Resultado de la Clasificación")
                    
                    # Color según la clase
                    class_colors = {
                        'CANDIDATE': '🟡',
                        'CONFIRMED': '🟢',
                        'FALSE POSITIVE': '🔴'
                    }
                    
                    emoji = class_colors.get(result['predicted_class'], '⚪')
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                        <h1 style='color: #1f77b4; margin: 0;'>{emoji} {result['predicted_class']}</h1>
                        <p style='font-size: 18px; color: #555; margin-top: 10px;'>
                            Confianza: {result['confidence']*100:.2f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Gráfico de probabilidades
                    col_graph1, col_graph2 = st.columns(2)
                    
                    with col_graph1:
                        st.markdown("### 📊 Probabilidades por Clase")
                        
                        # Crear gráfico de barras
                        fig_bar = go.Figure(data=[
                            go.Bar(
                                x=list(result['probabilities'].keys()),
                                y=list(result['probabilities'].values()),
                                marker_color=['#FFA500', '#4CAF50', '#F44336'],
                                text=[f"{v*100:.1f}%" for v in result['probabilities'].values()],
                                textposition='auto',
                            )
                        ])
                        
                        fig_bar.update_layout(
                            xaxis_title="Clase",
                            yaxis_title="Probabilidad",
                            yaxis_range=[0, 1],
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with col_graph2:
                        st.markdown("### 🥧 Distribución de Confianza")
                        
                        # Crear gráfico de pie
                        fig_pie = go.Figure(data=[
                            go.Pie(
                                labels=list(result['probabilities'].keys()),
                                values=list(result['probabilities'].values()),
                                marker_colors=['#FFA500', '#4CAF50', '#F44336'],
                                hole=0.4,
                                textinfo='label+percent',
                                hovertemplate='<b>%{label}</b><br>Probabilidad: %{value:.4f}<extra></extra>'
                            )
                        ])
                        
                        fig_pie.update_layout(
                            height=400,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Interpretación del resultado
                    st.markdown("---")
                    st.markdown("### 💡 Interpretación")
                    
                    if result['predicted_class'] == 'CONFIRMED':
                        st.success("""
                        **Exoplaneta Confirmado**: El modelo ha clasificado este objeto como un exoplaneta
                        confirmado con alta confianza. Esto significa que las características observadas
                        son consistentes con un planeta real orbitando la estrella.
                        """)
                    elif result['predicted_class'] == 'CANDIDATE':
                        st.warning("""
                        **Candidato a Exoplaneta**: El objeto muestra características compatibles con
                        un exoplaneta, pero requiere observaciones adicionales para su confirmación.
                        Se recomienda seguimiento observacional.
                        """)
                    else:
                        st.error("""
                        **Falso Positivo**: El modelo ha determinado que este objeto probablemente no
                        es un exoplaneta. Podría ser un eclipse estelar binario, variabilidad estelar,
                        o un artefacto instrumental.
                        """)
                    
                    # Mostrar valores de entrada
                    with st.expander("📋 Ver datos de entrada"):
                        st.json(input_data)
    
    # ==============================================================================
    # TAB 2: RENDIMIENTO DEL MODELO
    # ==============================================================================
    
    with tab2:
        st.header("📈 Rendimiento del Modelo en Evaluación")
        
        eval_results = artifacts['evaluation']
        
        # Métricas principales
        st.subheader("🎯 Métricas Globales")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric(
                "Accuracy",
                f"{eval_results['metrics']['accuracy']*100:.2f}%",
                help="Porcentaje de predicciones correctas"
            )
        
        with col_m2:
            st.metric(
                "F1-Score",
                f"{eval_results['metrics']['f1_macro']:.4f}",
                help="Media armónica de precisión y recall"
            )
        
        with col_m3:
            st.metric(
                "AUC-ROC",
                f"{eval_results['roc_auc_macro']:.4f}",
                help="Área bajo la curva ROC"
            )
        
        with col_m4:
            st.metric(
                "Kappa de Cohen",
                f"{eval_results['metrics']['cohen_kappa']:.4f}",
                help="Medida de concordancia"
            )
        
        st.markdown("---")
        
        # Métricas por clase
        st.subheader("📊 Métricas por Clase")
        
        metrics_df = pd.DataFrame(eval_results['metrics_per_class']).T
        metrics_df = metrics_df[['precision', 'recall', 'f1-score', 'support']]
        metrics_df.columns = ['Precision', 'Recall', 'F1-Score', 'Muestras']
        
        # Formatear
        for col in ['Precision', 'Recall', 'F1-Score']:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.4f}")
        metrics_df['Muestras'] = metrics_df['Muestras'].astype(int)
        
        st.dataframe(metrics_df, use_container_width=True)
        
        st.markdown("---")
        
        # Validación cruzada
        st.subheader("🔄 Validación Cruzada (5-Fold)")
        
        cv_results = eval_results['robustness']['cross_validation']
        
        col_cv1, col_cv2 = st.columns(2)
        
        with col_cv1:
            st.metric(
                "F1-Score Promedio",
                f"{cv_results['mean_f1']:.4f}",
                help="Media del F1-Score en los 5 folds"
            )
        
        with col_cv2:
            st.metric(
                "Desviación Estándar",
                f"{cv_results['std_f1']:.4f}",
                help="Estabilidad del modelo (menor es mejor)"
            )
        
        # Gráfico de scores por fold
        fig_cv = go.Figure()
        
        fig_cv.add_trace(go.Scatter(
            x=list(range(1, 6)),
            y=cv_results['scores'],
            mode='lines+markers',
            marker=dict(size=10, color='#1f77b4'),
            line=dict(width=2),
            name='F1-Score'
        ))
        
        # Línea de promedio
        fig_cv.add_hline(
            y=cv_results['mean_f1'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Media: {cv_results['mean_f1']:.4f}"
        )
        
        fig_cv.update_layout(
            title="F1-Score por Fold",
            xaxis_title="Fold",
            yaxis_title="F1-Score",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_cv, use_container_width=True)
        
        st.markdown("---")
        
        # Análisis de confianza
        st.subheader("🎯 Análisis de Confianza")
        
        conf_analysis = eval_results['confidence_analysis']
        
        col_conf1, col_conf2, col_conf3 = st.columns(3)
        
        with col_conf1:
            st.metric(
                "Confianza Promedio",
                f"{conf_analysis['mean']*100:.2f}%",
                help="Confianza promedio en las predicciones"
            )
        
        with col_conf2:
            st.metric(
                "Desv. Estándar",
                f"{conf_analysis['std']:.4f}",
                help="Variabilidad en la confianza"
            )
        
        with col_conf3:
            st.metric(
                "Baja Confianza",
                f"{conf_analysis['low_confidence_pct']:.2f}%",
                help="% de predicciones con confianza < 60%"
            )
        
        st.markdown("---")
        
        # Criterios de aceptación
        st.subheader("✅ Criterios de Aceptación")
        
        acceptance = eval_results['acceptance_criteria']
        
        criteria_data = {
            'Criterio': [
                'F1-Score ≥ 0.85',
                'Accuracy ≥ 0.90',
                'AUC-ROC ≥ 0.90',
                'Estabilidad CV',
                'Sin Overfitting'
            ],
            'Estado': [
                '✅ Aprobado' if acceptance['f1_threshold_0.85'] else '❌ No cumple',
                '✅ Aprobado' if acceptance['accuracy_threshold_0.90'] else '⚠️ Revisar',
                '✅ Aprobado' if acceptance.get('auc_threshold_0.90', False) else '⚠️ Revisar',
                '✅ Aprobado' if acceptance['cv_stability'] else '⚠️ Revisar',
                '✅ Aprobado' if acceptance['overfitting_check'] else '⚠️ Revisar'
            ]
        }
        
        st.table(pd.DataFrame(criteria_data))
    
    # ==============================================================================
    # TAB 3: DOCUMENTACIÓN
    # ==============================================================================
    
    with tab3:
        st.header("📚 Documentación del Proyecto")
        
        st.markdown("""
        ## 🎯 Objetivo del Proyecto
        
        Desarrollar un sistema de Machine Learning para clasificar automáticamente
        candidatos a exoplanetas detectados por el telescopio espacial Kepler de la NASA.
        
        ## 📊 Dataset
        
        - **Fuente**: NASA Kepler Exoplanet Search Results
        - **Registros**: 9,564 objetos celestiales
        - **Features**: 49 variables → 23 seleccionadas → 42 después de feature engineering
        - **Clases**: 
          - CANDIDATE: Candidato a exoplaneta (requiere confirmación)
          - CONFIRMED: Exoplaneta confirmado
          - FALSE POSITIVE: Falso positivo (no es un exoplaneta)
        
        ## 🔬 Metodología: CRISP-ML(Q)
        
        ### Fase 1: Comprensión del Negocio y Datos
        - Definición de objetivos
        - Análisis exploratorio de datos
        - Verificación de calidad
        
        ### Fase 2: Ingeniería de Datos
        - Selección de features (análisis de correlación, importancia)
        - Limpieza de datos (imputación de valores faltantes)
        - Tratamiento de outliers (winsorización)
        - Feature engineering (12 nuevas variables)
        - Balanceo de clases (SMOTE)
        - Estandarización (StandardScaler)
        
        ### Fase 3: Ingeniería del Modelo
        - Entrenamiento de 4 modelos:
          - Regresión Logística
          - Árbol de Decisión
          - Random Forest
          - XGBoost ⭐ (mejor modelo)
        - Validación cruzada estratificada (5-fold)
        - Optimización de hiperparámetros
        
        ### Fase 4: Evaluación del Modelo
        - Evaluación en conjunto de test
        - Análisis de robustez (pruebas con ruido)
        - Análisis de explicabilidad
        - Decisión de despliegue: ✅ APROBADO
        
        ### Fase 5: Despliegue
        - Aplicación web interactiva (Streamlit)
        - Interfaz amigable para usuarios
        - Visualización de resultados
        - Plan de monitoreo y mantenimiento
        
        ## 🎓 Características del Modelo
        
        **Algoritmo**: XGBoost (Gradient Boosting)
        
        **Rendimiento**:
        - Accuracy: 91.53%
        - F1-Score: 88.59%
        - AUC-ROC: 98.12%
        - Validación cruzada: 93.27% (±0.28%)
        
        **Ventajas**:
        - Alta precisión en las tres clases
        - Excelente discriminación (AUC-ROC > 98%)
        - Modelo robusto y estable
        - Interpretable (importancia de features)
        
        ## 📈 Features Más Importantes
        
        1. **koi_pdisposition**: Disposición preliminar de Kepler (68%)
        2. **composite_confidence_score**: Score compuesto de flags (6.8%)
        3. **koi_fpflag_nt**: Flag "not transit-like" (4.7%)
        4. **koi_score**: Score de disposición (2.6%)
        5. **koi_fpflag_co**: Flag "centroid offset" (1.7%)
        
        ## 🚀 Cómo Usar la Aplicación
        
        1. **Ingresa los datos** del objeto celestial en el formulario
        2. **Ajusta los parámetros** orbitales y estelares
        3. **Marca los flags** de falso positivo si aplican
        4. **Presiona "Clasificar"** para obtener la predicción
        5. **Revisa los resultados** y la confianza del modelo
        
        ## 📞 Contacto y Soporte
        
        - **Autor**: Leonardo Daniel Aviña Neri
        - **Email**: ld.avinaneri@ugto.mx
        - **Universidad**: Universidad de Guanajuato
        - **Curso**: Estadística Inferencial
        
        ## 📄 Referencias
        
        - NASA Kepler Mission: https://www.nasa.gov/mission_pages/kepler/main/index.html
        - CRISP-ML(Q): https://ml-ops.org/content/crisp-ml
        - XGBoost: https://xgboost.readthedocs.io/
        - Streamlit: https://streamlit.io/
        
        ## ⚖️ Licencia
        
        Proyecto académico desarrollado con fines educativos.
        Dataset público de NASA Kepler Mission.
        """)
        
        st.markdown("---")
        
        # Información adicional
        with st.expander("🔍 Ver información técnica del modelo"):
            tech_info = {
                "model_name": artifacts['evaluation']['model_name'],
                "model_type": artifacts['evaluation']['model_type'],
                "timestamp": artifacts['evaluation']['timestamp'],
                "test_set_size": artifacts['evaluation']['test_set_size'],
                "classes": list(artifacts['label_encoder'].classes_)
            }
            # Agregar feature_count solo si existe
            if 'feature_names' in artifacts['metadata']:
                tech_info["feature_count"] = len(artifacts['metadata']['feature_names'])
            else:
                tech_info["feature_count"] = artifacts['metadata'].get('n_features', 'N/A')
            
            st.json(tech_info)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>🪐 Exoplanet Classifier v1.0</p>
        <p>© 2025 | Universidad de Guanajuato | Estadística Inferencial</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# EJECUTAR APLICACIÓN
# ==============================================================================

if __name__ == "__main__":
    main()
