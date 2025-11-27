# 🪐 Clasificador de Exoplanetas - Streamlit App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

Sistema de Machine Learning para clasificación automática de candidatos a exoplanetas detectados por el telescopio espacial Kepler de la NASA.

## 🎯 Características

- **Predicción en tiempo real** de clasificación de exoplanetas
- **Interfaz intuitiva** para ingresar parámetros astronómicos
- **Visualización interactiva** de resultados y probabilidades
- **Métricas del modelo** y análisis de rendimiento
- **Documentación completa** del proyecto y metodología

## 📊 Rendimiento del Modelo

- **Accuracy:** 91.53%
- **F1-Score:** 88.59%
- **AUC-ROC:** 98.12%
- **Modelo:** XGBoost Classifier
- **Framework:** CRISP-ML(Q)

## 🚀 Uso Local

### Prerrequisitos

- Python 3.9 o superior
- pip

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app_streamlit.py
```

La aplicación se abrirá en `http://localhost:8501`

## 📂 Estructura del Proyecto

```
.
├── app_streamlit.py          # Aplicación principal
├── requirements.txt          # Dependencias
├── models/                   # Modelo entrenado
│   ├── best_model.pkl
│   └── evaluation/
│       └── evaluation_report.json
├── db/                       # Datos y preprocessors
│   └── processed/
│       ├── scaler.pkl
│       ├── label_encoder.pkl
│       ├── preprocessing_metadata.json
│       └── X_train_processed.csv
└── README.md
```

## 🔬 Metodología

El proyecto sigue el framework **CRISP-ML(Q)** (Cross-Industry Standard Process for Machine Learning with Quality assurance):

1. **Fase 1:** Comprensión del Negocio y Datos
2. **Fase 2:** Preparación de Datos (Feature Engineering, SMOTE, Normalización)
3. **Fase 3:** Modelado (XGBoost, Random Forest, Decision Tree, Logistic Regression)
4. **Fase 4:** Evaluación (Validación cruzada, pruebas de robustez)
5. **Fase 5:** Despliegue (Aplicación Streamlit)

## 📈 Dataset

- **Fuente:** NASA Kepler Exoplanet Search Results
- **Registros:** 9,564 objetos celestiales
- **Features:** 42 características (22 originales + 20 engineered)
- **Clases:**
  - CANDIDATE: Candidato a exoplaneta
  - CONFIRMED: Exoplaneta confirmado
  - FALSE POSITIVE: Falso positivo

## 🎓 Autores

- **Leonardo Daniel Aviña Neri**
- **Ricardo Ignacio Perez Mendoza**
- **Melchor Alexander Araiza Zavala**

**Universidad de Guanajuato**  
Licenciatura en Ingeniería de Datos e Inteligencia Artificial  
Estadística Inferencial - 2025

## 📄 Licencia

Proyecto académico desarrollado con fines educativos.  
Dataset público de NASA Kepler Mission.

## 🔗 Referencias

- [NASA Kepler Mission](https://www.nasa.gov/mission_pages/kepler/main/index.html)
- [CRISP-ML(Q) Framework](https://ml-ops.org/content/crisp-ml)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

⭐ Si encuentras útil este proyecto, considera darle una estrella!
