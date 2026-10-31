# Guía PC2 de machine learning

Abre [`guia_pc2.ipynb`](guia_pc2.ipynb) en Google Colab con **Archivo → Subir notebook**.

1. Ejecuta la instalación y carga tu CSV o Excel.
2. Configura `TIPO_PROBLEMA`, `TARGET`, columnas y preparación.
3. Revisa el EDA de entrenamiento y activa solamente las limpiezas que decidas aplicar.
4. Compara modelos en validación, escribe tu elección y evalúa prueba.
5. Carga una segunda base al final para predecir y exportar el resultado.

La guía cubre clasificación binaria y multiclase, regresión y una serie temporal regular.
Cada transformación crea otra variable. Las búsquedas de hiperparámetros, importancia y SHAP
están desactivadas inicialmente. El notebook se entrega sin datos propios ni resultados precalculados.

Las explicaciones se basan en los notebooks de [`referencias/`](referencias/).

Verificación local: instala las dependencias de la primera celda y `nbformat`, y ejecuta
`python tests/verificar_guia_pc2.py`. Las pruebas usan archivos sintéticos temporales y
no modifican el notebook. Comprueban los tres recorridos, clasificación multiclase,
persistencia, opciones de búsqueda/SHAP, exportación y errores de datos y esquema.
