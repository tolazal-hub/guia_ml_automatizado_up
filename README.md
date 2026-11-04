# Guía PC2 de regresión y series en Google Colab

Abre [guia_pc2.ipynb](guia_pc2.ipynb) en Colab y sube tu CSV/Excel al panel Archivos.
El recorrido está pensado para organizar un examen de **1 h 50 min**; los tiempos de entrenamiento dependen de tu base.

1. Ejecuta la instalación y completa archivo, tipo de problema, target y exclusiones.
2. Revisa calidad, escribe solo tus correcciones y realiza el EDA.
3. Sigue **regresión (sección 4)** o **series (sección 5)**. El otro recorrido se omite.
4. Compara, elige con validación y evalúa prueba (sección 6).
5. Predice y exporta si lo pide el enunciado (sección 7). Descarga los CSV antes de cerrar Colab.

No necesitas instalar Python ni ejecutar verificadores locales para usar el notebook en tu PC2.

En **2.2a**, después de convertir tipos y antes de nulos/EDA, la ruta de series agrupa la base por `FRECUENCIA`: `D` por día, `W-MON` de lunes a domingo o `MS` por mes. `FRECUENCIA_ORIGEN` describe el archivo recibido y `AGREGACION_TEMPORAL` permite sumar cantidades (`sum`), promediar observaciones (`mean`) o conservar una serie ya preparada (`None`). Se conserva `df_original` y el resultado queda en `df_temporal`.

Con origen diario se comprueba cobertura: por defecto se excluyen semanas/meses incompletos en los extremos; huecos interiores y grupos con valores nulos quedan pendientes de resolver. No se crean ceros ni se distribuyen totales mensuales en días. Para operaciones individuales, origen `None` indica que la cobertura no puede deducirse solo de los registros. Al cambiar la frecuencia adapta también horizonte, rezagos y ventana: por ejemplo, mensual con horizonte 3, rezagos `[1, 2, 3]` y ventana 3.

## Modelos y preparación

- **Regresión:** promedio, lineal, árbol sin límite, árbol podado, Random Forest, XGBoost y LightGBM; una sección visible por modelo. División 60/20/20. El Pipeline aprende mediana, moda, escalado y categorías con entrenamiento.
- **Series:** persistencia, Random Forest y XGBoost; calendario y rezagos anteriores, media móvil desplazada y pronóstico recursivo. Validación y prueba tienen HORIZONTE periodos cada una. Se rechazan huecos y fechas duplicadas.
- **Comparación de series (5.6–5.9):** barras de RMSE/MAE/MAPE, ranking de los tres mejores por CRITERIO_SERIES, dispersión y curvas por fecha para los candidatos. El uso del ganador continúa en la evaluación de prueba (6) y el pronóstico operativo (7.3); las observaciones posteriores permiten revisar métricas, residuos y fechas con mayor error.
- **Selección:** MODELO_ELEGIDO="auto" usa RMSE en regresión y CRITERIO_SERIES en series (MAPE_pct por defecto). Puedes escribir otro nombre de la tabla antes de evaluar prueba.
- RF/boosting usan 100 árboles inicialmente. Los hiperparámetros están escritos junto al modelo; no necesitas configurar una lista extensa.

La predicción usa el modelo en memoria. Solo el modelo seleccionado se reentrena: con train+validación para prueba y, después, con toda la historia para el uso operativo. En series, cambiar fechas de pronóstico no exige reentrenar el modelo operativo si ya ejecutaste su celda.

## Salidas y anexos

En /content/salidas_guia_pc2 se generan, según el caso:

- predicciones_base_nueva.csv para una segunda base tabular sin necesidad de target.
- pronostico_completo.csv, pronostico_intervalo.csv y pronostico_mensual.csv para series.

El resumen mensual indica cobertura y meses parciales; usa sum para cantidades acumulables y mean para niveles. La comparación interanual requiere ambos meses completos y denominador distinto de cero.

Al final se conservan clasificación, GridSearch/RandomizedSearch, backtesting, LazyPredict, FLAML, red neuronal, SHAP y guardado joblib. Están apagados y no instalan paquetes ni entrenan por defecto. Clasificación se activa seleccionando clasificacion en la configuración; los otros anexos tienen su propia bandera.

Las referencias principales son RapiMenu, agentes/recaudación y el caso de series de [referencias/](referencias/). Se conserva su estilo de preparación, entrenamiento, comparación y predicción, corrigiendo selección sobre prueba y medias móviles que incluían el target actual. Las referencias no se modifican.

## Verificación local (fuera del examen)

- Ligera: NumPy, pandas y openpyxl; ejecuta python tests/verificar_series_pc2.py.
- Completa: instala las librerías de la celda de instalación y nbformat; ejecuta python tests/verificar_guia_pc2.py.
- Ruta temporal ampliada: python tests/verificar_ruta_series_pc2.py. Comprueba la ruta sin pasar por regresión, ranking, gráficos, exportación y observaciones nuevas con estimadores simulados; requiere NumPy, pandas y Matplotlib.
- Agrupación temprana: python tests/verificar_agregacion_temporal_pc2.py. Comprueba sumas/promedios, cobertura, semanas lunes-domingo, meses bisiestos, nulos/huecos, protección contra desagregación y el flujo completo D/W/MS con estimadores simulados.
- PC2_VALIDATION_PACKAGES permite indicar otra carpeta de dependencias. Los verificadores no agregan paquetes automáticamente.

Las pruebas usan archivos sintéticos temporales y no modifican el notebook. La prueba ligera cubre estructura, anexos apagados, lectura, fechas, recursión, cobertura mensual y selección. La completa ejecuta todos los algoritmos principales, gráficos, categorías nuevas, clasificación, búsquedas, backtesting y guardado/recarga. AutoML, red y SHAP permanecen apagados en esa verificación.

El notebook se entrega sin datos propios ni resultados precalculados. La última celda registra qué verificaciones se ejecutaron realmente.

**Estado de esta entrega:** pasó la verificación ligera. La completa no pudo iniciarse por falta de dependencias; su descarga fue rechazada. No se ejecutó Colab ni se midieron tiempos de entrenamiento.
