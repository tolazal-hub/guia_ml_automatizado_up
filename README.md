# Guía PC2 de machine learning

Abre [`guia_pc2.ipynb`](guia_pc2.ipynb) en Google Colab con **Archivo → Subir notebook**.
Empieza por **Uso rápido en tu PC2**, al inicio del notebook: indica qué cargar,
qué configuración editar, cómo seguir cada recorrido y qué archivos entregar.
Para usar la guía en Colab no necesitas instalar Python ni ejecutar pruebas en tu PC.

1. Ejecuta la instalación y carga tu CSV o Excel.
2. Configura `TIPO_PROBLEMA`, `TARGET`, columnas y preparación.
3. Revisa calidad, aplica tus limpiezas y realiza el EDA; después divide para modelar.
4. Compara modelos en validación, escribe tu elección y evalúa prueba.
5. Carga una segunda base al final para predecir y exportar el resultado.

La guía cubre clasificación binaria y multiclase, regresión y una serie temporal regular.
Cada transformación crea otra variable. Las búsquedas de hiperparámetros, importancia y SHAP
están desactivadas inicialmente. El notebook se entrega sin datos propios ni resultados precalculados.

Las explicaciones se basan en los notebooks de [`referencias/`](referencias/).

El recorrido temporal incorpora las mejoras de `PD_Solucion_Regresion_Series.ipynb`:
calendario ampliado y feriados declarados, reservas por cantidad de periodos, línea base
estacional y errores por día de semana y paso del horizonte. Las opciones nuevas empiezan
desactivadas; se explican junto a la configuración temporal.

Para cantidades diarias, puedes activar `VENTANA_MEDIA_EDA=7`,
`AGREGACION_MENSUAL="sum"` y `PERIODO_ESTACIONAL=7`. Usa `"mean"` para niveles.
`CALENDARIO_AMPLIADO=True` permite declarar `FERIADOS`; incluye las fechas futuras.
`PERIODOS_VALIDACION_SERIE` y `PERIODOS_PRUEBA_SERIE` sustituyen sus respectivas
fracciones solo en series. Compara esas duraciones con `HORIZONTE`.

Sin archivo futuro, `FECHA_FIN_PRONOSTICO` permite generar hasta una fecha de la frecuencia
guardada. `FECHA_INICIO_REPORTE` y `FECHA_FIN_REPORTE` filtran después de pronosticar
todos los pasos. Se exportan `predicciones_base_nueva.csv`, `pronostico_intervalo.csv`
y, al activar la agregación mensual, `pronostico_mensual.csv`. El resumen indica cobertura
y meses parciales; solo compara contra el año anterior cuando ambos meses están completos.

Verificación local: instala las dependencias de la primera celda y `nbformat`, y ejecuta
`python tests/verificar_guia_pc2.py`. Las pruebas usan archivos sintéticos temporales y
no modifican el notebook. Comprueban los tres recorridos, clasificación multiclase,
persistencia, base estacional, calendario, recursión sin targets futuros, cobertura mensual,
opciones de búsqueda/SHAP, exportación y errores de datos y esquema.

Para comprobar solo cálculos y fechas con NumPy y pandas, ejecuta
`python tests/verificar_series_pc2.py`. Esta prueba ligera omite gráficos y entrenamiento;
no sustituye la verificación completa. Puedes elegir otra carpeta de dependencias para
el verificador completo mediante la variable de entorno `PC2_VALIDATION_PACKAGES`.
