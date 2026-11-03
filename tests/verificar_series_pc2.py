"""Pruebas de cálculos temporales: solo requieren NumPy y pandas; omiten gráficos.

Uso: python tests/verificar_series_pc2.py
No entrena modelos ni sustituye la verificación completa del notebook.
"""
from pathlib import Path
import ast
import copy
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'guia_pc2.ipynb').read_text(encoding='utf-8'))
CELLS = {c['id']: ''.join(c['source']) for c in NOTEBOOK['cells'] if c['cell_type'] == 'code'}
plt = SimpleNamespace(show=lambda: None, close=lambda *args: None)


def execute_cell(identifier, namespace):
    """Ejecuta una celda real sin modificar el notebook."""
    exec(compile(CELLS[identifier], f'cell:{identifier}', 'exec'), namespace)


def check_temporal_functions():
    """Casos calculables sin entrenar: fuga, feriados, estacionalidad y meses."""
    namespace = {'pd': pd, 'np': np}
    execute_cell('funciones-temporales', namespace)
    features = namespace['construir_rezagos']
    predict = namespace['pronostico_recursivo']
    monthly = namespace['resumir_mensual']
    dates = pd.date_range('2025-12-27', periods=9, freq='D')
    history = pd.Series(np.arange(9, dtype=float) + 1, index=dates)
    untouched = history.copy()
    holidays = ['2026-01-01', '2026-01-06']
    engineered = features(history, [1, 3], [2], holidays, True)
    assert engineered.loc['2026-01-01', 'es_feriado'] == 1
    assert engineered.loc['2025-12-29', 'semana_iso'] == 1
    assert engineered.loc['2025-12-27', 'fin_de_semana'] == 1
    assert engineered.loc['2025-12-29', 'fin_de_semana'] == 0
    assert engineered.iloc[-1]['media_2'] == 7.5
    changed = features(pd.concat([history.iloc[:-1], pd.Series([9999.], index=dates[-1:])]), [1, 3], [2], holidays, True)
    assert_frame_equal(engineered.drop(columns='objetivo'), changed.drop(columns='objetivo'))
    assert_series_equal(history, untouched)
    assert holidays == ['2026-01-01', '2026-01-06']

    class CaptureModel:
        def __init__(self):
            self.rows = []

        def predict(self, X):
            self.rows.append(X.copy())
            return np.array([float(X['lag_1'].iloc[0]) + 10])

    model = CaptureModel()
    future = pd.date_range(dates[-1] + pd.offsets.Day(), periods=4, freq='D')
    recursive = predict(model, history, future, [1, 3], [2], holidays, True)
    np.testing.assert_allclose(recursive, [19, 29, 39, 49])
    assert list(model.rows[0].columns) == list(engineered.drop(columns='objetivo').columns)
    assert model.rows[1]['lag_1'].iloc[0] == 19
    assert model.rows[1]['media_2'].iloc[0] == 14
    assert model.rows[1]['es_feriado'].iloc[0] == 1
    # El primer paso coincide con construir una fila aún sin target conocido.
    next_table = features(pd.concat([history, pd.Series([np.nan], index=future[:1])]), [1, 3], [2], holidays, True)
    assert_frame_equal(model.rows[0], next_table.drop(columns='objetivo').iloc[[-1]], check_freq=False)
    seasonal = predict(None, history, future, [1], [2], periodo_estacional=3)
    np.testing.assert_allclose(seasonal, [7, 8, 9, 7])
    assert_series_equal(history, untouched)
    for bad_dates in [pd.date_range(future[0] + pd.offsets.Day(), periods=4, freq='D'),
                      pd.DatetimeIndex([future[0], future[2]])]:
        try:
            predict(None, history, bad_dates, [1], [2])
        except ValueError:
            pass
        else:
            raise AssertionError('Se aceptaron fechas futuras no consecutivas')

    daily = pd.Series(1., index=pd.date_range('2024-01-01', '2024-03-15', freq='D'))
    daily_copy = daily.copy()
    summary = monthly(daily, 'D', 'sum')
    assert summary.loc['2024-02', 'valor'] == 29
    assert summary.loc['2024-02', 'mes_completo']
    assert summary.loc['2024-03', 'valor'] == 15
    assert not summary.loc['2024-03', 'mes_completo']
    assert np.isclose(summary.loc['2024-03', 'cobertura_pct'], 15 / 31 * 100)
    assert not monthly(daily.drop(pd.Timestamp('2024-02-10')), 'D', 'sum').loc['2024-02', 'mes_completo']
    assert monthly(daily, 'D', 'mean').loc['2024-02', 'valor'] == 1
    assert_series_equal(daily, daily_copy)
    weekly = pd.Series(2., index=pd.date_range('2024-01-01', '2024-03-31', freq='W-MON'))
    assert monthly(weekly, 'W-MON', 'sum')['mes_completo'].all()
    spaced = pd.Series(2., index=pd.date_range('2024-01-03', '2024-03-31', freq='7D'))
    assert monthly(spaced, '7D', 'sum').loc['2024-01', 'mes_completo']
    assert monthly(spaced, '7D', 'sum').loc['2024-01', 'periodos_esperados'] == 5
    assert not monthly(spaced.iloc[1:], '7D', 'sum').loc['2024-01', 'mes_completo']
    monthly_values = pd.Series([2., 3.], index=pd.date_range('2024-01-01', periods=2, freq='MS'))
    assert monthly(monthly_values, 'MS', 'mean')['mes_completo'].all()

    # Comparación interanual: solo pares completos, sin división entre cero.
    historical = pd.Series(2., index=pd.date_range('2024-01-01', '2024-03-31', freq='D'))
    predictions = pd.Series(3., index=pd.date_range('2025-01-01', '2025-03-15', freq='D'))
    report_namespace = {**namespace, 'tipo_cargado': 'serie_tiempo', 'FECHA_INICIO_REPORTE': None,
        'FECHA_FIN_REPORTE': None, 'fechas_futuras_esperadas': predictions.index,
        'configuracion_cargada': {'fecha': 'fecha', 'frecuencia': 'D', 'agregacion_mensual': 'sum'},
        'df_nueva_fechas': pd.DataFrame({'fecha': predictions.index}),
        'resultado_base_nueva': pd.DataFrame({'fecha': predictions.index, 'ml_prediccion': predictions.to_numpy()}),
        'paquete_modelo_cargado': {'historia': historical}, 'plt': plt, 'display': lambda *a: None}
    with tempfile.TemporaryDirectory() as tmp, patch.object(pd.Series, 'plot'):
        report_namespace['directorio_salida'] = Path(tmp)
        execute_cell('reportes-temporales', report_namespace)
        report = report_namespace['resumen_mensual_pronostico']
        assert np.isclose(report.loc['2025-01', 'variacion_interanual_pct'], 50)
        assert pd.isna(report.loc['2025-03', 'real_anio_anterior'])
        report_namespace['paquete_modelo_cargado'] = {'historia': historical.drop(pd.Timestamp('2024-01-10'))}
        execute_cell('reportes-temporales', report_namespace)
        assert pd.isna(report_namespace['resumen_mensual_pronostico'].loc['2025-01', 'real_anio_anterior'])
        report_namespace['paquete_modelo_cargado'] = {'historia': historical * 0}
        execute_cell('reportes-temporales', report_namespace)
        assert pd.isna(report_namespace['resumen_mensual_pronostico'].loc['2025-01', 'variacion_interanual_pct'])
    print('OK utilidades temporales: calendario, recursión sin fuga, estacionalidad y cobertura mensual', flush=True)


def check_partitions_and_future_dates():
    """Comprueba las celdas de partición y fechas con un cargador de paquete simulado."""
    namespace = {'pd': pd, 'np': np, 'display': lambda *args: None}
    execute_cell('configuracion-problema', namespace)
    execute_cell('funciones-temporales', namespace)
    source = pd.DataFrame({'fecha': pd.date_range('2025-10-01', periods=120, freq='D'),
                           'valor': np.arange(120, dtype=float)})
    frozen = source.copy(deep=True)
    namespace.update(TIPO_PROBLEMA='serie_tiempo', TARGET='valor', FECHA='fecha',
                     df_limpio=source, REZAGOS=[1, 7], VENTANAS=[3],
                     PERIODOS_VALIDACION_SERIE=7, PERIODOS_PRUEBA_SERIE=9,
                     HORIZONTE=5, CALENDARIO_AMPLIADO=True, FERIADOS=['2026-01-01'])
    execute_cell('reserva-temporal', namespace)
    assert len(namespace['df_reserva_train']) == 104
    assert len(namespace['df_reserva_validacion']) == 7
    assert len(namespace['df_reserva_prueba']) == 9
    assert namespace['df_reserva_validacion']['fecha'].min() > namespace['df_reserva_train']['fecha'].max()
    assert namespace['df_reserva_prueba']['fecha'].min() > namespace['df_reserva_validacion']['fecha'].max()
    assert_frame_equal(source, frozen)

    namespace.update(PERIODOS_VALIDACION_SERIE=None, PERIODOS_PRUEBA_SERIE=None)
    execute_cell('reserva-temporal', namespace)
    assert len(namespace['df_reserva_validacion']) == 24
    assert len(namespace['df_reserva_prueba']) == 24

    def rejects(cell_id, overrides, message):
        """Un caso inválido no debe aceptarse ni producir un error incidental."""
        candidate = {**namespace, **overrides}
        try:
            execute_cell(cell_id, candidate)
        except ValueError as error:
            assert message.lower() in str(error).lower(), str(error)
        else:
            raise AssertionError(f'{cell_id} aceptó {overrides}')

    for field in ['PERIODOS_VALIDACION_SERIE', 'PERIODOS_PRUEBA_SERIE', 'PERIODO_ESTACIONAL', 'HORIZONTE']:
        for invalid in [0, -1, 1.5, True]:
            rejects('reserva-temporal', {field: invalid}, 'entero positivo')
    rejects('reserva-temporal', {'PERIODOS_VALIDACION_SERIE': 110}, 'Historia insuficiente')
    rejects('reserva-temporal', {'df_limpio': source.drop(index=20)}, 'huecos')
    rejects('reserva-temporal', {'df_limpio': source.iloc[::-1]}, 'desordenadas')
    rejects('reserva-temporal', {'df_limpio': pd.concat([source, source.iloc[[-1]]])}, 'varias filas')
    rejects('reserva-temporal', {'CALENDARIO_AMPLIADO': False}, 'Activa CALENDARIO_AMPLIADO')
    rejects('reserva-temporal', {'FERIADOS': ['fecha inválida']}, 'match format')

    with tempfile.TemporaryDirectory() as tmp:
        # Solo el cargador se simula: ejecutamos las celdas reales de fechas,
        # validación y predicción, con la línea base que no necesita sklearn.
        config = {'tipo_problema': 'serie_tiempo', 'target': 'valor', 'fecha': 'fecha',
                  'frecuencia': 'D', 'formato_fecha': '%Y-%m-%d', 'horizonte': 5,
                  'rezagos': [1, 7], 'ventanas': [3], 'feriados': ['2026-02-01'],
                  'calendario_ampliado': True, 'periodo_estacional': 7,
                  'agregacion_mensual': 'sum', 'identificadores': [],
                  'reglas_numericas': [], 'reglas_textos': [], 'reglas_fechas': {},
                  'reglas_reemplazos': {}, 'tipos_columnas': {}, 'formatos_fecha': {}}
        history = source.set_index('fecha')['valor']
        package = {'configuracion': config, 'modelo': None, 'historia': history, 'nombre_modelo': 'Estacional'}
        namespace.update(joblib=SimpleNamespace(load=lambda path: copy.deepcopy(package)),
                         ruta_modelo_guardado=Path(tmp) / 'modelo_no_utilizado.joblib', plt=plt)
        # Extraer solo las definiciones: las celdas también contienen configuración/ejemplos.
        for cell_id in ['funcion-reglas', 'cambiar-tipos']:
            definitions = [n for n in ast.parse(CELLS[cell_id]).body if isinstance(n, ast.FunctionDef)]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), cell_id, 'exec'), namespace)
        execute_cell('configuracion-nuevos', namespace)
        end = history.index[-1] + pd.offsets.Day(40)
        namespace.update(FECHA_FIN_PRONOSTICO=end.strftime('%Y-%m-%d'),
                         FECHA_INICIO_REPORTE=(history.index[-1] + pd.offsets.Day(4)).strftime('%Y-%m-%d'),
                         FECHA_FIN_REPORTE=(history.index[-1] + pd.offsets.Day(8)).strftime('%Y-%m-%d'),
                         directorio_salida=Path(tmp))
        execute_cell('recargar-y-leer-nuevos', namespace)
        execute_cell('validar-nuevos', namespace)
        with patch.object(pd.DataFrame, 'plot'), patch.object(pd.Series, 'plot'):
            execute_cell('predecir-nuevos', namespace)
            execute_cell('reportes-temporales', namespace)
        assert len(namespace['resultado_base_nueva']) == 40
        assert len(namespace['resultado_intervalo']) == 5
        np.testing.assert_allclose(namespace['resultado_intervalo']['ml_prediccion'],
                                   namespace['resultado_base_nueva']['ml_prediccion'].iloc[3:8])
        np.testing.assert_allclose(namespace['predicciones_futuras_nuevas'].iloc[:7], history.iloc[-7:])
        assert (Path(tmp) / 'pronostico_mensual.csv').is_file()
        assert namespace['paquete_modelo_cargado']['configuracion']['feriados'] == config['feriados']
        assert_series_equal(package['historia'], source.set_index('fecha')['valor'])
        rejects('recargar-y-leer-nuevos', {'FECHA_FIN_PRONOSTICO': history.index[-1].strftime('%Y-%m-%d')}, 'posterior al origen')
        rejects('recargar-y-leer-nuevos', {'USAR_ARCHIVO_FUTURO': True}, 'el archivo define el horizonte')
        rejects('reportes-temporales', {'FECHA_FIN_REPORTE': (end + pd.offsets.Day()).strftime('%Y-%m-%d')}, 'dentro del horizonte')
        rejects('reportes-temporales', {'FECHA_INICIO_REPORTE': end.strftime('%Y-%m-%d')}, 'ordenadas')

        # Frecuencias semanal y mensual: fechas finales deben pertenecer a su cuadrícula.
        for freq, start in [('W-MON', '2024-01-01'), ('MS', '2024-01-01')]:
            other_history = pd.Series(np.arange(20, dtype=float), index=pd.date_range(start, periods=20, freq=freq))
            package.update(historia=other_history, configuracion={**config, 'frecuencia': freq})
            namespace.update(FECHA_FIN_PRONOSTICO=None, FECHA_INICIO_REPORTE=None, FECHA_FIN_REPORTE=None)
            execute_cell('recargar-y-leer-nuevos', namespace)
            execute_cell('validar-nuevos', namespace)
            with patch.object(pd.DataFrame, 'plot'):
                execute_cell('predecir-nuevos', namespace)
            assert len(namespace['resultado_base_nueva']) == 5
            bad_end = namespace['fechas_futuras_esperadas'][-1] + pd.offsets.Day()
            rejects('recargar-y-leer-nuevos', {'FECHA_FIN_PRONOSTICO': bad_end.strftime('%Y-%m-%d')}, 'coincidir con un periodo')
    print('OK particiones explícitas/fracciones, fechas futuras y reporte posterior a la recursión', flush=True)


if __name__ == '__main__':
    assert NOTEBOOK['nbformat'] == 4
    assert len({c['id'] for c in NOTEBOOK['cells']}) == len(NOTEBOOK['cells'])
    for identifier, code in CELLS.items():
        if identifier != 'instalacion':
            ast.parse(code)
    check_temporal_functions()
    check_partitions_and_future_dates()
    print('Verificación temporal ligera completa; gráficos y entrenamiento de ML no ejecutados.', flush=True)

