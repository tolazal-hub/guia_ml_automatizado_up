"""Verifica cálculos y estructura sin entrenar ML: requiere NumPy, pandas y openpyxl.
Uso: python tests/verificar_series_pc2.py. No cambia el notebook ni las referencias.
"""
from pathlib import Path
import ast
import contextlib
import io
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


class OverrideAssignments(ast.NodeTransformer):
    """Configura casos de prueba en memoria, sin editar el archivo."""
    def __init__(self, values):
        self.values = values

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.values:
            value = self.values[node.targets[0].id]
            expression = f'Path({str(value)!r})' if isinstance(value, Path) else repr(value)
            node.value = ast.parse(expression, mode='eval').body
        return self.generic_visit(node)


def execute_cell(identifier, namespace, overrides=None):
    tree = ast.parse(CELLS[identifier])
    if overrides:
        tree = OverrideAssignments(overrides).visit(tree)
        ast.fix_missing_locations(tree)
    exec(compile(tree, 'cell:' + identifier, 'exec'), namespace)


def reject(identifier, namespace, message, overrides=None):
    try:
        execute_cell(identifier, namespace, overrides)
    except (ValueError, TypeError) as error:
        assert message.lower() in str(error).lower(), str(error)
    else:
        raise AssertionError('Se aceptó un caso inválido en ' + identifier)


def check_structure():
    assert NOTEBOOK['nbformat'] == 4
    assert len({c['id'] for c in NOTEBOOK['cells']}) == len(NOTEBOOK['cells'])
    core = NOTEBOOK['cells'][:next(i for i, c in enumerate(NOTEBOOK['cells']) if c['id'] == 'anexos')]
    core_lines = sum(len(''.join(c['source']).splitlines()) for c in core if c['cell_type'] == 'code')
    assert core_lines <= 650, core_lines
    assert len([n for n in ast.parse(CELLS['configuracion']).body if isinstance(n, ast.Assign)]) == 8
    for cell in NOTEBOOK['cells']:
        if cell['cell_type'] != 'code':
            continue
        assert cell['execution_count'] is None and not cell['outputs']
        if cell['id'] == 'instalacion':
            continue
        tree = ast.parse(''.join(cell['source']))
        assert not any(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                assert ast.get_docstring(node), node.name
    for kind in ['regresion', 'serie_tiempo']:
        for identifier in CELLS:
            if identifier.startswith('anexo-'):
                with patch('builtins.__import__', side_effect=AssertionError('Un anexo apagado intentó importar/instalar')):
                    execute_cell(identifier, {'TIPO_PROBLEMA': kind})
    print(f'OK estructura: {len(NOTEBOOK["cells"])} celdas, {core_lines} líneas principales y anexos apagados')


def check_reading_and_preparation():
    with tempfile.TemporaryDirectory(prefix='pc2_lectura_') as tmp:
        data = pd.DataFrame({'id': np.arange(80), 'valor': np.arange(80, dtype=float), 'zona': ['A', 'B'] * 40})
        for suffix, separator in [('.csv', ','), ('.csv', ';'), ('.xlsx', None)]:
            path = Path(tmp) / ('datos' + suffix)
            if suffix == '.xlsx':
                data.to_excel(path, index=False)
            else:
                data.to_csv(path, sep=separator, index=False)
            ns = {'Path': Path, 'pd': pd, 'np': np, 'display': lambda *args: None}
            execute_cell('configuracion', ns, {'RUTA_ARCHIVO': str(path), 'TARGET': 'valor', 'COLUMNAS_EXCLUIR': ['id']})
            execute_cell('carga', ns)
            execute_cell('limpieza', ns)
            assert_frame_equal(ns['df_original'], data, check_dtype=False)
            assert_frame_equal(ns['df_limpio'], ns['df_original'])
            ns['df_original'] = data.assign(valor=np.nan)
            reject('limpieza', ns, 'target tiene nulos')
        ns = {'pd': pd, 'np': np, 'display': lambda *args: None, 'TIPO_PROBLEMA': 'serie_tiempo',
              'TARGET': 'valor', 'FECHA': 'fecha', 'FORMATO_FECHA': '%Y-%m-%d', 'FRECUENCIA': 'D', 'HORIZONTE': 5}
        source = pd.DataFrame({'fecha': pd.date_range('2025-01-01', periods=120, freq='D'), 'valor': np.arange(120, dtype=float)})
        frozen = source.copy(deep=True)
        ns['df_limpio'] = source
        execute_cell('funciones-series', ns)
        execute_cell('preparacion-series', ns)
        assert len(ns['historia_train']) == 110
        assert len(ns['y_validacion_temporal']) == len(ns['y_prueba_temporal']) == 5
        assert ns['historia_train'].index[-1] < ns['y_validacion_temporal'].index[0] < ns['y_prueba_temporal'].index[0]
        assert_frame_equal(source, frozen)
        for invalid in [0, -1, True, 2.5]:
            reject('preparacion-series', {**ns, 'HORIZONTE': invalid}, 'entero positivo')
        reject('preparacion-series', {**ns, 'df_limpio': source.iloc[:29]}, 'Historia insuficiente')
        reject('preparacion-series', {**ns, 'df_limpio': source.drop(index=40)}, 'huecos')
        reject('preparacion-series', {**ns, 'df_limpio': pd.concat([source, source.iloc[[0]]])}, 'duplicadas')
        ns['df_limpio'] = source.iloc[::-1]
        execute_cell('preparacion-series', ns)
        assert ns['serie'].index.is_monotonic_increasing
    print('OK CSV/Excel, conservación de datos, reservas cronológicas, huecos e historia insuficiente')


def check_temporal_functions():
    ns = {'pd': pd, 'np': np}
    execute_cell('funciones-series', ns)
    execute_cell('funcion-mensual', ns)
    features, forecast, monthly = ns['crear_rezagos'], ns['pronosticar'], ns['resumir_mensual']
    history = pd.Series(np.arange(1., 10.), index=pd.date_range('2025-12-27', periods=9, freq='D'))
    original = history.copy()
    holidays = ['2026-01-01', '2026-01-06']
    table = features(history, [1, 3], 2, holidays)
    assert table.loc['2026-01-01', 'feriado'] == 1
    assert ns['calendario'](history.index, holidays).loc['2025-12-29', 'semana_iso'] == 1
    assert table.iloc[-1]['media_movil'] == 7.5
    changed = pd.concat([history.iloc[:-1], pd.Series([9999.], index=history.index[-1:])])
    assert_frame_equal(table.drop(columns='objetivo'), features(changed, [1, 3], 2, holidays).drop(columns='objetivo'))

    class CaptureModel:
        def __init__(self):
            self.rows = []

        def predict(self, X):
            self.rows.append(X.copy())
            return np.array([X['lag_1'].iloc[0] + 10])

    model = CaptureModel()
    future = pd.date_range(history.index[-1] + pd.offsets.Day(), periods=4, freq='D')
    result = forecast(model, history, future, [1, 3], 2, holidays)
    np.testing.assert_allclose(result, [19, 29, 39, 49])
    assert model.rows[1]['lag_1'].iloc[0] == 19 and model.rows[1]['media_movil'].iloc[0] == 14
    assert model.rows[1]['feriado'].iloc[0] == 1
    extended = pd.concat([history, pd.Series([999.], index=future[:1])])
    assert_frame_equal(model.rows[0], features(extended, [1, 3], 2, holidays).drop(columns='objetivo').iloc[[-1]], check_freq=False)
    np.testing.assert_allclose(forecast(None, history, future, [1], 2), 9)
    assert_series_equal(history, original)
    assert holidays == ['2026-01-01', '2026-01-06']
    for bad_dates in [pd.date_range(future[1], periods=3, freq='D'), pd.DatetimeIndex([future[0], future[2]])]:
        try:
            forecast(None, history, bad_dates, [1], 2)
        except ValueError:
            pass
        else:
            raise AssertionError('Se aceptaron fechas futuras inválidas')
    for value in [np.nan, -5.]:
        fixed = SimpleNamespace(predict=lambda X, v=value: [v])
        if np.isnan(value):
            try:
                forecast(fixed, history, future, [1], 2)
            except ValueError:
                pass
            else:
                raise AssertionError('Se aceptó una predicción no finita')
        else:
            np.testing.assert_allclose(forecast(fixed, history, future, [1], 2), -5.)
    daily = pd.Series(1., index=pd.date_range('2024-01-01', '2024-03-15', freq='D'))
    summary = monthly(daily, 'D', 'sum')
    assert summary.loc['2024-02', 'valor'] == 29 and summary.loc['2024-02', 'mes_completo']
    assert summary.loc['2024-03', 'valor'] == 15 and not summary.loc['2024-03', 'mes_completo']
    assert np.isclose(summary.loc['2024-03', 'cobertura_pct'], 15 / 31 * 100)
    assert not monthly(daily.drop(pd.Timestamp('2024-02-10')), 'D', 'sum').loc['2024-02', 'mes_completo']
    assert monthly(daily, 'D', 'mean').loc['2024-02', 'valor'] == 1
    for freq, start in [('W-MON', '2024-01-01'), ('7D', '2024-01-03'), ('MS', '2024-01-01')]:
        values = pd.Series(2., index=pd.date_range(start, '2024-03-31', freq=freq))
        assert monthly(values, freq, 'sum')['mes_completo'].all()
    print('OK calendario, medias sin fuga, recursión, persistencia y cobertura mensual')


def check_reports_and_selection():
    with tempfile.TemporaryDirectory(prefix='pc2_reporte_') as tmp:
        ns = {'pd': pd, 'np': np, 'display': lambda *args: None, 'TIPO_PROBLEMA': 'serie_tiempo', 'FECHA': 'fecha',
              'FRECUENCIA': 'D', 'SALIDAS': Path(tmp),
              'serie': pd.Series(2., index=pd.date_range('2024-01-01', '2024-03-31', freq='D'))}
        execute_cell('funcion-mensual', ns)
        dates = pd.date_range('2025-01-01', '2025-03-15', freq='D')
        ns['reporte'] = pd.DataFrame({'fecha': dates, 'ml_prediccion': 3.})
        execute_cell('reporte-mensual', ns)
        assert np.isclose(ns['resumen_mensual'].loc['2025-01', 'variacion_interanual_pct'], 50)
        assert pd.isna(ns['resumen_mensual'].loc['2025-03', 'real_anio_anterior'])
        ns['serie'] = ns['serie'].drop(pd.Timestamp('2024-01-10'))
        execute_cell('reporte-mensual', ns)
        assert pd.isna(ns['resumen_mensual'].loc['2025-01', 'real_anio_anterior'])
        ns['serie'] = pd.Series(0., index=pd.date_range('2024-01-01', '2024-03-31', freq='D'))
        execute_cell('reporte-mensual', ns)
        assert pd.isna(ns['resumen_mensual'].loc['2025-01', 'variacion_interanual_pct'])
    for kind, metric in [('regresion', 'RMSE'), ('serie_tiempo', 'MAE')]:
        ns = {'TIPO_PROBLEMA': kind, 'tabla_validacion': pd.DataFrame({metric: [10., 2.]}, index=['A', 'B']), 'modelos': {'A': object(), 'B': object()}}
        execute_cell('eleccion', ns)
        assert ns['nombre_elegido'] == 'B'
        execute_cell('eleccion', ns, {'MODELO_ELEGIDO': 'A'})
        assert ns['nombre_elegido'] == 'A'
        reject('eleccion', ns, 'nombre de la tabla', {'MODELO_ELEGIDO': 'Inexistente'})
    print('OK selección automática/manual y comparación interanual con meses completos')


def check_forecast_route():
    """Ejecuta la ruta operativa real con persistencia; solo omite el gráfico."""
    with tempfile.TemporaryDirectory(prefix='pc2_pronostico_') as tmp:
        ns = {'pd': pd, 'np': np, 'Path': Path, 'display': lambda *a: None,
              'plt': SimpleNamespace(show=lambda: None)}
        execute_cell('configuracion', ns, {'TIPO_PROBLEMA': 'serie_tiempo', 'TARGET': 'valor', 'HORIZONTE': 5})
        ns['df_limpio'] = pd.DataFrame({'fecha': pd.date_range('2025-01-01', periods=300, freq='D'),
                                      'valor': 50 + np.sin(np.arange(300) * 2 * np.pi / 7)})
        original = ns['df_limpio'].copy(deep=True)
        execute_cell('funciones-series', ns)
        execute_cell('preparacion-series', ns)
        execute_cell('funcion-mensual', ns)
        ns['modelo_elegido'] = None
        execute_cell('modelo-operativo-series', ns)
        execute_cell('configuracion-prediccion', ns, {'SALIDAS': Path(tmp)})
        with patch.object(pd.DataFrame, 'plot'):
            execute_cell('prediccion-series', ns)
        execute_cell('reporte-mensual', ns)
        assert len(ns['pronostico']) == 5
        assert ns['fechas_futuras'][0] == ns['serie'].index[-1] + pd.offsets.Day()
        np.testing.assert_allclose(ns['pronostico']['ml_prediccion'], ns['serie'].iloc[-1])
        for name in ['pronostico_completo.csv', 'pronostico_intervalo.csv', 'pronostico_mensual.csv']:
            assert (Path(tmp) / name).is_file()
        assert np.isclose(ns['resumen_mensual']['valor'].sum(), ns['pronostico']['ml_prediccion'].sum())
        assert not ns['resumen_mensual']['mes_completo'].any()
        origin = ns['serie'].index[-1]
        ns.update(FECHA_FIN_PRONOSTICO=(origin + pd.offsets.Day(70)).strftime('%Y-%m-%d'),
                  INICIO_REPORTE=(origin + pd.offsets.Day(4)).strftime('%Y-%m-%d'),
                  FIN_REPORTE=(origin + pd.offsets.Day(8)).strftime('%Y-%m-%d'))
        with patch.object(pd.DataFrame, 'plot'):
            execute_cell('prediccion-series', ns)
            assert len(ns['pronostico']) == 70 and len(ns['reporte']) == 5
            np.testing.assert_allclose(ns['reporte']['ml_prediccion'], ns['pronostico']['ml_prediccion'].iloc[3:8])
            reject('prediccion-series', {**ns, 'FIN_REPORTE': (origin + pd.offsets.Day(71)).strftime('%Y-%m-%d')}, 'dentro del horizonte')
            reject('prediccion-series', {**ns, 'FECHA_FIN_PRONOSTICO': origin.strftime('%Y-%m-%d')}, 'posterior al origen')
        assert_frame_equal(ns['df_limpio'], original)
        for freq, start in [('W-MON', '2024-01-01'), ('MS', '2020-01-01')]:
            ns.update(FRECUENCIA=freq, INICIO_REPORTE=None, FIN_REPORTE=None, FECHA_FIN_PRONOSTICO=None)
            ns['df_limpio'] = pd.DataFrame({'fecha': pd.date_range(start, periods=60, freq=freq), 'valor': np.arange(60, dtype=float)})
            execute_cell('preparacion-series', ns)
            execute_cell('modelo-operativo-series', ns)
            with patch.object(pd.DataFrame, 'plot'):
                execute_cell('prediccion-series', ns)
                assert len(ns['pronostico']) == 5
                bad_end = ns['fechas_futuras'][-1] + pd.offsets.Day()
                reject('prediccion-series', {**ns, 'FECHA_FIN_PRONOSTICO': bad_end.strftime('%Y-%m-%d')}, 'periodo de la frecuencia')
        # En tabular sin archivo nuevo no debe intentar ajustar ni predecir.
        inactive = {'TIPO_PROBLEMA': 'regresion', 'RUTA_BASE_NUEVA': None}
        execute_cell('modelo-operativo-regresion', inactive)
        execute_cell('prediccion-regresion', inactive)
    print('OK ruta operativa, intervalos, exportaciones, frecuencias D/W/MS y tabular sin otra base')


def main():
    check_structure()
    with contextlib.redirect_stdout(io.StringIO()):
        check_reading_and_preparation()
        check_temporal_functions()
        check_reports_and_selection()
        check_forecast_route()
    print('OK cálculos, lectura/exportación, fechas, reservas y selección. No se entrenó ML ni se dibujaron gráficos.')


if __name__ == '__main__':
    main()
