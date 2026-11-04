"""Prueba la ruta temporal sin ejecutar regresión ni instalar dependencias de ML.

Usa NumPy, pandas y Matplotlib reales y estimadores simulados para comprobar
el orden del flujo, las reservas, ranking, gráficos y comparación posterior.
No comprueba el entrenamiento real de Random Forest ni XGBoost.
"""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'guia_pc2.ipynb').read_text(encoding='utf-8'))
CELLS = {cell['id']: ''.join(cell['source']) for cell in NOTEBOOK['cells']}


class OffsetModel:
    """Estimador determinista para detectar cambios en la cadena recursiva."""

    def __init__(self, offset, **kwargs):
        self.offset = offset

    def fit(self, X, y):
        self.fit_dates = y.index.copy()
        return self

    def predict(self, X):
        return X['lag_1'].to_numpy() + self.offset


class NoPipeline:
    """Permite ejecutar la presentación sin instalar scikit-learn."""


def run(identifier, namespace):
    exec(compile(CELLS[identifier], 'cell:' + identifier, 'exec'), namespace)


def r2(reales, predicciones):
    denominator = np.sum((reales - np.mean(reales)) ** 2)
    if denominator == 0:
        return 1. if np.array_equal(reales, predicciones) else 0.
    return 1 - np.sum((reales - predicciones) ** 2) / denominator


def namespace(folder, values, horizon=5, criterion='MAPE_pct'):
    return {
        'pd': pd, 'np': np, 'plt': plt, 'Path': Path,
        'display': lambda *args: None, 'clone': copy.deepcopy,
        'mean_absolute_error': lambda real, pred: np.mean(np.abs(real - pred)),
        'mean_squared_error': lambda real, pred: np.mean((real - pred) ** 2),
        'r2_score': r2, 'Pipeline': NoPipeline,
        'RandomForestRegressor': lambda **kw: OffsetModel(1., **kw),
        'XGBRegressor': lambda **kw: OffsetModel(2., **kw),
        'TIPO_PROBLEMA': 'serie_tiempo', 'TARGET': 'valor', 'UNIDAD': 'transacciones',
        'FECHA': 'fecha', 'FORMATO_FECHA': '%Y-%m-%d', 'FRECUENCIA': 'D',
        'HORIZONTE': horizon, 'REZAGOS': [1, 7, 14], 'VENTANA': 7,
        'USAR_FERIADOS_PERU': False, 'FERIADOS': [], 'SEMILLA': 42,
        'TOLERANCIA': None, 'CRITERIO_SERIES': criterion,
        'FRECUENCIA_ORIGEN': 'D', 'AGREGACION_TEMPORAL': 'sum',
        'EXCLUIR_EXTREMOS_INCOMPLETOS': True, 'FRECUENCIA_ORIGEN_REALES': None,
        'SEPARADOR': None, 'CODIFICACION': 'utf-8-sig', 'DECIMAL': '.', 'HOJA_EXCEL': 0,
        'SALIDAS': folder, 'FECHA_FIN_PRONOSTICO': None,
        'INICIO_REPORTE': None, 'FIN_REPORTE': None, 'RUTA_REALES_NUEVOS': None,
        'df_limpio': pd.DataFrame({'fecha': pd.date_range('2025-01-01', periods=len(values)), 'valor': values}),
    }


def check_route(folder, values, criterion='MAPE_pct', only_persistence=False, horizon=5, preview=None):
    ns = namespace(folder, values, horizon, criterion)
    original = ns['df_limpio'].copy(deep=True)
    # Sigue solo configuración común y ruta temporal; nunca ejecuta sección 4.
    run('herramientas', ns)
    run('funciones-agregacion-temporal', ns)
    run('funciones-series', ns)
    ns['modelos']['Modelo tabular previo'] = object()
    run('preparacion-series', ns)
    assert ns['modelos'] == {}
    run('series-persistencia', ns)
    if not only_persistence:
        run('series-rf', ns)
        run('series-xgb', ns)
    run('comparacion-series', ns)
    run('ranking-series', ns)
    run('candidatos-series', ns)
    if preview:
        # Reejecuta sin cerrar la figura para inspección visual del layout.
        closing_show = plt.show
        try:
            plt.show = lambda: None
            run('candidatos-series', ns)
            plt.gcf().savefig(preview, dpi=110)
        finally:
            plt.close('all')
            plt.show = closing_show
    expected = 'Persistencia' if only_persistence else 'Random Forest'
    if np.all(np.asarray(values) == 0):
        expected = 'Persistencia'
    assert ns['mejor_modelo_series'] == expected
    assert ns['candidatos_series'][0] == expected
    assert len(ns['tabla_comparacion']) == 2 * len(ns['modelos'])
    assert len(ns['y_validacion_temporal']) == len(ns['y_prueba_temporal']) == horizon
    assert ns['historia_train'].index[-1] < ns['y_validacion_temporal'].index[0]
    assert ns['y_validacion_temporal'].index[-1] < ns['y_prueba_temporal'].index[0]
    if not only_persistence:
        assert ns['modelos']['Random Forest'].fit_dates.max() == ns['historia_train'].index[-1]
        # Cambiar targets de validación/prueba no debe cambiar un pronóstico.
        forecast_before = ns['predicciones_validacion']['Random Forest'].copy()
        ns['y_prueba_temporal'] = ns['y_prueba_temporal'] * 10000
        rerun = ns['pronosticar'](ns['modelos']['Random Forest'], ns['historia_train'], ns['fechas_validacion'], ns['REZAGOS'], ns['VENTANA'])
        assert_series_equal(forecast_before, rerun)
        ns['y_prueba_temporal'] = ns['serie'].iloc[-horizon:].copy()
    run('eleccion', ns)
    assert ns['nombre_elegido'] == expected
    run('prueba-series', ns)
    assert np.isfinite(ns['metricas_prueba']['MAE'])
    if ns['modelo_final'] is not None:
        assert ns['modelo_final'].fit_dates.max() < ns['fechas_prueba'][0]
    run('modelo-operativo-series', ns)
    run('prediccion-series', ns)
    assert len(ns['pronostico']) == horizon
    assert ns['fechas_futuras'][0] > ns['serie'].index[-1]
    assert (folder / 'pronostico_completo.csv').is_file()
    assert (folder / 'pronostico_intervalo.csv').is_file()
    assert_frame_equal(original, ns['df_limpio'])
    return ns


def main():
    for cell in NOTEBOOK['cells']:
        if cell['cell_type'] == 'code' and cell['id'] != 'instalacion':
            ast.parse(''.join(cell['source']))
    ids = [c['id'] for c in NOTEBOOK['cells']]
    assert ids.index('comparacion-series') < ids.index('ranking-series') < ids.index('candidatos-series') < ids.index('eleccion')
    # Los bloques de regresión deben poder omitirse/ejecutarse sin tabla_validacion en series.
    run('9c33b2c6', {'TIPO_PROBLEMA': 'serie_tiempo'})
    run('a5e75125', {'TIPO_PROBLEMA': 'serie_tiempo'})
    plt.show = lambda: plt.close('all')
    with tempfile.TemporaryDirectory(prefix='pc2_series_') as temporary, contextlib.redirect_stdout(io.StringIO()):
        folder = Path(temporary)
        preview_dir = ROOT / '.tmp_guia'
        preview_dir.mkdir(exist_ok=True)
        ns = check_route(folder, 50. + np.arange(150), preview=preview_dir / 'candidatos_series.png')
        run('funcion-mensual', ns)
        run('reporte-mensual', ns)
        assert (folder / 'pronostico_mensual.csv').is_file()

        # Reales nuevos desordenados: unión por fecha, métricas y residuos exactos.
        future_real = ns['pronostico'][['fecha']].copy()
        future_real['valor'] = ns['pronostico']['ml_prediccion'] + np.arange(ns['HORIZONTE'])
        future_real.iloc[::-1].to_csv(folder / 'reales.csv', index=False)
        ns.update(RUTA_REALES_NUEVOS=str(folder / 'reales.csv'),
                  leer_archivo=lambda ruta, *args: pd.read_csv(ruta),
                  preparar_datos=lambda datos, **kwargs: datos.copy())
        history = ns['historia_operativa'].copy()
        pred = ns['pronostico'].copy(deep=True)
        run('reales-nuevos', ns)
        np.testing.assert_allclose(ns['comparacion_nueva']['residuo'], np.arange(ns['HORIZONTE']))
        assert (folder / 'comparacion_reales_nuevos.csv').is_file()
        assert_series_equal(history, ns['historia_operativa'])
        assert_frame_equal(pred, ns['pronostico'])

        check_route(folder, 50. + np.arange(150), only_persistence=True)
        check_route(folder, np.zeros(150), criterion='MAE')
        check_route(folder, 50. + np.arange(150), criterion='RMSE', horizon=1)

        # MAPE indefinido detiene el ranking con un mensaje accionable.
        ns = namespace(folder, np.zeros(100))
        ns['tabla_validacion'] = pd.DataFrame({'MAPE_pct': [np.nan]}, index=['Persistencia'])
        try:
            run('ranking-series', ns)
        except ValueError as error:
            assert 'MAE o RMSE' in str(error)
        else:
            raise AssertionError('MAPE indefinido no debe recomendar un ganador')

        # Las nuevas celdas de series no se activan en regresión.
        for identifier in ['preparacion-series', 'comparacion-series', 'ranking-series', 'candidatos-series', 'prediccion-series', 'reales-nuevos']:
            run(identifier, {'TIPO_PROBLEMA': 'regresion'})
    print('OK: sintaxis, ruta sin regresión, reservas, ranking MAPE/MAE/RMSE, 1/3 candidatos, horizonte 1, gráficos, exportación y reales nuevos.')
    print('Modelos simulados; no se verificó entrenamiento real de RF/XGBoost ni Google Colab.')


if __name__ == '__main__':
    main()
