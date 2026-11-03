"""Integra el notebook con datos sintéticos, sin modificarlo.
Uso: python tests/verificar_guia_pc2.py. Requiere librerías de instalación y nbformat.
PC2_VALIDATION_PACKAGES permite usar una carpeta de dependencias alternativa.
"""
from pathlib import Path
import os
import sys
if os.environ.get('PC2_VALIDATION_PACKAGES'):
    sys.path.insert(0, os.environ['PC2_VALIDATION_PACKAGES'])
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import contextlib
import copy
import io
import tempfile
import time
import warnings
from unittest.mock import patch
import nbformat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.testing import assert_frame_equal
from sklearn.pipeline import Pipeline
from joblib import parallel_backend
from threadpoolctl import threadpool_limits
from verificar_series_pc2 import ROOT, CELLS, execute_cell, reject, main as light_main


def run_case(folder, kind, choice='auto', classes=2, extras=False):
    rng = np.random.default_rng(42)
    n = 240
    if kind == 'serie_tiempo':
        source = pd.DataFrame({'fecha': pd.date_range('2025-01-01', periods=n, freq='D'),
            'valor': 50 + np.arange(n) * .05 + 8 * np.sin(np.arange(n) * 2 * np.pi / 7) + rng.normal(0, 1, n)})
        source.loc[10, 'valor'] = 0
        config = {'TIPO_PROBLEMA': kind, 'TARGET': 'valor', 'FECHA': 'fecha', 'HORIZONTE': 5}
        new_data = None
    else:
        signal = rng.normal(size=n)
        source = pd.DataFrame({'id': np.arange(n), 'cantidad': signal, 'otra': rng.normal(size=n),
                               'zona': np.where(signal > 0, 'A', 'B')})
        if kind == 'regresion':
            source['valor'] = 10 + 4 * signal + rng.normal(size=n)
            source.loc[np.arange(0, n, 17), 'valor'] = 0
        else:
            source['valor'] = np.array(['Abandona', 'Permanece', 'Otra'][:classes])[np.arange(n) % classes]
        source.loc[[3, 13, 25], 'cantidad'] = np.nan
        source.loc[[4, 14, 26], 'zona'] = np.nan
        new_data = source.head(8).drop(columns='valor').assign(zona=['Nueva'] + ['A'] * 7)
        config = {'TIPO_PROBLEMA': kind, 'TARGET': 'valor', 'COLUMNAS_EXCLUIR': ['id']}
    name = f'{kind}_{choice}_{classes}'.replace(' ', '_')
    path = folder / (name + '.csv')
    source.to_csv(path, index=False)
    new_path = folder / (name + '_nuevos.xlsx')
    if new_data is not None:
        new_data.to_excel(new_path, index=False)
    output = folder / name
    overrides = {'configuracion': {**config, 'RUTA_ARCHIVO': str(path)},
        'eleccion': {'MODELO_ELEGIDO': choice},
        'configuracion-prediccion': {'SALIDAS': output, 'RUTA_BASE_NUEVA': str(new_path) if new_data is not None else None},
        'anexo-persistencia': {'GUARDAR_MODELO': True}}
    ns = {'__name__': '__main__'}
    captured = io.StringIO()
    frozen = None
    started = time.perf_counter()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured), warnings.catch_warnings(), \
         parallel_backend('threading', n_jobs=2), threadpool_limits(limits=1), \
         patch('IPython.display.display', lambda *a, **kw: None), patch.object(plt, 'show', lambda: plt.close('all')):
        warnings.simplefilter('ignore', UserWarning)
        for identifier in CELLS:
            if identifier == 'instalacion':
                continue
            try:
                if identifier == 'prediccion-regresion':
                    with patch.object(Pipeline, 'fit', side_effect=AssertionError('Inferencia no debe ajustar')):
                        execute_cell(identifier, ns, overrides.get(identifier))
                else:
                    execute_cell(identifier, ns, overrides.get(identifier))
            except Exception:
                print(captured.getvalue()[-4000:], file=sys.__stdout__)
                print('ERROR en', identifier, name, file=sys.__stdout__)
                raise
            if identifier == 'carga':
                frozen = ns['df_original'].copy(deep=True)
        assert_frame_equal(ns['df_original'], frozen)
        assert_frame_equal(ns['df_limpio'], frozen)
        assert (output / 'modelo_pc2.joblib').is_file()
        assert np.isfinite(ns['metricas_prueba']['MAE']) if kind != 'clasificacion' else True
        if kind == 'regresion':
            assert len(ns['modelos']) == 7
            assert len(ns['tabla_comparacion']) == 14
            assert 'id' not in ns['X'].columns and 'valor' not in ns['X'].columns
            assert (output / 'predicciones_base_nueva.csv').is_file()
            result = pd.read_csv(output / 'predicciones_base_nueva.csv')
            assert len(result) == len(new_data) and np.isfinite(result['ml_prediccion']).all()
            np.testing.assert_array_equal(result['id'], new_data['id'])
            loaded = ns['paquete_recargado']['modelo']
            np.testing.assert_allclose(loaded.predict(ns['X_nueva']), ns['resultado_nuevo']['ml_prediccion'])
            imputer = ns['modelo_final'].named_steps['preparacion'].named_transformers_['num'].named_steps['imputar']
            np.testing.assert_allclose(imputer.statistics_, ns['X_desarrollo'].select_dtypes(include='number').median())
            assert len(ns['y_desarrollo']) == len(ns['y_train']) + len(ns['y_validacion'])
            missing_path = folder / (name + '_incompleta.xlsx')
            new_data.drop(columns='cantidad').to_excel(missing_path, index=False)
            with patch.object(Pipeline, 'fit', side_effect=AssertionError('Validar columnas antes de entrenar')):
                reject('modelo-operativo-regresion', {**ns, 'RUTA_BASE_NUEVA': str(missing_path)}, 'Faltan predictores')
                execute_cell('modelo-operativo-regresion', {**ns, 'RUTA_BASE_NUEVA': None})
                execute_cell('prediccion-regresion', {**ns, 'RUTA_BASE_NUEVA': None})
            original_new = ns['resultado_nuevo'].copy()
            ns['X_nueva'] = ns['X_nueva'].assign(zona='Otra categoría nueva')
            with patch.object(Pipeline, 'fit', side_effect=AssertionError('Inferencia sin entrenamiento')):
                execute_cell('prediccion-regresion', ns)
            assert len(ns['resultado_nuevo']) == len(original_new)
            selected = ns['nombre_elegido']
            saved_test = ns['y_prueba'].copy()
            ns['y_prueba'] = saved_test * 10000
            execute_cell('eleccion', ns, {'MODELO_ELEGIDO': choice})
            assert ns['nombre_elegido'] == selected
            ns['y_prueba'] = saved_test
            assert ns['metricas']([0, 2, 4], [0, 2, 4])['ceros_MAPE'] == 1
            assert np.isnan(ns['metricas']([0, 0], [0, 1])['MAPE_pct'])
            if extras:
                for search in ['grid', 'random']:
                    execute_cell('anexo-busqueda', ns, {'EJECUTAR_BUSQUEDA': True, 'TIPO_BUSQUEDA': search,
                        'grilla': {'modelo__max_depth': [3, 4], 'modelo__min_samples_leaf': [2]}})
                    assert len(ns['busqueda'].cv_results_['params']) == 2
        elif kind == 'serie_tiempo':
            assert len(ns['modelos']) == 3 and len(ns['tabla_comparacion']) == 6
            assert len(ns['y_validacion_temporal']) == len(ns['y_prueba_temporal']) == 5
            assert ns['historia_desarrollo'].index[-1] < ns['fechas_prueba'][0]
            assert ns['historia_operativa'].index[-1] == ns['serie'].index[-1]
            assert ns['fechas_futuras'][0] > ns['serie'].index[-1]
            assert len(ns['pronostico']) == 5
            assert (output / 'pronostico_completo.csv').is_file()
            assert (output / 'pronostico_intervalo.csv').is_file()
            assert (output / 'pronostico_mensual.csv').is_file()
            pkg = ns['paquete_recargado']
            reloaded = ns['pronosticar'](pkg['modelo'], pkg['historia'], ns['fechas_futuras'], pkg['rezagos'], pkg['ventana'], pkg['feriados'])
            np.testing.assert_allclose(reloaded, ns['pred_futuro'])
            saved_history = ns['historia_train'].copy()
            before = ns['pronosticar'](ns['modelos']['Random Forest'], saved_history, ns['fechas_validacion'], ns['REZAGOS'], ns['VENTANA'])
            ns['y_validacion_temporal'] = ns['y_validacion_temporal'] * 0 - 999999
            after = ns['pronosticar'](ns['modelos']['Random Forest'], saved_history, ns['fechas_validacion'], ns['REZAGOS'], ns['VENTANA'])
            np.testing.assert_allclose(before, after)
            origin = ns['serie'].index[-1]
            ns.update(FECHA_FIN_PRONOSTICO=(origin + pd.offsets.Day(60)).strftime('%Y-%m-%d'),
                      INICIO_REPORTE=(origin + pd.offsets.Day(4)).strftime('%Y-%m-%d'),
                      FIN_REPORTE=(origin + pd.offsets.Day(8)).strftime('%Y-%m-%d'))
            execute_cell('prediccion-series', ns)
            execute_cell('reporte-mensual', ns)
            assert len(ns['pronostico']) == 60 and len(ns['reporte']) == 5
            np.testing.assert_allclose(ns['reporte']['ml_prediccion'], ns['pronostico']['ml_prediccion'].iloc[3:8])
            reject('prediccion-series', {**ns, 'FIN_REPORTE': (origin + pd.offsets.Day(61)).strftime('%Y-%m-%d')}, 'dentro del horizonte')
            reject('prediccion-series', {**ns, 'FECHA_FIN_PRONOSTICO': origin.strftime('%Y-%m-%d')}, 'posterior al origen')
            if extras:
                execute_cell('anexo-backtesting', ns, {'EJECUTAR_BACKTEST': True})
                assert len(ns['tabla_backtest']) == 9
                assert ns['tabla_backtest']['Origen'].max() < ns['historia_train'].index[-1]
        else:
            assert len(ns['modelos_clf']) == 3
            result = pd.read_csv(output / 'predicciones_base_nueva.csv')
            assert set(result['ml_prediccion']).issubset(set(source['valor']))
            assert len(result) == len(new_data)
            np.testing.assert_array_equal(ns['paquete_recargado']['modelo'].predict(ns['X_prueba_clf']),
                                          ns['modelo_operativo'].predict(ns['X_prueba_clf']))
    print(f'OK {name}: ejecución, gráficos, exportación y guardado/recarga ({time.perf_counter() - started:.1f} s)', flush=True)


def main():
    notebook = nbformat.read(ROOT / 'guia_pc2.ipynb', as_version=4)
    nbformat.validate(notebook)
    light_main()
    with tempfile.TemporaryDirectory(prefix='pc2_integracion_') as tmp:
        folder = Path(tmp)
        run_case(folder, 'regresion', extras=True)
        run_case(folder, 'regresion', choice='Promedio')
        run_case(folder, 'serie_tiempo', choice='Random Forest', extras=True)
        run_case(folder, 'serie_tiempo', choice='XGBoost')
        run_case(folder, 'serie_tiempo', choice='Persistencia')
        run_case(folder, 'clasificacion', classes=2)
        run_case(folder, 'clasificacion', classes=3)
    print('Todas las verificaciones finalizaron correctamente. No se instalaron anexos de AutoML/red/SHAP.')


if __name__ == '__main__':
    main()
