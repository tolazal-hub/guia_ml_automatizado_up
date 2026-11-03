"""Verifica el notebook con archivos sintéticos temporales, sin modificarlo.

Uso: python tests/verificar_guia_pc2.py
Requiere las librerías de la celda de instalación y nbformat.
Si existe .validation_packages/, se usa esa instalación local.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = Path(os.environ.get('PC2_VALIDATION_PACKAGES', ROOT / '.validation_packages'))
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import ast
import contextlib
import copy
import io
import json
import tempfile
import warnings
from unittest.mock import patch

import nbformat
from verificar_series_pc2 import check_temporal_functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.testing import assert_frame_equal, assert_series_equal
from joblib import parallel_backend
from threadpoolctl import threadpool_limits
from sklearn.pipeline import Pipeline

NOTEBOOK = nbformat.read(ROOT / 'guia_pc2.ipynb', as_version=4)
CELLS = {c.id: c.source for c in NOTEBOOK.cells if c.cell_type == 'code'}


class OverrideAssignments(ast.NodeTransformer):
    def __init__(self, values):
        self.values = values

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.values:
            node.value = ast.parse(repr(self.values[node.targets[0].id]), mode='eval').body
        return self.generic_visit(node)


def execute_cell(identifier, namespace, overrides=None):
    tree = ast.parse(CELLS[identifier])
    if overrides:
        tree = OverrideAssignments(overrides).visit(tree)
        ast.fix_missing_locations(tree)
    exec(compile(tree, f'cell:{identifier}', 'exec'), namespace)


def check_structure():
    nbformat.validate(NOTEBOOK)
    definitions = []
    for cell in NOTEBOOK.cells:
        if cell.cell_type != 'code' or cell.id == 'instalacion':
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assigned = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
                referenced = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
                assert not assigned & referenced, (cell.id, 'Reasignación autorreferente', assigned & referenced)
                # Estado de adaptadores, registro de candidatos y opciones de TensorFlow;
                # ninguna excepción permite modificar columnas de un DataFrame.
                allowed_state = {
                    ('flaml-adaptador', 'self._model'),
                    ('flaml-busqueda', 'modelos_flaml_ajustados[familia_original]'),
                    ('red-tabular', 'opciones_dataset_red.threading.private_threadpool_size'),
                }
                assert all(isinstance(t, (ast.Name, ast.Tuple, ast.List)) or (
                    (cell.id, ast.unparse(t)) in allowed_state
                ) for t in node.targets), (cell.id, 'Mutación de columna/atributo')
            if isinstance(node, ast.AugAssign):
                raise AssertionError((cell.id, 'Asignación acumulativa'))
            if isinstance(node, ast.Call):
                assert not any(k.arg == 'inplace' and isinstance(k.value, ast.Constant) and k.value.value for k in node.keywords)
            if isinstance(node, ast.FunctionDef):
                definitions.append(node.name)
                assert ast.get_docstring(node), node.name
    required = {'aplicar_reglas', 'cambiar_tipos', 'limpiar_explicitamente',
                'metricas_regresion', 'metricas_clasificacion', 'predecir_tabular',
                'variables_calendario', 'construir_rezagos', 'pronostico_recursivo',
                'resumir_mensual', 'errores_temporales'}
    assert required.issubset(definitions), required - set(definitions)
    markdown = '\n'.join(c.source for c in NOTEBOOK.cells if c.cell_type == 'markdown')
    for name in required:
        assert f'{name}(' in markdown, name
    assert all(not c.get('outputs') and c.get('execution_count') is None
               for c in NOTEBOOK.cells if c.cell_type == 'code')
    print(f'OK estructura: {len(NOTEBOOK.cells)} celdas, funciones requeridas y docstrings', flush=True)


def expect_error(identifier, namespace, exception, message):
    try:
        execute_cell(identifier, namespace)
    except exception as error:
        assert message.lower() in str(error).lower(), str(error)
    else:
        raise AssertionError(f'{identifier} no rechazó {message}')


def run_case(folder, kind, classes=2, optional=False, positive_first=False, persistence=False,
             seasonal=False, temporal_options=False):
    rng = np.random.default_rng(420 + classes)
    n = 240
    if kind == 'serie_tiempo':
        source = pd.DataFrame({'fecha': pd.date_range('2025-01-01', periods=n, freq='D'),
            'valor': 40 + np.arange(n) * .08 + np.sin(np.arange(n) * 2 * np.pi / 7) * 8 + rng.normal(0, 1, n)})
        future_dates = pd.date_range('2025-01-01', periods=n + 5, freq='D')[-5:]
        new_data = pd.DataFrame({'fecha': future_dates, 'valor': [50., 51., 53., 55., 57.]})
        config = {'TIPO_PROBLEMA': kind, 'TARGET': 'valor', 'FECHA': 'fecha', 'HORIZONTE': 5,
                  'VENTANAS_BACKTEST': 2, 'PASO_BACKTEST': 5, 'REZAGOS': [1, 7], 'VENTANAS': [3]}
        if temporal_options or seasonal:
            config.update(CALENDARIO_AMPLIADO=True, FERIADOS=['2025-01-01', '2025-08-29'],
                          PERIODO_ESTACIONAL=7, PERIODOS_VALIDACION_SERIE=5,
                          PERIODOS_PRUEBA_SERIE=5, VENTANA_MEDIA_EDA=7,
                          AGREGACION_MENSUAL='sum')
    else:
        numbers = rng.normal(size=n)
        source = pd.DataFrame({'cliente_id': np.arange(n), 'cantidad': numbers,
            'otra': rng.normal(size=n), 'pais': np.where(numbers > 0, 'Perú', 'Chile')})
        if kind == 'clasificacion':
            labels = ['Abandona', 'Permanece'] if classes == 2 else ['Alta', 'Media', 'Baja']
            source['objetivo'] = np.array(labels)[np.arange(n) % classes]
        else:
            source['objetivo'] = 10 + numbers * 4 + rng.normal(size=n)
            source.loc[np.arange(0, n, 7), 'objetivo'] = 0
        source.loc[[2, 12, 27], 'cantidad'] = np.nan
        source.loc[[7, 18, 33], 'pais'] = np.nan
        new_data = source.head(12).copy().assign(pais=['Colombia'] + source.head(11)['pais'].tolist(), nota='columna extra')
        config = {'TIPO_PROBLEMA': kind, 'TARGET': 'objetivo', 'COLUMNAS_NUMERICAS': ['cantidad', 'otra'],
            'COLUMNAS_CATEGORICAS': ['pais'], 'COLUMNAS_ID': ['cliente_id'],
            'IMPUTAR_NUMERICAS': True, 'IMPUTAR_CATEGORICAS': True,
            'CLASE_POSITIVA': 'Abandona' if positive_first else 'Permanece',
            'REGLAS_TEXTOS': []}
    name = f'{kind}_{classes}_{"optional" if optional else "base"}_{"seasonal" if seasonal else "persist" if persistence else "model"}_{temporal_options}'
    source_path = folder / f'{name}.csv'
    new_path = folder / f'{name}_new.xlsx'
    source.to_csv(source_path, index=False)
    new_data.to_excel(new_path, index=False)
    output = folder / name
    overrides = {
        'configuracion-carga': {'RUTA_ARCHIVO': str(source_path)},
        'configuracion-problema': config,
        'guardar-modelo': {'DIRECTORIO_SALIDA': str(output)},
        'configuracion-nuevos': {'RUTA_BASE_NUEVA': str(new_path), 'FORMATO_BASE_NUEVA': 'excel',
            'EVALUAR_BASE_NUEVA_CON_TARGET': True, 'USAR_ARCHIVO_FUTURO': True},
        'busquedas-opcionales': {'EJECUTAR_RANDOM_SEARCH': optional, 'EJECUTAR_GRID_SEARCH': optional,
            'ITERACIONES_RANDOM': 2, 'GRILLA_RF': {'modelo__n_estimators': [10, 15], 'modelo__max_depth': [3]}},
        'importancia-opcional': {'EJECUTAR_IMPORTANCIA': optional},
        'shap-opcional': {'EJECUTAR_SHAP': optional, 'FILAS_FONDO_SHAP': 8, 'FILAS_EXPLICAR_SHAP': 2,
            'CLASE_SHAP': 'Abandona' if positive_first else 'Permanece'},
        'backtesting': {'MODELO_TEMPORAL_ELEGIDO': 'Estacional' if seasonal else 'Persistencia' if persistence else 'Random Forest'},
    }
    namespace = {'__name__': '__main__'}
    captured = io.StringIO()
    snapshots = {}
    frozen = None
    with warnings.catch_warnings(), contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured), \
            parallel_backend('threading', n_jobs=2), threadpool_limits(limits=1), \
            patch('IPython.display.display', lambda *a, **k: None), patch.object(plt, 'show', lambda: plt.close('all')):
        warnings.simplefilter('ignore', UserWarning)
        for identifier in CELLS:
            if identifier == 'instalacion':
                continue
            if identifier == 'predecir-nuevos':
                frozen = copy.deepcopy(namespace['paquete_modelo_cargado'])
                with patch.object(Pipeline, 'fit', side_effect=AssertionError('No debe reajustarse al predecir')), \
                        patch.object(Pipeline, 'fit_transform', side_effect=AssertionError('No debe reajustarse al predecir')):
                    execute_cell(identifier, namespace, overrides.get(identifier))
            else:
                try:
                    execute_cell(identifier, namespace, overrides.get(identifier))
                except Exception:
                    print(captured.getvalue()[-5000:], file=sys.__stdout__)
                    print(f'ERROR {name} en {identifier}', file=sys.__stdout__, flush=True)
                    raise
            if identifier in {'carga', 'funcion-reglas', 'reserva-temporal', 'funcion-limpieza'}:
                for key, value in namespace.items():
                    if isinstance(value, pd.DataFrame) and key not in snapshots:
                        snapshots[key] = value.copy(deep=True)
        for key, value in snapshots.items():
            assert_frame_equal(namespace[key], value, obj=f'{name}:{key}')
        assert_frame_equal(namespace['df_train_limpio'], namespace['df_reserva_train'])
        assert len(namespace['resultado_base_nueva']) == len(new_data)
        assert (output / 'predicciones_base_nueva.csv').is_file()
        assert (output / 'modelo_y_configuracion.joblib').is_file()
        assert (output / 'configuracion.json').is_file()
        assert np.isfinite(namespace['resultado_base_nueva']['ml_prediccion']).all() if kind != 'clasificacion' else set(namespace['resultado_base_nueva']['ml_prediccion']).issubset(set(source['objetivo']))
        np.testing.assert_array_equal(namespace['resultado_base_nueva']['ml_fila_origen'], np.arange(len(new_data)))

        # Segunda inferencia sin target: ningún ajuste ni alteración de predicciones.
        original_new_frame = namespace['df_nueva_original']
        predicted_before = namespace['resultado_base_nueva']['ml_prediccion'].copy()
        namespace['df_nueva_original'] = original_new_frame.drop(columns=namespace['target_cargado'])
        namespace['EVALUAR_BASE_NUEVA_CON_TARGET'] = False
        execute_cell('validar-nuevos', namespace)
        with patch.object(Pipeline, 'fit', side_effect=AssertionError('Inferencia sin entrenamiento')):
            execute_cell('predecir-nuevos', namespace)
        pd.testing.assert_series_equal(namespace['resultado_base_nueva']['ml_prediccion'], predicted_before)
        namespace['df_nueva_original'] = original_new_frame

        # La transformación final usa exactamente la mediana y categorías de desarrollo.
        if kind != 'serie_tiempo':
            final_model = namespace['pipeline_final_ajustado']
            imputer = final_model.named_steps['preparacion'].named_transformers_['numericas'].named_steps['imputacion']
            np.testing.assert_allclose(imputer.statistics_, namespace['X_desarrollo_final'][['cantidad', 'otra']].median().to_numpy())
            loaded_imputer = namespace['modelo_cargado'].named_steps['preparacion'].named_transformers_['numericas'].named_steps['imputacion']
            np.testing.assert_allclose(loaded_imputer.statistics_, imputer.statistics_)
            assert 'Colombia' in namespace['categorias_desconocidas_nuevas']['pais']
            saved_new = namespace['df_nueva_original']
            namespace['df_nueva_original'] = saved_new.drop(columns='cantidad')
            expect_error('validar-nuevos', namespace, ValueError, 'faltan columnas')
            namespace['df_nueva_original'] = saved_new.assign(cantidad='texto incompatible')
            expect_error('validar-nuevos', namespace, TypeError, 'debe ser numérica')
            namespace['df_nueva_original'] = saved_new.assign(pais=1)
            expect_error('validar-nuevos', namespace, TypeError, 'códigos numéricos y texto')
            namespace['df_nueva_original'] = saved_new
            namespace['configuracion_cargada'] = {**namespace['configuracion_cargada'], 'categoria_desconocida': 'error'}
            expect_error('validar-nuevos', namespace, ValueError, 'categorías desconocidas')
            if kind == 'clasificacion':
                assert len(namespace['tabla_metricas_clasificacion']) == 14
                np.testing.assert_allclose(namespace['probabilidades_nuevas'].sum(axis=1), 1)
                original_train = namespace['df_train_limpio']
                namespace['df_train_limpio'] = original_train.iloc[:0]
                # La partición inicial también rechaza un target nulo, antes de entrenar.
                namespace['df_limpio'] = namespace['df_limpio'].assign(objetivo=np.nan)
                expect_error('validacion-configuracion', namespace, ValueError, 'targets nulos')
        else:
            history = namespace['historia_train']
            history_copy = history.copy()
            dates = pd.date_range(history.index[-1] + pd.offsets.Day(), periods=5, freq='D')
            pred_one = namespace['pronostico_recursivo'](None, history, dates, [1], [2])
            np.testing.assert_allclose(pred_one, history.iloc[-1])
            assert_series_equal(history, history_copy)
            lag_example = namespace['construir_rezagos'](pd.Series([10., 20., 30.], index=pd.date_range('2025-01-01', periods=3)), [1], [2])
            assert lag_example.iloc[-1]['lag_1'] == 20 and lag_example.iloc[-1]['media_2'] == 15
            assert namespace['fechas_prueba_temporal'][0] > namespace['historia_desarrollo_temporal'].index[-1]
            assert namespace['tabla_backtest']['Inicio'].max() <= history.index[-5]
            for bad_source, allowed_order, expected in [
                (source.iloc[::-1], False, 'desordenadas'),
                (source.drop(index=40), False, 'huecos'),
                (pd.concat([source, source.iloc[[10]]]).sort_values('fecha'), True, 'varias filas'),
            ]:
                namespace['df_limpio'] = bad_source
                namespace['PERMITIR_ORDENAR_FECHAS'] = allowed_order
                expect_error('reserva-temporal', namespace, ValueError, expected)
            # La ruta temporal predeterminada también permite fechas futuras sin otro archivo.
            namespace['USAR_ARCHIVO_FUTURO'] = False
            execute_cell('recargar-y-leer-nuevos', namespace)
            execute_cell('validar-nuevos', namespace)
            execute_cell('predecir-nuevos', namespace)
            assert len(namespace['resultado_base_nueva']) == config['HORIZONTE']
            check_temporal_case(namespace, source, new_data, output, temporal_options or seasonal)

        # Funciones sin efectos laterales y métricas en presencia de ceros.
        input_clean = pd.DataFrame({'a': [1, 1, 2], 'b': [' x ', ' x ', ' y ']})
        input_clean_copy = input_clean.copy(deep=True)
        assert len(namespace['limpiar_explicitamente'](input_clean, quitar_duplicados=True)) == 2
        converted = namespace['aplicar_reglas'](input_clean, [], ['b'], {}, {})
        assert converted['b'].tolist() == ['x', 'x', 'y']
        assert_frame_equal(input_clean, input_clean_copy)
        assert namespace['metricas_regresion']([0, 2, 4], [0, 2, 4])['ceros_excluidos_MAPE'] == 1
        assert np.isnan(namespace['metricas_regresion']([0, 0], [0, 1])['MAPE_pct'])
    print(f'OK {name}: ejecución completa, Excel nuevo, exportación y casos adversos', flush=True)
    return namespace


def check_temporal_case(namespace, source, new_data, output, enhanced):
    """Verifica el recorrido temporal real, incluyendo recarga y reportes."""
    history = namespace['paquete_modelo_cargado']['historia']
    config = namespace['configuracion_cargada']
    before = namespace['resultado_base_nueva']['ml_prediccion'].copy()
    # Cambiar los targets del archivo futuro no puede cambiar las predicciones.
    namespace['df_nueva_original'] = new_data.assign(valor=-999999.)
    execute_cell('validar-nuevos', namespace)
    execute_cell('predecir-nuevos', namespace)
    np.testing.assert_allclose(namespace['resultado_base_nueva']['ml_prediccion'], before)
    unpersisted = namespace['pronostico_recursivo'](
        namespace['paquete_modelo']['modelo'], history, namespace['fechas_futuras_esperadas'],
        config['rezagos'], config['ventanas'], config['feriados'], config['calendario_ampliado'],
        config['periodo_estacional'] if namespace['paquete_modelo']['nombre_modelo'] == 'Estacional' else None)
    np.testing.assert_allclose(unpersisted, namespace['predicciones_futuras_nuevas'])
    assert namespace['error_por_paso_temporal'].index.get_level_values('paso').max() == config['horizonte']
    assert (output / 'pronostico_intervalo.csv').is_file()
    if enhanced:
        assert len(namespace['df_reserva_validacion']) == 5
        assert len(namespace['df_reserva_prueba']) == 5
        assert 'Estacional' in namespace['tabla_backtest']['Modelo'].unique()
        assert {'semana_iso', 'fin_de_semana', 'es_feriado'} <= set(namespace['X_train_temporal'].columns)
        assert (output / 'pronostico_mensual.csv').is_file()

    # Horizonte completo hasta fecha final; el reporte comienza después del origen.
    namespace['USAR_ARCHIVO_FUTURO'] = False
    end = history.index[-1] + pd.offsets.Day(70)
    namespace['FECHA_FIN_PRONOSTICO'] = end.strftime('%Y-%m-%d')
    namespace['FECHA_INICIO_REPORTE'] = (history.index[-1] + pd.offsets.Day(4)).strftime('%Y-%m-%d')
    namespace['FECHA_FIN_REPORTE'] = (history.index[-1] + pd.offsets.Day(8)).strftime('%Y-%m-%d')
    execute_cell('recargar-y-leer-nuevos', namespace)
    execute_cell('validar-nuevos', namespace)
    execute_cell('predecir-nuevos', namespace)
    execute_cell('reportes-temporales', namespace)
    assert len(namespace['resultado_base_nueva']) == 70
    assert len(namespace['resultado_intervalo']) == 5
    np.testing.assert_allclose(namespace['resultado_intervalo']['ml_prediccion'],
                               namespace['resultado_base_nueva']['ml_prediccion'].iloc[3:8])
    namespace['FECHA_FIN_REPORTE'] = (end + pd.offsets.Day()).strftime('%Y-%m-%d')
    expect_error('reportes-temporales', namespace, ValueError, 'dentro del horizonte')
    namespace['FECHA_FIN_PRONOSTICO'] = history.index[-1].strftime('%Y-%m-%d')
    expect_error('recargar-y-leer-nuevos', namespace, ValueError, 'posterior al origen')
    namespace['FECHA_FIN_PRONOSTICO'] = None
    namespace['FECHA_INICIO_REPORTE'] = None
    namespace['FECHA_FIN_REPORTE'] = None




if __name__ == '__main__':
    check_structure()
    check_temporal_functions()
    with tempfile.TemporaryDirectory(prefix='guia_pc2_') as temporary:
        folder = Path(temporary)
        run_case(folder, 'clasificacion', classes=2, optional=True, positive_first=True)
        run_case(folder, 'clasificacion', classes=3)
        run_case(folder, 'regresion', optional=True)
        run_case(folder, 'serie_tiempo', optional=True)
        run_case(folder, 'serie_tiempo', persistence=True)
        run_case(folder, 'serie_tiempo', temporal_options=True)
        run_case(folder, 'serie_tiempo', seasonal=True, optional=True)
    print('Todas las verificaciones finalizaron correctamente.', flush=True)
