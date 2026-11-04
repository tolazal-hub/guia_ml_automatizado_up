"""Verifica agregación temporal temprana y su integración con series.

Requiere NumPy, pandas y Matplotlib. El entrenamiento se simula; la agrupación,
fechas, limpieza, gráficos, exportación y comparación usan el código del notebook.
"""
import contextlib
import io
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from verificar_ruta_series_pc2 import NOTEBOOK, namespace, run


def rejects(function, message, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError('Se aceptó una configuración inválida')


def check_aggregation():
    ns = {'pd': pd, 'np': np}
    run('funciones-agregacion-temporal', ns)
    aggregate = ns['agrupar_serie_temporal']
    # Meses completos, bisiesto y conservación de la base.
    original = pd.DataFrame({'fecha': pd.date_range('2024-01-01', '2024-03-31'), 'valor': 1.})
    frozen = original.copy(deep=True)
    monthly, coverage = aggregate(original, 'fecha', 'valor', 'MS')
    np.testing.assert_allclose(monthly['valor'], [31, 29, 31])
    assert coverage['completo'].all() and not coverage['excluido'].any()
    assert_frame_equal(original, frozen)
    # Extremos parciales no se interpretan como totales de meses completos.
    partial = pd.DataFrame({'fecha': pd.date_range('2024-01-15', '2024-03-20'), 'valor': 1.})
    result, coverage = aggregate(partial, 'fecha', 'valor', 'MS')
    assert result['fecha'].tolist() == [pd.Timestamp('2024-02-01')]
    assert result['valor'].tolist() == [29.]
    assert coverage['excluido'].tolist() == [True, False, True]
    retained, _ = aggregate(partial, 'fecha', 'valor', 'MS', excluir_extremos=False)
    assert retained['valor'].isna().tolist() == [True, False, True]
    # Un día ausente en el interior no se suma parcialmente.
    missing = original.drop(index=original.index[original['fecha'].eq('2024-02-10')])
    result, coverage = aggregate(missing, 'fecha', 'valor', 'MS')
    assert pd.isna(result.loc[1, 'valor'])
    assert coverage.loc['2024-02-01', 'periodos_observados'] == 28
    # Un TARGET nulo tampoco queda oculto por sum/mean que omiten NaN.
    nulls = original.copy()
    nulls.loc[40, 'valor'] = np.nan
    for operation in ['sum', 'mean']:
        result, coverage = aggregate(nulls, 'fecha', 'valor', 'MS', operation)
        assert pd.isna(result.loc[1, 'valor'])
        assert coverage.loc['2024-02-01', 'nulos_origen'] == 1
    # Semana de lunes a domingo: el lunes siguiente pertenece a otro grupo.
    days = pd.DataFrame({'fecha': pd.date_range('2026-11-02', '2026-11-15'), 'valor': np.arange(1., 15.)})
    weekly, coverage = aggregate(days, 'fecha', 'valor', 'W-MON')
    assert weekly['fecha'].tolist() == [pd.Timestamp('2026-11-02'), pd.Timestamp('2026-11-09')]
    np.testing.assert_allclose(weekly['valor'], [28, 77])
    weekly_mean, _ = aggregate(days, 'fecha', 'valor', 'W-MON', 'mean')
    np.testing.assert_allclose(weekly_mean['valor'], [4, 11])
    assert coverage['periodos_esperados'].tolist() == [7, 7]
    # Operaciones distintas del mismo día se suman; el orden de entrada no afecta.
    events = pd.DataFrame({'fecha': pd.to_datetime(['2026-11-04 12:00', '2026-11-02 10:00', '2026-11-02 11:00']), 'valor': [30., 10., 20.]})
    daily, coverage = aggregate(events, 'fecha', 'valor', 'D', frecuencia_origen=None)
    assert daily['valor'].iloc[0] == 30 and pd.isna(daily['valor'].iloc[1]) and daily['valor'].iloc[2] == 30
    assert coverage['completo'].isna().all()
    # No se distribuyen totales mensuales en días ni semanas entre meses.
    rejects(aggregate, 'desagregar', monthly, 'fecha', 'valor', 'D', frecuencia_origen='MS')
    rejects(aggregate, 'desagregar', weekly, 'fecha', 'valor', 'MS', frecuencia_origen='W-MON')
    preserved, _ = aggregate(monthly, 'fecha', 'valor', 'MS', None, 'MS')
    assert_frame_equal(preserved, monthly)
    rejects(aggregate, 'sin fechas duplicadas', pd.concat([monthly, monthly.iloc[[0]]]), 'fecha', 'valor', 'MS', None, 'MS')
    rejects(aggregate, 'No quedan periodos completos', partial.head(5), 'fecha', 'valor', 'MS')
    rejects(aggregate, 'Usa D', original, 'fecha', 'valor', 'INVALIDA')


def check_pipeline(folder, frequency, horizon, lags, window):
    ns = namespace(folder, 50. + np.arange(900), horizon=horizon)
    ns.update(FRECUENCIA=frequency, REZAGOS=lags, VENTANA=window,
              CATEGORICAS=[], COLUMNAS_EXCLUIR=[], df_original=ns['df_limpio'].copy())
    frozen = ns['df_original'].copy(deep=True)
    run('herramientas', ns)
    run('limpieza', ns)
    run('funciones-agregacion-temporal', ns)
    run('agregacion-temporal', ns)
    run('nulos', ns)
    run('duplicados', ns)
    run('resultado-limpieza', ns)
    assert_frame_equal(frozen, ns['df_original'])
    assert ns['df_limpio']['fecha'].is_unique
    expected, _ = ns['agrupar_serie_temporal'](frozen, 'fecha', 'valor', frequency)
    assert_frame_equal(ns['df_limpio'], expected)
    assert_frame_equal(ns['preparar_datos'](frozen), ns['df_limpio'])
    run('funcion-mensual', ns)
    run('eda-serie', ns)
    run('funciones-series', ns)
    run('preparacion-series', ns)
    run('series-persistencia', ns)
    run('series-rf', ns)
    run('series-xgb', ns)
    run('comparacion-series', ns)
    run('ranking-series', ns)
    run('candidatos-series', ns)
    run('eleccion', ns)
    run('prueba-series', ns)
    run('modelo-operativo-series', ns)
    run('prediccion-series', ns)
    assert len(ns['pronostico']) == horizon
    assert ns['historia_train'].index[-1] < ns['y_validacion_temporal'].index[0] < ns['y_prueba_temporal'].index[0]
    assert (folder / 'pronostico_completo.csv').is_file()
    if frequency == 'MS':
        # Observaciones posteriores diarias se convierten a meses ANTES del cruce.
        dates = ns['fechas_futuras']
        daily = pd.DataFrame({'fecha': pd.date_range(dates[0], dates[-1] + pd.offsets.MonthEnd(0)), 'valor': 1.})
        daily.to_csv(folder / 'reales_diarios.csv', index=False)
        ns.update(RUTA_REALES_NUEVOS=str(folder / 'reales_diarios.csv'), leer_archivo=lambda ruta, *args: pd.read_csv(ruta))
        run('reales-nuevos', ns)
        np.testing.assert_allclose(ns['comparacion_nueva']['valor'], dates.days_in_month)
        assert len(ns['comparacion_nueva']) == horizon
        # También acepta otro archivo que YA trae totales mensuales.
        monthly = ns['comparacion_nueva'][['fecha', 'valor']].copy()
        monthly.to_csv(folder / 'reales_mensuales.csv', index=False)
        ns.update(RUTA_REALES_NUEVOS=str(folder / 'reales_mensuales.csv'), FRECUENCIA_ORIGEN_REALES='MS')
        run('reales-nuevos', ns)
        np.testing.assert_allclose(ns['comparacion_nueva']['valor'], monthly['valor'])
    assert_frame_equal(frozen, ns['df_original'])


def main():
    ids = [c['id'] for c in NOTEBOOK['cells']]
    assert ids.index('limpieza') < ids.index('agregacion-temporal') < ids.index('nulos') < ids.index('eda-serie') < ids.index('preparacion-series')
    plt.show = lambda: plt.close('all')
    with tempfile.TemporaryDirectory(prefix='pc2_agregacion_') as temporary, contextlib.redirect_stdout(io.StringIO()):
        check_aggregation()
        folder = Path(temporary)
        check_pipeline(folder, 'D', 5, [1, 7, 14], 7)
        check_pipeline(folder, 'W-MON', 4, [1, 2, 4], 4)
        check_pipeline(folder, 'MS', 3, [1, 2, 3], 3)
        # En tabular se preservan todas las columnas y filas de la etapa de tipos.
        for kind in ['regresion', 'clasificacion']:
            source = pd.DataFrame({'a': [1, 2], 'b': ['A', 'B']})
            ns = {'TIPO_PROBLEMA': kind, 'df_tipos': source}
            run('agregacion-temporal', ns)
            assert_frame_equal(source, ns['df_temporal'])
            assert ns['df_temporal'] is not source
    print('OK: sum/mean, semanas lunes-domingo, bisiesto, extremos parciales, nulos/huecos, eventos y bloqueo de desagregación.')
    print('OK: limpieza → EDA → train/validación/prueba → pronóstico D/W/MS y reales nuevos diarios/mensuales; tabular conservado.')
    print('Entrenamiento simulado; no se ejecutó Google Colab.')


if __name__ == '__main__':
    main()
