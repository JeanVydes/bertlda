# -*- coding: utf-8 -*-
"""
Script00_Auditoria_Corpus.py
============================
PRIMER script del pipeline. Todos los demás consumen su salida.

QUÉ HACE
  1. Valida que existan las 4 columnas obligatorias (TI, AB, PY, Country).
  2. Crea un doc_id ESTABLE que se propaga a todas las salidas posteriores.
     (En v2 el enlace documento -> año/país era posicional e implícito: si
     BD.xlsx cambiaba de orden, los checkpoints antiguos desalineaban todo
     el análisis geográfico y temporal sin producir ningún error visible.)
  3. Detecta corpus exportado en MAYÚSCULAS y marca si hay que normalizar.
  4. Verifica duplicados (DOI y título normalizado) — REPORTA, no elimina.
  5. Excluye documentos sin texto suficiente, dejando constancia de cuáles.
  6. Normaliza nombres de país conservando original y normalizado.
  7. Deriva candidatos empíricos a stopwords de dominio (revisión manual).
  8. Exporta corpus_quality.xlsx y el checkpoint ck_corpus_base.pkl.

SALIDAS
  04_Resultados_Finales/Corpus/corpus_quality.xlsx
  03_Checkpoints/ck_corpus_base.pkl
"""

import os
import re
import sys
import pickle
from time import perf_counter

try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
sys.path.insert(0, _DIR)

import numpy as np
import pandas as pd

from config import CFG, rutas, crear_rutas, montar_drive, registrar_entorno
import procesador_v3 as pr

# =============================================================================
# NORMALIZACIÓN DE PAÍSES
# =============================================================================
# Variantes que aparecen en exportaciones de WoS/Scopus. Se documenta cada
# transformación en la hoja Paises_Normalizacion del Excel de salida.
MAPA_PAISES = {
    # Reino Unido: WoS exporta las naciones constituyentes por separado.
    'ENGLAND': 'United Kingdom', 'SCOTLAND': 'United Kingdom',
    'WALES': 'United Kingdom', 'NORTH IRELAND': 'United Kingdom',
    'NORTHERN IRELAND': 'United Kingdom', 'UK': 'United Kingdom',
    'GREAT BRITAIN': 'United Kingdom',
    # Estados Unidos
    'USA': 'United States', 'U.S.A.': 'United States',
    'UNITED STATES OF AMERICA': 'United States', 'US': 'United States',
    # Otras variantes frecuentes
    'RUSSIAN FEDERATION': 'Russia',
    'VIET NAM': 'Vietnam',
    'PEOPLES R CHINA': 'China', "PEOPLE'S REPUBLIC OF CHINA": 'China',
    'REPUBLIC OF KOREA': 'South Korea', 'KOREA': 'South Korea',
    'SOUTH KOREA': 'South Korea',
    'U ARAB EMIRATES': 'United Arab Emirates',
    'CZECH REPUBLIC': 'Czechia',
    'MARSHALL ISLAND': 'Marshall Islands',
    'TRINIDAD & TOBAGO': 'Trinidad and Tobago',
    'BOSNIA & HERCEG': 'Bosnia and Herzegovina',
    'CAPE VERDE': 'Cabo Verde',
    'IVORY COAST': "Cote d'Ivoire", 'COTE IVOIRE': "Cote d'Ivoire",
}

# Territorios dependientes. Se mantienen separados por defecto; cambiar a
# True si se quiere agregarlos a su Estado soberano (decisión a documentar).
AGREGAR_TERRITORIOS = False
MAPA_TERRITORIOS = {
    'FRENCH GUIANA': 'France', 'JERSEY': 'United Kingdom',
    'FAROE ISLANDS': 'Denmark', 'PUERTO RICO': 'United States',
    'HONG KONG': 'China', 'MACAU': 'China',
}


def normalizar_pais(valor):
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip()
    if not v or v.lower() == 'nan':
        return np.nan
    key = v.upper()
    if key in MAPA_PAISES:
        return MAPA_PAISES[key]
    if AGREGAR_TERRITORIOS and key in MAPA_TERRITORIOS:
        return MAPA_TERRITORIOS[key]
    return v.title() if v.isupper() else v


def _norm_titulo(s):
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':

    t0 = perf_counter()
    crear_rutas()
    montar_drive()

    print("\n" + "=" * 66)
    print("SCRIPT 00 — AUDITORÍA Y VALIDACIÓN DEL CORPUS")
    print("=" * 66)

    COL_TI = CFG['col_titulo']
    COL_AB = CFG['col_abstract']
    COL_PY = CFG['col_anio']
    COL_CO = CFG['col_pais']

    # ── 1. CARGA ─────────────────────────────────────────────────────────────
    ruta_bd = os.path.join(rutas['base'], CFG['archivo_bd'])
    if not os.path.exists(ruta_bd):
        raise FileNotFoundError(
            f"No se encontró {CFG['archivo_bd']} en {rutas['base']}\n"
            f"Edita RUTA_COLAB / RUTA_LOCAL en config.py."
        )
    df = pd.read_excel(ruta_bd)
    n_original = len(df)
    print(f"\n── CARGA ──\n  {ruta_bd}\n  {n_original:,} filas × {df.shape[1]} columnas")

    # ── 2. VALIDACIÓN DE COLUMNAS OBLIGATORIAS ───────────────────────────────
    print("\n── VALIDACIÓN DE COLUMNAS ──")
    obligatorias = [COL_TI, COL_AB, COL_PY, COL_CO]
    faltan = [c for c in obligatorias if c not in df.columns]
    if faltan:
        raise KeyError(
            "FALTAN COLUMNAS OBLIGATORIAS: " + ", ".join(faltan) + "\n"
            f"El pipeline requiere como mínimo: {obligatorias}\n"
            f"Columnas presentes: {df.columns.tolist()}"
        )
    for c in obligatorias:
        print(f"  OK  {c:<10} nulos={df[c].isna().sum():>5}")

    opcionales = [c for c in CFG['cols_opcionales'] if c in df.columns]
    print(f"  Opcionales detectadas: {opcionales or 'ninguna'}")

    # ── 3. doc_id ESTABLE ────────────────────────────────────────────────────
    df = df.reset_index(drop=True)
    df.insert(0, 'doc_id', np.arange(len(df), dtype=int))

    # ── 4. CASING ────────────────────────────────────────────────────────────
    print("\n── CASING ──")
    frac_ti, mayus_ti = pr.detectar_mayusculas(df[COL_TI],
                                               CFG['umbral_deteccion_mayusculas'])
    frac_ab, mayus_ab = pr.detectar_mayusculas(df[COL_AB],
                                               CFG['umbral_deteccion_mayusculas'])
    print(f"  {COL_TI}: {frac_ti*100:5.1f}% en MAYÚSCULAS")
    print(f"  {COL_AB}: {frac_ab*100:5.1f}% en MAYÚSCULAS")
    normalizar_mayus = bool(mayus_ti or mayus_ab)
    if normalizar_mayus:
        print("  -> Se activará normalización de casing antes de los embeddings.")
    else:
        print("  -> Casing natural; no se normaliza.")

    # Enriquecimiento opcional desde columnas *_raw (verificado)
    usar_raw = False
    if CFG['recuperar_casing_desde_raw']:
        ti_r, ab_r = f"{COL_TI}_raw", f"{COL_AB}_raw"
        if ti_r in df.columns and ab_r in df.columns:
            coincide = (df[ab_r].astype(str).str.upper().str.replace(r'\s+',' ',regex=True).str.strip()
                        == df[COL_AB].astype(str).str.replace(r'\s+',' ',regex=True).str.strip())
            print(f"  Verificación {ab_r}.upper() == {COL_AB}: "
                  f"{coincide.sum()}/{len(df)}")
            if coincide.all():
                usar_raw = True
                print(f"  -> Se usará {ti_r}/{ab_r} (casing original preservado).")
            else:
                print("  -> Verificación fallida; se ignora la columna cruda.")
        else:
            print("  -> Columnas *_raw no presentes; se ignora la opción.")

    # ── 5. DUPLICADOS (verificación, no eliminación) ─────────────────────────
    print("\n── VERIFICACIÓN DE DUPLICADOS ──")
    dup_info = {}
    if 'DI' in df.columns:
        doi = df['DI'].dropna().astype(str).str.strip().str.lower()
        doi = doi[(doi != '') & (doi != 'nan')]
        dup_doi = len(doi) - doi.nunique()
        dup_info['DOI'] = dup_doi
        print(f"  DOI presentes: {len(doi):,} | duplicados: {dup_doi}")
    ti_norm = df[COL_TI].map(_norm_titulo)
    dup_ti = int(ti_norm.duplicated().sum())
    dup_info['Titulo_normalizado'] = dup_ti
    print(f"  Títulos normalizados duplicados: {dup_ti}")
    if dup_doi if 'DI' in df.columns else 0 or dup_ti:
        print("  AVISO: hay duplicados. Este script NO los elimina "
              "(se asume base ya deduplicada); revísalos si no lo esperabas.")
    else:
        print("  Sin duplicados. Base deduplicada confirmada.")

    # ── 6. TEXTO Y EXCLUSIONES ───────────────────────────────────────────────
    print("\n── CONSTRUCCIÓN DEL TEXTO ──")
    col_ti_src = f"{COL_TI}_raw" if usar_raw else COL_TI
    col_ab_src = f"{COL_AB}_raw" if usar_raw else COL_AB
    if usar_raw:
        normalizar_mayus = False

    # Scopus anexa el título original entre corchetes ("...; [标题]"), que es
    # por donde entran caracteres no latinos al corpus. Se elimina siempre.
    n_traducidos = int(df[col_ti_src].astype(str)
                       .str.contains(r';\s*\[.*\]\s*$', regex=True).sum())
    ti_src = df[col_ti_src].fillna('').astype(str).map(pr.limpiar_titulo)
    ab_src = df[col_ab_src].fillna('').astype(str)
    if n_traducidos:
        print(f"  Títulos con traducción anexa entre corchetes: {n_traducidos} "
              f"(eliminada)")

    df['texto'] = (ti_src.str.strip() + ". " + ab_src.str.strip()).str.strip()

    tiene_ti = ti_src.str.strip().str.len() > 0
    tiene_ab = ab_src.str.strip().str.len() > 0
    n_pal = df['texto'].str.split().str.len()
    suficiente = n_pal >= CFG['min_palabras_doc']

    print(f"  Con título          : {int(tiene_ti.sum()):,}")
    print(f"  Con abstract        : {int(tiene_ab.sum()):,}")
    print(f"  Con ambos           : {int((tiene_ti & tiene_ab).sum()):,}")
    print(f"  Sin texto útil      : {int((~tiene_ti & ~tiene_ab).sum()):,}")
    print(f"  < {CFG['min_palabras_doc']} palabras       : {int((~suficiente).sum()):,}")
    print(f"  Palabras: min={int(n_pal.min())} "
          f"mediana={int(n_pal.median())} max={int(n_pal.max())}")
    n_largos = int((n_pal > 350).sum())
    print(f"  > 350 palabras (riesgo de truncado a 512 tokens): {n_largos:,}")

    df_excluidos = df.loc[~suficiente, ['doc_id', COL_TI, COL_PY]].copy()
    df_excluidos['motivo'] = f"< {CFG['min_palabras_doc']} palabras"
    df_ok = df.loc[suficiente].copy().reset_index(drop=True)
    print(f"  DOCUMENTOS FINALES  : {len(df_ok):,} "
          f"(excluidos {len(df_excluidos):,})")

    # ── 7. AÑOS ──────────────────────────────────────────────────────────────
    print("\n── AÑOS ──")
    py = pd.to_numeric(df_ok[COL_PY], errors='coerce')
    n_py_malos = int(py.isna().sum())
    validos = py.between(1900, 2100)
    print(f"  No numéricos / fuera de rango: {n_py_malos + int((~validos & py.notna()).sum())}")
    print(f"  Rango: {int(py.min())}–{int(py.max())}")
    df_ok['anio'] = py.where(validos)          # NaN, nunca 0
    tabla_anios = (df_ok['anio'].dropna().astype(int).value_counts()
                   .sort_index().rename_axis('year').reset_index(name='documents'))
    tabla_anios['percentage'] = (tabla_anios['documents'] /
                                 tabla_anios['documents'].sum() * 100).round(2)
    ultimo = int(tabla_anios['year'].max())
    print(f"  ÚLTIMO AÑO = {ultimo} con {int(tabla_anios.iloc[-1]['documents'])} docs "
          f"— probablemente INCOMPLETO: márcalo en las figuras de tendencia.")

    # ── 8. PAÍSES ────────────────────────────────────────────────────────────
    print("\n── PAÍSES ──")
    seps = {s: int(df_ok[COL_CO].astype(str).str.contains(re.escape(s)).sum())
            for s in [';', '|']}
    print(f"  Documentos con separador: {seps}")
    multi = any(v > 0 for v in seps.values())
    print(f"  Estructura: {'MÚLTIPLES países por documento' if multi else 'UN país por documento'}")
    if not multi:
        print("  -> El análisis de colaboración internacional NO es posible "
              "desde esta columna (§17: no inventarlo).")

    df_ok['Country_original']   = df_ok[COL_CO]
    df_ok['Country_normalized'] = df_ok[COL_CO].map(normalizar_pais)

    cambios = (df_ok[['Country_original', 'Country_normalized']]
               .dropna()
               .loc[lambda d: d['Country_original'].astype(str).str.strip()
                    != d['Country_normalized'].astype(str).str.strip()]
               .value_counts().rename_axis(['original', 'normalizado'])
               .reset_index(name='documentos'))
    print(f"  Nombres únicos: {df_ok['Country_original'].nunique()} -> "
          f"{df_ok['Country_normalized'].nunique()} tras normalizar")
    if len(cambios):
        print("  Transformaciones aplicadas:")
        for _, r in cambios.head(10).iterrows():
            print(f"    {str(r['original']):<24} -> {r['normalizado']:<20} ({r['documentos']})")

    tabla_paises = (df_ok['Country_normalized'].dropna().value_counts()
                    .rename_axis('country').reset_index(name='documents'))
    tabla_paises['percentage'] = (tabla_paises['documents'] /
                                  tabla_paises['documents'].sum() * 100).round(2)
    print(f"  Sin país: {int(df_ok['Country_normalized'].isna().sum())}")
    print(f"  Top 5: " + ", ".join(
        f"{r['country']}({r['documents']})" for _, r in tabla_paises.head(5).iterrows()))

    # ── 9. IDIOMA (opcional) ─────────────────────────────────────────────────
    tabla_idioma = None
    if 'LA' in df_ok.columns:
        print("\n── IDIOMA (columna LA, opcional) ──")
        tabla_idioma = (df_ok['LA'].fillna('(no informado)').value_counts()
                        .rename_axis('idioma').reset_index(name='documents'))
        for _, r in tabla_idioma.head(6).iterrows():
            print(f"  {str(r['idioma']):<16} {r['documents']:>5}")
        cjk = int(df_ok['texto'].str.contains(r'[\u4e00-\u9fff]', regex=True).sum())
        cyr = int(df_ok['texto'].str.contains(r'[\u0400-\u04ff]', regex=True).sum())
        print(f"  Caracteres no latinos en el texto: CJK={cjk} cirílico={cyr}")
        if cjk == 0 and cyr == 0:
            print("  -> LA indica el idioma ORIGINAL de publicación; los "
                  "abstracts están en inglés. No se requiere estrategia multilingüe.")

    # ── 10. CANDIDATOS A STOPWORDS DE DOMINIO ────────────────────────────────
    print("\n── STOPWORDS DE DOMINIO (candidatos empíricos) ──")
    cand = pr.derivar_stopwords_dominio(df_ok['texto'].tolist(),
                                        umbral_df=0.50, top_n=60)
    tabla_stops = pd.DataFrame(cand, columns=['termino', 'frecuencia_documental'])
    tabla_stops['frecuencia_documental'] = (tabla_stops['frecuencia_documental']
                                            * 100).round(1)
    tabla_stops['aplicar'] = ''      # ← el usuario marca X manualmente
    print("  Términos presentes en >50% de los documentos:")
    for _, r in tabla_stops.head(15).iterrows():
        print(f"    {r['termino']:<20} {r['frecuencia_documental']:>5.1f}%")
    print("  -> Revisa la hoja 'Stopwords_Candidatas' y marca con 'X' la")
    print("     columna 'aplicar' en los que quieras excluir. NO se aplican solos.")

    # ── 11. EXPORTACIÓN ──────────────────────────────────────────────────────
    print("\n── EXPORTACIÓN ──")
    resumen = pd.DataFrame([
        {'metrica': 'Registros totales',        'valor': n_original},
        {'metrica': 'Con título',               'valor': int(tiene_ti.sum())},
        {'metrica': 'Con abstract',             'valor': int(tiene_ab.sum())},
        {'metrica': 'Con ambos',                'valor': int((tiene_ti & tiene_ab).sum())},
        {'metrica': 'Sin texto',                'valor': int((~tiene_ti & ~tiene_ab).sum())},
        {'metrica': 'Duplicados DOI',           'valor': dup_info.get('DOI', 'n/a')},
        {'metrica': 'Duplicados título',        'valor': dup_ti},
        {'metrica': 'Excluidos por longitud',   'valor': len(df_excluidos)},
        {'metrica': 'DOCUMENTOS UTILIZADOS',    'valor': len(df_ok)},
        {'metrica': 'Rango de años',            'valor': f"{int(py.min())}-{int(py.max())}"},
        {'metrica': 'Países (normalizados)',    'valor': int(df_ok['Country_normalized'].nunique())},
        {'metrica': 'Corpus en MAYÚSCULAS',     'valor': f"{frac_ab*100:.1f}%"},
        {'metrica': 'Normalización de casing',  'valor': normalizar_mayus},
        {'metrica': 'Docs >350 palabras',       'valor': n_largos},
    ])

    xl = os.path.join(rutas['res_corpus'], 'corpus_quality.xlsx')
    with pd.ExcelWriter(xl, engine='xlsxwriter') as w:
        resumen.to_excel(w, sheet_name='Resumen', index=False)
        tabla_anios.to_excel(w, sheet_name='Distribucion_Anual', index=False)
        tabla_paises.to_excel(w, sheet_name='Distribucion_Paises', index=False)
        if len(cambios):
            cambios.to_excel(w, sheet_name='Paises_Normalizacion', index=False)
        if tabla_idioma is not None:
            tabla_idioma.to_excel(w, sheet_name='Idiomas', index=False)
        tabla_stops.to_excel(w, sheet_name='Stopwords_Candidatas', index=False)
        if len(df_excluidos):
            df_excluidos.to_excel(w, sheet_name='Documentos_Excluidos', index=False)
        pd.DataFrame([registrar_entorno()]).T.reset_index().to_excel(
            w, sheet_name='Entorno', index=False, header=['clave', 'valor'])
    print(f"  corpus_quality.xlsx -> {xl}")

    P = pr.get_params(len(df_ok))
    cols_guardar = ['doc_id', 'texto', 'anio',
                    'Country_original', 'Country_normalized', COL_TI]
    cols_guardar += [c for c in opcionales if c not in cols_guardar]
    checkpoint = {
        'df': df_ok[cols_guardar].copy(),
        'normalizar_mayus': normalizar_mayus,
        'usar_raw': usar_raw,
        'params': P,
        'n_docs': len(df_ok),
        'entorno': registrar_entorno(),
    }
    ck = os.path.join(rutas['check'], 'ck_corpus_base.pkl')
    pickle.dump(checkpoint, open(ck, 'wb'))
    print(f"  ck_corpus_base.pkl  -> {ck}")

    # ── RESUMEN ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("SCRIPT 00 COMPLETADO")
    print(f"  Documentos utilizables : {len(df_ok):,}  [{P['escala']}]")
    print(f"  Perfil derivado        : K {P['k_min']}-{P['k_max']} | "
          f"mcs_ref={P['mcs_ref']} | mcs_grid={P['mcs_grid']}")
    print(f"  UMAP                   : {len(P['umap_grid'])} configuraciones "
          f"(min_dist=0.0)")
    print(f"  Tiempo                 : {perf_counter()-t0:.1f}s")
    print("\n── SIGUIENTE PASO ──")
    print("  1) Revisa corpus_quality.xlsx (hoja Stopwords_Candidatas)")
    print("  2) Script02b_Embeddings_Compare.py  -> elegir modelo de embeddings")
    print("  3) Script02_Grid_BERTopic_v3.py -> explorar UMAP/HDBSCAN")
    print("  4) Script02_Final_BERTopic_v3.py -> modelo final tras selección manual")
    print("=" * 66)
