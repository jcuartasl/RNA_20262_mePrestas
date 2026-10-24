"""
src/data/leakage.py
Módulo para auditoría sistemática y eliminación de fuga de información (data leakage).
Curso: Redes Neuronales y Algoritmos Bioinspirados - Universidad Nacional de Colombia.
"""

from typing import Dict, List
import pandas as pd


def get_leakage_columns(strict: bool = True) -> Dict[str, List[str]]:
    """
    Retorna el diccionario de variables prohibidas organizadas por categoría de fuga.

    Parámetros:
    -----------
    strict : bool, default=True
        Si es True, incluye las variables de scoring interno de Lending Club
        (grade, sub_grade, int_rate, installment) para garantizar coherencia
        con un motor de pre-aprobación / admisión sin sesgo heredado.
    """
    leakage_dict = {
        "pagos_acumulados": [
            "total_pymnt",
            "total_pymnt_inv",
            "total_rec_prncp",
            "total_rec_int",
            "total_rec_late_fee",
        ],
        "eventos_post_default": [
            "recoveries",
            "collection_recovery_fee",
        ],
        "saldos_dinamicos": [
            "out_prncp",
            "out_prncp_inv",
        ],
        "fechas_posteriores": [
            "last_pymnt_d",
            "last_pymnt_amnt",
            "next_pymnt_d",
            "last_credit_pull_d",
        ],
        "identificadores_alta_cardinalidad": [
            "id",
            "member_id",
            "url",
            "desc",
            "title",
            "emp_title",
            "zip_code",
        ],
    }

    if strict:
        leakage_dict["scoring_interno_lc"] = [
            "grade",
            "sub_grade",
            "int_rate",
            "installment",
        ]

    return leakage_dict


def get_leakage_justifications() -> Dict[str, Dict[str, str]]:
    """
    Retorna las justificaciones operativas y de negocio para cada categoría de fuga.
    """
    return {
        "pagos_acumulados": {
            "motivo": "Pagos acumulados a lo largo de la vida del crédito.",
            "justificacion": (
                "Al momento de originación y pre-aprobación el cliente aún no ha realizado ningún abono. "
                "Incluir pagos acumulados permitiría al modelo identificar clientes cumplidos por el simple "
                "hecho de haber completado sus pagos en el tiempo, generando una fuga de información masiva y fraudulenta."
            ),
        },
        "eventos_post_default": {
            "motivo": "Montos de recuperación y tarifas de cobro tras el cargo a pérdidas.",
            "justificacion": (
                "Estas variables solo toman valores positivos cuando el crédito ya fue castigado y remitido a cobranza judicial. "
                "Son esenciales para cuantificar la pérdida real en LGD, pero su uso en PD predeciría el default con certeza retrospectiva."
            ),
        },
        "saldos_dinamicos": {
            "motivo": "Saldo insoluto o remanente vigente del principal.",
            "justificacion": (
                "En la originación el saldo pendiente es exactamente igual al monto financiado. "
                "Cualquier reducción refleja pagos posteriores al desembolso que no existen en la admisión."
            ),
        },
        "fechas_posteriores": {
            "motivo": "Eventos de pago y consultas de buró realizadas durante la vigencia del crédito.",
            "justificacion": (
                "Fechas de último pago o último chequeo de buró ocurren meses o años después de que la decisión "
                "de otorgamiento fue tomada. No forman parte del conjunto de información disponible en el momento de decidir."
            ),
        },
        "identificadores_alta_cardinalidad": {
            "motivo": "Identificadores únicos o campos de texto libre de altísima variabilidad.",
            "justificacion": (
                "Carecen de capacidad predictiva generalizable, inducen sobreajuste memorístico "
                "y violan principios de anonimización y privacidad crediticia."
            ),
        },
        "scoring_interno_lc": {
            "motivo": "Rating interno y tasa comercial asignados por Lending Club.",
            "justificacion": (
                "En un caso de uso estricto de pre-aprobación/admisión, nuestro modelo busca evaluar el riesgo intrínseco "
                "del solicitante antes de fijar una tasa o categoría. Incluirlas haría que la red aprenda el scoring ajeno "
                "de Lending Club en lugar de modelar el riesgo real a partir del historial y perfil crediticio."
            ),
        },
    }


def remove_leakage(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    Elimina todas las variables identificadas como fuga de información del DataFrame.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame de créditos limpios.
    strict : bool, default=True
        Si es True, también elimina grade, sub_grade, int_rate e installment.

    Retorna:
    --------
    pd.DataFrame sin variables de fuga.
    """
    leakage_dict = get_leakage_columns(strict=strict)
    all_leakage_cols = set()
    for cols in leakage_dict.values():
        all_leakage_cols.update(cols)

    cols_to_drop = [c for c in all_leakage_cols if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def generate_leakage_report(df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """
    Genera un informe detallado con cada variable candidata a fuga, indicando
    su categoría, si está presente en el conjunto de datos y su justificación metodológica.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame de referencia para auditar columnas presentes.
    output_path : str, opcional
        Ruta para guardar el reporte en CSV (ej. results/tables/leakage_report.csv).
    """
    leakage_dict = get_leakage_columns(strict=True)
    justifications = get_leakage_justifications()

    rows = []
    for category, cols in leakage_dict.items():
        cat_info = justifications.get(category, {"motivo": "Fuga", "justificacion": "No disponible en pre-aprobación."})
        for col in cols:
            present = col in df.columns
            rows.append({
                "columna": col,
                "categoria_fuga": category,
                "presente_en_datos": present,
                "motivo": cat_info["motivo"],
                "justificacion_negocio": cat_info["justificacion"],
                "tratamiento": "Excluir en PD; retener variables de recuperación para LGD" if category == "eventos_post_default" or col == "total_rec_prncp" else "Excluir completamente para pre-aprobación de PD"
            })

    report_df = pd.DataFrame(rows)

    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Reporte de fuga guardado exitosamente en: {output_path}")

    return report_df


if __name__ == "__main__":
    test_cols = [
        "loan_amnt", "funded_amnt", "dti", "annual_inc",
        "total_pymnt", "recoveries", "grade", "int_rate"
    ]
    sample = pd.DataFrame(columns=test_cols)
    print("Columnas originales de prueba:", sample.columns.tolist())

    strict_df = remove_leakage(sample, strict=True)
    print("Columnas tras remove_leakage(strict=True):", strict_df.columns.tolist())

    rep = generate_leakage_report(sample)
    print(f"Reporte de auditoría generado con {len(rep)} variables catalogadas.")
