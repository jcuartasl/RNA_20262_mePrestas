"""Pruebas de la importancia por permutación."""
import numpy as np
import pytest

from src.interpret.permutation import grupos_por_variable, importancia_permutacion


def test_grupos_numericas_y_categoricas():
    cols = ["funded_amnt", "funded_amnt_inv", "term_ 36 months", "term_ 60 months", "emp_length_missing"]
    g = grupos_por_variable(cols, ["funded_amnt", "funded_amnt_inv", "term", "emp_length"])
    assert g == {"funded_amnt": [0], "funded_amnt_inv": [1], "term": [2, 3], "emp_length": [4]}


def test_grupos_extra_y_columna_desconocida():
    g = grupos_por_variable(["a", "b", "c"], ["a", "b", "c"], {"ab": ["a", "b"]})
    assert g == {"c": [2], "ab": [0, 1]}       # a y b ya no aparecen solas
    with pytest.raises(ValueError):
        grupos_por_variable(["zzz"], ["a"])


def test_solo_importa_la_variable_que_genera_y():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3))
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-2 * X[:, 0]))).astype(int)
    tabla = importancia_permutacion(lambda M: M[:, 0], X, y, {"x0": [0], "x1": [1], "x2": [2]}, n_repeticiones=5)
    imp = tabla.set_index("variable")["importancia_media"]
    assert imp["x0"] > 0.2 and abs(imp["x1"]) < 1e-12 and abs(imp["x2"]) < 1e-12
    assert tabla.set_index("variable").loc["x0", "direccion_corr"] > 0.99


def test_permutacion_conjunta_conserva_one_hot():
    rng = np.random.default_rng(1)
    cat = rng.integers(0, 3, size=500)
    X = np.eye(3)[cat]
    vistos = []

    def predecir(M):
        vistos.append(M.sum(axis=1))
        return M[:, 0]
    importancia_permutacion(predecir, X, (cat == 0).astype(int), {"cat": [0, 1, 2]}, n_repeticiones=3)
    assert all(np.all(s == 1) for s in vistos)
