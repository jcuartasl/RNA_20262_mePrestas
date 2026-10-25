"""MLP de LGD y análisis previo de la distribución de la LGD que justifica la pérdida y la salida."""
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_splits
from src.models.lgd_linear import predict_lgd_linear, train_lgd_linear
from src.models.losses import beta_con_logits, get_loss, penalizacion_l1, penalizacion_l2
from src.models.mlp import build_mlp
from src.utils.config import ROOT


def _opciones(cfg):
    """Traduce la config anidada de P5 (model.hidden, train.lr, loss.name...) a valores planos."""
    modelo, entrenamiento, perdida = cfg.get("model", {}), cfg.get("train", {}), cfg.get("loss", {})
    # el atajo de la guía (--override loss=beta) deja loss como texto en vez de diccionario
    if isinstance(perdida, str): perdida = {"name": perdida}
    # los valores por defecto hacen falta porque la demo de P5 solo trae train.epochs y sample_fraction
    return {"hidden_layers": list(modelo.get("hidden", [64, 32])),
            "activation": modelo.get("activation", "relu"),
            "dropout": float(modelo.get("dropout", 0.2)), "perdida": perdida.get("name", "mse"),
            "eps": float(perdida.get("eps", 1e-4)), "lr": float(entrenamiento.get("lr", 1e-3)),
            "epochs": int(entrenamiento.get("epochs", 100)),
            "batch_size": int(entrenamiento.get("batch_size", 256)),
            "patience": int(entrenamiento.get("patience", 10)), "l1": float(entrenamiento.get("l1", 0.0)),
            "l2": float(entrenamiento.get("l2", 0.0))}


def entrenar_lgd_mlp(X_train, y_train, X_val, y_val, cfg, seed=0):
    """Entrena el mlp de lgd con adam, l1/l2 y parada temprana; devuelve modelo, log_phi e historial."""
    op = _opciones(cfg)
    torch.manual_seed(seed)
    X_train, X_val = (torch.tensor(np.asarray(X), dtype=torch.float32) for X in (X_train, X_val))
    # el mlp devuelve (n, 1) y las pérdidas exigen que el target tenga la misma forma
    y_train, y_val = (torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1) for y in (y_train, y_val))
    modelo = build_mlp({"input_dim": X_train.shape[1], "hidden_layers": op["hidden_layers"],
                        "activation": op["activation"], "dropout": op["dropout"], "output_activation": "none"})
    perdida = get_loss(op["perdida"])
    # phi de la beta se aprende junto con los pesos; arranca en 4 por los momentos de la lgd de train
    log_phi = None
    if op["perdida"] == "beta": log_phi = torch.tensor(np.log(4.0), dtype=torch.float32, requires_grad=True)
    parametros = list(modelo.parameters()) + ([log_phi] if log_phi is not None else [])
    optimizador = torch.optim.Adam(parametros, lr=op["lr"])
    lotes = DataLoader(TensorDataset(X_train, y_train), batch_size=op["batch_size"], shuffle=True,
                       generator=torch.Generator().manual_seed(seed))

    def perdida_datos(logits, y):
        return perdida(logits, y, log_phi, op["eps"]) if log_phi is not None else perdida(logits, y)

    historial, mejor_val, mejor_estado, sin_mejora = {"train_loss": [], "val_loss": []}, np.inf, None, 0
    for epoca in range(op["epochs"]):
        modelo.train()
        total = 0.0
        for X_lote, y_lote in lotes:
            optimizador.zero_grad()
            valor = perdida_datos(modelo(X_lote), y_lote)
            if op["l1"]: valor = valor + op["l1"] * penalizacion_l1(modelo)
            if op["l2"]: valor = valor + op["l2"] * penalizacion_l2(modelo)
            valor.backward()
            optimizador.step()
            total += valor.item() * len(X_lote)
        # sin eval() el dropout seguiría activo y la pérdida de validación saldría ruidosa
        modelo.eval()
        with torch.no_grad(): val = perdida_datos(modelo(X_val), y_val).item()
        historial["train_loss"].append(total / len(X_train))
        historial["val_loss"].append(val)
        if val < mejor_val:
            mejor_val, sin_mejora = val, 0
            phi_actual = None if log_phi is None else log_phi.item()
            mejor_estado = (copy.deepcopy(modelo.state_dict()), phi_actual, epoca + 1)
        else:
            sin_mejora += 1
        if sin_mejora >= op["patience"]: break
    # parada temprana: se vuelve a los pesos de la mejor época, no a los de la última
    modelo.load_state_dict(mejor_estado[0])
    if log_phi is not None:
        with torch.no_grad(): log_phi.fill_(mejor_estado[1])
    modelo.eval()
    historial.update(best_epoch=mejor_estado[2], best_val_loss=mejor_val,
                     epochs_trained=len(historial["val_loss"]))
    return modelo, log_phi, historial


@torch.no_grad()
def predecir_lgd(modelo, X):
    """LGD predicha en [0, 1]: sigmoide del logit, con el dropout apagado."""
    modelo.eval()
    return torch.sigmoid(modelo(torch.tensor(np.asarray(X), dtype=torch.float32))).numpy().ravel()


def contar_parametros(modelo, log_phi=None):
    """Parámetros entrenables del mlp, más 1 si se aprendió el phi de la beta."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad) + (log_phi is not None)


def metricas_lgd(y, p):
    """Rmse, mae y r2 (provisional hasta que exista metrics.py de P4)."""
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    error = y - p
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
            "r2": float(1 - np.sum(error ** 2) / np.sum((y - y.mean()) ** 2))}


def guardar_lgd_mlp(modelo, log_phi, cfg, ruta):
    """Guarda arquitectura, pesos, pérdida y log_phi, con el mismo formato base que pd_export.py de P2."""
    op = _opciones(cfg)
    arquitectura = {"input_dim": modelo.network[0].in_features, "hidden_layers": op["hidden_layers"],
                    "activation": op["activation"], "dropout": op["dropout"], "output_activation": "none"}
    pesos = {k: v.detach().cpu() for k, v in modelo.state_dict().items()}
    torch.save({"architecture": arquitectura, "state_dict": pesos, "perdida": op["perdida"],
                "log_phi": None if log_phi is None else log_phi.item()}, ruta)


def cargar_lgd_mlp(ruta):
    """Devuelve (modelo, log_phi, nombre de la pérdida) sin reentrenar."""
    guardado = torch.load(ruta, map_location="cpu", weights_only=True)
    modelo = build_mlp(guardado["architecture"])
    modelo.load_state_dict(guardado["state_dict"])
    modelo.eval()
    return modelo, guardado["log_phi"], guardado["perdida"]


def ajustar_beta_marginal(y, eps=1e-4):
    """Ajusta una beta(mu, phi) constante por máxima verosimilitud con la pérdida beta de losses.py."""
    y = torch.as_tensor(y, dtype=torch.float64)
    z = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    log_phi = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizador = torch.optim.LBFGS([z, log_phi], max_iter=500, line_search_fn="strong_wolfe")

    def paso():
        optimizador.zero_grad()
        perdida = beta_con_logits(z.expand(len(y)), y, log_phi, eps)
        perdida.backward()
        return perdida

    optimizador.step(paso)
    return torch.sigmoid(z).item(), log_phi.exp().item()


def figuras_distribucion_lgd(carpeta=ROOT / "results" / "figures"):
    """Genera las tres figuras de la sección 3.3 (lgd de train aleatorio) y devuelve las cifras citadas."""
    datos = load_splits("lgd", "random")
    y, n = datos["y_train"].astype(np.float64), len(datos["y_train"])
    mu, phi = ajustar_beta_marginal(y)
    a, b, media, sd = mu * phi, (1 - mu) * phi, y.mean(), y.std()
    cifras = {"n": n, "media": media, "sd": sd, "asimetria": stats.skew(y), "mu_beta": mu, "phi_beta": phi,
              "loglik_beta": stats.beta.logpdf(np.clip(y, 1e-4, 1 - 1e-4), a, b).mean(),
              "loglik_normal": stats.norm.logpdf(y, media, sd).mean(),
              "normal_fuera_de_0_1": stats.norm.cdf(0, media, sd) + stats.norm.sf(1, media, sd)}

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(y, bins=50, density=True, color="lightgray", edgecolor="gray", label="LGD observada")
    malla, malla_amplia = np.linspace(0.001, 0.999, 500), np.linspace(-0.3, 1.3, 500)
    ax.plot(malla, stats.beta.pdf(malla, a, b), lw=2, label=f"Beta MLE (μ={mu:.3f}, φ={phi:.2f})")
    ax.plot(malla_amplia, stats.norm.pdf(malla_amplia, media, sd), "--", lw=2,
            label=f"Normal MLE (μ={media:.3f}, σ={sd:.3f})")
    # la parte de la normal fuera de [0, 1] son lgd imposibles
    ax.axvspan(-0.3, 0, color="red", alpha=0.08)
    ax.axvspan(1, 1.3, color="red", alpha=0.08, label="fuera de [0, 1]")
    ax.set(xlabel="LGD", ylabel="Densidad",
           title="LGD observada frente a las densidades Beta y normal ajustadas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "lgd_ajuste_beta_normal.png", dpi=300)
    plt.close(fig)

    probabilidades, empiricos = (np.arange(1, n + 1) - 0.5) / n, np.sort(y)
    fig, ejes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, nombre, teoricos in [(ejes[0], "Beta", stats.beta.ppf(probabilidades, a, b)),
                                 (ejes[1], "Normal", stats.norm.ppf(probabilidades, media, sd))]:
        ax.plot(teoricos, empiricos, ".", ms=2)
        ax.plot([-0.3, 1.3], [-0.3, 1.3], "k--", lw=1)
        ax.set(xlabel=f"Cuantiles teóricos ({nombre})", title=f"QQ contra la {nombre}", xlim=(-0.3, 1.3))
    ejes[0].set_ylabel("Cuantiles observados de la LGD")
    fig.tight_layout()
    fig.savefig(carpeta / "lgd_qq_beta_normal.png", dpi=300)
    plt.close(fig)

    # la media condicional se aproxima con la lineal de P2; se agrupa en deciles de esa predicción
    lineal, metricas = train_lgd_linear(datos["X_train"], datos["y_train"], datos["X_val"], datos["y_val"])
    p_val, y_val = predict_lgd_linear(lineal, datos["X_val"]), datos["y_val"].astype(np.float64)
    decil = pd.qcut(predict_lgd_linear(lineal, datos["X_train"]), 10, labels=False)
    grupos = pd.DataFrame({"y": y, "decil": decil}).groupby("decil")["y"]
    medias, varianzas = grupos.mean().to_numpy(), grupos.var().to_numpy()
    # cada supuesto se escala para que su varianza promedio coincida con la observada; solo se compara la forma
    forma = medias * (1 - medias)
    escala = varianzas.mean() / forma.mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(medias, varianzas, "o-", lw=2, label="varianza observada por decil")
    ax.axhline(varianzas.mean(), ls="--", color="gray", label="MSE: varianza constante")
    ax.plot(medias, escala * forma, "s--", label="BCE fraccional y Beta (φ constante): ∝ μ(1−μ)")
    ax.set(xlabel="LGD media del decil", ylabel="Varianza de la LGD dentro del decil",
           title="Varianza de la LGD según su media (deciles de la predicción lineal)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "lgd_varianza_por_decil.png", dpi=300)
    plt.close(fig)

    cifras.update(r2_lineal_val=1 - ((y_val - p_val) ** 2).sum() / ((y_val - y_val.mean()) ** 2).sum(),
                  rmse_lineal_val=metricas["rmse"], sd_pred_lineal_val=p_val.std(), sd_y_val=y_val.std(),
                  medias_decil=medias.round(3).tolist(), varianzas_decil=varianzas.round(4).tolist(),
                  cociente_var_forma=(varianzas / forma).round(3).tolist(),
                  cuantiles={q: (np.quantile(y, q), stats.beta.ppf(q, a, b), stats.norm.ppf(q, media, sd))
                             for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]})
    return cifras


if __name__ == "__main__":
    for clave, valor in figuras_distribucion_lgd().items(): print(f"{clave}: {valor}")
