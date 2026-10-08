"""
treinar.py
Treina a MLP (PyTorch) a partir de um arquivo de configuração e registra tudo no MLflow:
parâmetros, métricas por época, artefatos e um trace com os spans preparar/treinar/validar.

Uso:
  python treinar.py --config config/run_a.yaml
  python treinar.py --config config/run_a.yaml --lr 0.005 --nome_run teste-rapido   (sobrescreve)

O conjunto de TESTE é separado aqui, mas nunca é usado neste script.
Ele só é avaliado depois da escolha, com avaliar_teste.py.
"""
import argparse
import copy
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ----------------------------------------------------------------- configuração
def mesclar(base: dict, extra: dict) -> dict:
    saida = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(saida.get(k), dict):
            saida[k] = mesclar(saida[k], v)
        else:
            saida[k] = v
    return saida


def carregar_config(caminho_run: str, base: str = "config/base.yaml") -> dict:
    cfg = yaml.safe_load(Path(base).read_text(encoding="utf-8"))
    run = yaml.safe_load(Path(caminho_run).read_text(encoding="utf-8")) or {}
    return mesclar(cfg, run)


def fixar_sementes(semente: int):
    # O módulo "random" do Python NÃO é semeado de propósito: o MLflow usa esse
    # gerador para criar os IDs dos traces. Com random.seed(42) em todas as runs,
    # todos os traces recebiam o mesmo ID e se sobrescreviam. O treino não usa
    # "random": o split usa random_state e a rede usa as sementes do NumPy e do PyTorch.
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ----------------------------------------------------------------- dados
def dividir(cfg: dict):
    """Split estratificado e determinístico. Reutilizado por avaliar_teste.py."""
    df = pd.read_csv(cfg["dados"]["arquivo"])
    alvo = cfg["dados"]["coluna_alvo"]
    entradas = [c for c in df.columns if c not in (alvo, "pais")]
    semente = cfg["semente"]
    estrat = cfg["split"]["estratificado"]

    resto, teste = train_test_split(
        df, test_size=cfg["split"]["teste"], random_state=semente,
        stratify=df[alvo] if estrat else None)
    val_rel = cfg["split"]["validacao"] / (1 - cfg["split"]["teste"])
    treino, val = train_test_split(
        resto, test_size=val_rel, random_state=semente,
        stratify=resto[alvo] if estrat else None)
    return treino, val, teste, entradas, alvo


def ajustar_preprocessamento(treino, entradas):
    """Imputação e padronização ajustadas SÓ no treino (evita vazamento)."""
    prep = Pipeline([("imputar", SimpleImputer(strategy="median")),
                     ("padronizar", StandardScaler())])
    prep.fit(treino[entradas])
    return prep


def para_tensor(prep, df, entradas, alvo, codificador):
    X = torch.tensor(prep.transform(df[entradas]), dtype=torch.float32)
    y = torch.tensor(codificador.transform(df[alvo]), dtype=torch.long)
    return X, y


# ----------------------------------------------------------------- modelo
class MLP(nn.Module):
    def __init__(self, n_entradas, camadas, n_classes):
        super().__init__()
        blocos, anterior = [], n_entradas
        for n in camadas:
            blocos += [nn.Linear(anterior, n), nn.ReLU()]
            anterior = n
        blocos.append(nn.Linear(anterior, n_classes))
        self.rede = nn.Sequential(*blocos)

    def forward(self, x):
        return self.rede(x)


def avaliar(modelo, X, y, perda_fn):
    modelo.eval()
    with torch.no_grad():
        logits = modelo(X)
        perda = perda_fn(logits, y).item()
        pred = logits.argmax(dim=1).numpy()
    return perda, pred


# ----------------------------------------------------------------- utilidades
def commit_git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "sem-git"


def salvar_curvas(hist, caminho):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(hist["train_loss"], label="treino")
    a1.plot(hist["val_loss"], label="validação")
    a1.set_title("Perda por época"); a1.set_xlabel("época"); a1.legend()
    a2.plot(hist["val_accuracy"], color="green")
    a2.set_title("Acurácia de validação"); a2.set_xlabel("época"); a2.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(caminho, dpi=120); plt.close(fig)


def salvar_matriz(y_true, y_pred, classes, caminho, titulo):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False); ax.set_title(titulo)
    fig.tight_layout(); fig.savefig(caminho, dpi=120); plt.close(fig)
    return cm


# ----------------------------------------------------------------- execução
def executar(cfg: dict):
    fixar_sementes(cfg["semente"])
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experimento"])
    t = cfg["treino"]

    with mlflow.start_run(run_name=cfg.get("nome_run", "sem-nome")) as run:
        # Parâmetros (o que define a run) e tags (ambiente de execução)
        mlflow.log_params({
            "taxa_aprendizado": t["taxa_aprendizado"], "l2": t["l2"], "epocas": t["epocas"],
            "otimizador": "adam", "camadas_ocultas": str(cfg["modelo"]["camadas_ocultas"]),
            "semente": cfg["semente"], "split_teste": cfg["split"]["teste"],
            "split_validacao": cfg["split"]["validacao"],
            "estratificado": cfg["split"]["estratificado"],
            "dataset": cfg["dados"]["arquivo"],
        })
        mlflow.set_tags({"commit_git": commit_git(), "python": platform.python_version(),
                         "torch": torch.__version__, "mlflow": mlflow.__version__,
                         "avaliado_no_teste": "nao"})

        with mlflow.start_span(name="pipeline_treinamento") as raiz:
            raiz.set_inputs({"run": cfg.get("nome_run"), "lr": t["taxa_aprendizado"], "l2": t["l2"]})

            # ---------------- PREPARAR
            with mlflow.start_span(name="preparar") as span:
                treino, val, teste, entradas, alvo = dividir(cfg)
                prep = ajustar_preprocessamento(treino, entradas)
                cod = LabelEncoder().fit(pd.concat([treino, val, teste])[alvo])
                Xtr, ytr = para_tensor(prep, treino, entradas, alvo, cod)
                Xva, yva = para_tensor(prep, val, entradas, alvo, cod)
                span.set_outputs({"n_treino": len(treino), "n_validacao": len(val),
                                  "n_teste_reservado": len(teste), "n_entradas": len(entradas),
                                  "classes": list(cod.classes_)})

            # ---------------- TREINAR
            with mlflow.start_span(name="treinar") as span:
                modelo = MLP(len(entradas), cfg["modelo"]["camadas_ocultas"], len(cod.classes_))
                otim = torch.optim.Adam(modelo.parameters(), lr=t["taxa_aprendizado"],
                                        weight_decay=t["l2"])  # weight_decay = penalidade L2
                perda_fn = nn.CrossEntropyLoss()
                hist = {"train_loss": [], "val_loss": [], "val_accuracy": []}
                inicio = time.time()
                for epoca in range(t["epocas"]):
                    modelo.train()
                    otim.zero_grad()
                    perda = perda_fn(modelo(Xtr), ytr)
                    perda.backward()
                    otim.step()

                    vl, vpred = avaliar(modelo, Xva, yva, perda_fn)
                    va = accuracy_score(yva.numpy(), vpred)
                    tl = perda.item()
                    if not np.isfinite(tl):
                        raise RuntimeError(f"Perda de treino virou {tl} na época {epoca} "
                                           "(divergência - verifique a taxa de aprendizado).")
                    hist["train_loss"].append(tl); hist["val_loss"].append(vl)
                    hist["val_accuracy"].append(va)
                    mlflow.log_metrics({"train_loss": tl, "val_loss": vl, "val_accuracy": va},
                                       step=epoca)
                span.set_outputs({"epocas": t["epocas"], "segundos": round(time.time() - inicio, 2),
                                  "train_loss_final": tl, "val_loss_final": vl})

            # ---------------- VALIDAR
            with mlflow.start_span(name="validar") as span:
                vl, vpred = avaliar(modelo, Xva, yva, perda_fn)
                f1 = f1_score(yva.numpy(), vpred, average="macro")
                acc = accuracy_score(yva.numpy(), vpred)
                melhor_epoca = int(np.argmin(hist["val_loss"]))
                mlflow.log_metrics({"val_f1_macro_final": f1, "val_accuracy_final": acc,
                                    "val_loss_final": vl, "melhor_epoca_val_loss": melhor_epoca,
                                    "gap_perda_val_menos_treino": vl - hist["train_loss"][-1]})

                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    salvar_curvas(hist, tmp / "curvas.png")
                    cm = salvar_matriz(yva.numpy(), vpred, cod.classes_,
                                       tmp / "matriz_confusao_validacao.png", "Validação")
                    torch.save(modelo.state_dict(), tmp / "modelo.pt")
                    joblib.dump({"preprocessamento": prep, "codificador": cod,
                                 "entradas": entradas}, tmp / "preprocessamento.joblib")
                    resumo_prep = {
                        "imputacao": "mediana, ajustada só no treino",
                        "padronizacao": "StandardScaler, ajustado só no treino",
                        "medianas_treino": dict(zip(entradas, prep.named_steps["imputar"].statistics_.round(4).tolist())),
                        "medias_treino": dict(zip(entradas, prep.named_steps["padronizar"].mean_.round(4).tolist())),
                        "paises_treino": treino["pais"].tolist(),
                        "paises_validacao": val["pais"].tolist(),
                        "paises_teste_reservado": teste["pais"].tolist(),
                    }
                    (tmp / "resumo_preprocessamento.json").write_text(
                        json.dumps(resumo_prep, indent=2, ensure_ascii=False), encoding="utf-8")
                    (tmp / "config_efetiva.yaml").write_text(
                        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
                    mlflow.log_artifacts(str(tmp))
                if Path(cfg["dados"]["resumo"]).exists():
                    mlflow.log_artifact(cfg["dados"]["resumo"])

                span.set_outputs({"val_accuracy": acc, "val_f1_macro": f1,
                                  "matriz_confusao": cm.tolist()})
            raiz.set_outputs({"run_id": run.info.run_id, "val_f1_macro": f1})

        print(f"[{cfg.get('nome_run')}] run_id={run.info.run_id} | "
              f"val_acc={acc:.3f} | val_f1={f1:.3f} | val_loss={vl:.4f}")
        return run.info.run_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--lr", type=float); p.add_argument("--l2", type=float)
    p.add_argument("--epocas", type=int); p.add_argument("--nome_run")
    a = p.parse_args()
    cfg = carregar_config(a.config)
    if a.lr is not None: cfg["treino"]["taxa_aprendizado"] = a.lr
    if a.l2 is not None: cfg["treino"]["l2"] = a.l2
    if a.epocas is not None: cfg["treino"]["epocas"] = a.epocas
    if a.nome_run: cfg["nome_run"] = a.nome_run
    executar(cfg)


if __name__ == "__main__":
    main()
