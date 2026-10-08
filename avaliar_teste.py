"""
avaliar_teste.py
Avalia UMA ÚNICA VEZ, no conjunto de teste reservado, o modelo da run escolhida.
O resultado é gravado na própria run (métricas test_*) e a run recebe a tag
avaliado_no_teste=sim. Se já tiver sido avaliada, o script se recusa a repetir.

Uso:
  python avaliar_teste.py --run_id <RUN_ID_ESCOLHIDA>
"""
import argparse
import tempfile
from pathlib import Path

import joblib
import mlflow
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score

from treinar import MLP, dividir, salvar_matriz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True)
    p.add_argument("--tracking_uri", default="sqlite:///mlflow.db")
    a = p.parse_args()

    mlflow.set_tracking_uri(a.tracking_uri)
    run = mlflow.get_run(a.run_id)
    if run.data.tags.get("avaliado_no_teste") == "sim":
        raise SystemExit("Esta run já foi avaliada no teste. O teste só deve ser usado uma vez.")

    with tempfile.TemporaryDirectory() as tmp:
        pasta = Path(mlflow.artifacts.download_artifacts(run_id=a.run_id, dst_path=tmp))
        cfg = yaml.safe_load((pasta / "config_efetiva.yaml").read_text(encoding="utf-8"))
        pacote = joblib.load(pasta / "preprocessamento.joblib")
        prep, cod, entradas = pacote["preprocessamento"], pacote["codificador"], pacote["entradas"]

        _, _, teste, _, alvo = dividir(cfg)  # mesmo split determinístico da run
        X = torch.tensor(prep.transform(teste[entradas]), dtype=torch.float32)
        y = cod.transform(teste[alvo])

        modelo = MLP(len(entradas), cfg["modelo"]["camadas_ocultas"], len(cod.classes_))
        modelo.load_state_dict(torch.load(pasta / "modelo.pt"))
        modelo.eval()
        with torch.no_grad():
            logits = modelo(X)
            perda = nn.CrossEntropyLoss()(logits, torch.tensor(y)).item()
            pred = logits.argmax(1).numpy()

        acc, f1 = accuracy_score(y, pred), f1_score(y, pred, average="macro")
        relatorio = classification_report(y, pred, labels=range(len(cod.classes_)),
                                          target_names=cod.classes_, zero_division=0)

        with mlflow.start_run(run_id=a.run_id):
            with mlflow.start_span(name="testar_modelo_escolhido") as span:
                mlflow.log_metrics({"test_accuracy": acc, "test_f1_macro": f1, "test_loss": perda})
                salvar_matriz(y, pred, cod.classes_, Path(tmp) / "matriz_confusao_teste.png", "Teste")
                (Path(tmp) / "relatorio_teste.txt").write_text(relatorio, encoding="utf-8")
                mlflow.log_artifact(str(Path(tmp) / "matriz_confusao_teste.png"))
                mlflow.log_artifact(str(Path(tmp) / "relatorio_teste.txt"))
                mlflow.set_tags({"avaliado_no_teste": "sim", "modelo_escolhido": "sim"})
                span.set_outputs({"n_teste": len(y), "test_accuracy": acc, "test_f1_macro": f1})

    print(f"Teste ({len(y)} países): acurácia={acc:.3f} | F1 macro={f1:.3f} | perda={perda:.4f}")
    print(relatorio)


if __name__ == "__main__":
    main()
