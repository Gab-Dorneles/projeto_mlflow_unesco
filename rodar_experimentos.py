"""
rodar_experimentos.py
Um comando, três configurações: executa as runs A, B e C (mesma arquitetura,
split, semente e épocas; muda só taxa de aprendizado e L2).

Uso:
  python rodar_experimentos.py                 # A, B e C
  python rodar_experimentos.py --incidente     # A, B, C + run D do incidente
"""
import argparse
import traceback

from treinar import carregar_config, executar

RUNS = ["config/run_a.yaml", "config/run_b.yaml", "config/run_c.yaml"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--incidente", action="store_true", help="Inclui a run D (taxa alta)")
    a = p.parse_args()
    lista = RUNS + (["config/run_incidente.yaml"] if a.incidente else [])

    resultados = {}
    for caminho in lista:
        cfg = carregar_config(caminho)
        print(f"\n=== {cfg['nome_run']} ({caminho}) ===")
        try:
            resultados[cfg["nome_run"]] = executar(cfg)
        except Exception as e:  # a run falha, mas fica registrada no MLflow (status FAILED)
            print(f"Run {cfg['nome_run']} falhou: {e}")
            traceback.print_exc()
            resultados[cfg["nome_run"]] = "FALHOU"

    print("\nResumo:")
    for nome, rid in resultados.items():
        print(f"  {nome}: {rid}")
    print("\nAbra a interface com:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
