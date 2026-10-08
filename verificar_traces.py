"""
verificar_traces.py
Lista os traces gravados no mlflow.db e mostra a qual run cada um pertence.

Uso:
  python verificar_traces.py
"""
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
exp = mlflow.get_experiment_by_name("mlp-unesco-rd-opri")
if exp is None:
    raise SystemExit("Experimento 'mlp-unesco-rd-opri' não encontrado no mlflow.db.")

# Nome de cada run, para facilitar a leitura
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], run_view_type=mlflow.entities.ViewType.ALL)
nomes = dict(zip(runs["run_id"], runs["tags.mlflow.runName"]))

traces = mlflow.search_traces(experiment_ids=[exp.experiment_id], return_type="list", max_results=500)
print(f"Runs no experimento: {len(runs)}")
print(f"Traces gravados:     {len(traces)}\n")

for t in traces:
    meta = getattr(t.info, "trace_metadata", None) or getattr(t.info, "request_metadata", {}) or {}
    run_id = meta.get("mlflow.sourceRun", "sem run associada")
    spans = [s.name for s in t.data.spans]
    print(f"- trace {t.info.trace_id}")
    print(f"    run:   {nomes.get(run_id, run_id)}")
    print(f"    spans: {spans}")

sem_trace = set(nomes) - {
    (getattr(t.info, "trace_metadata", None) or getattr(t.info, "request_metadata", {}) or {}).get("mlflow.sourceRun")
    for t in traces
}
if sem_trace:
    print("\nRuns SEM trace:", ", ".join(nomes[r] for r in sem_trace))
