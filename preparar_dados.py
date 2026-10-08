"""
preparar_dados.py
Transforma o CSV do UNESCO DataHub (R&D – Other Policy Relevant Indicators, uis005)
em uma tabela de classificação: uma linha por país, uma coluna por indicador.

Uso:
  1) Explorar indicadores disponíveis (cobertura por país):
     python preparar_dados.py --entrada data/raw/opri.csv --listar

  2) Gerar o dataset (escolhendo o indicador-alvo pelo ID visto no passo 1):
     python preparar_dados.py --entrada data/raw/opri.csv --alvo <ID_DO_INDICADOR>

O que este script NÃO faz (de propósito): imputar valores faltantes ou padronizar.
Essas etapas aprendem com os dados e, por isso, são ajustadas SÓ no conjunto de treino,
dentro de treinar.py, depois do split. Fazer aqui causaria vazamento de informação.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

# Nomes possíveis de colunas (BDDS da UIS e exportação do DataHub variam).
CANDIDATAS = {
    "indicador": ["indicatorid", "indicator", "indicatorcode", "indicador"],
    "pais": ["countryid", "geounit", "geounitid", "country", "countrycode", "iso3", "pais"],
    "ano": ["year", "timeperiod", "ano"],
    "valor": ["value", "obsvalue", "valor"],
    "rotulo": ["indicatorlabelen", "indicatorlabel", "indicatorname", "label", "rotulo"],
}

# Palavras que indicam variáveis derivadas do próprio alvo (participação feminina).
# Elas são excluídas das entradas para evitar vazamento.
PALAVRAS_VAZAMENTO = ["female", "women", "woman", "feminin", "mulher"]


def normalizar(nome: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(nome).lower())


def detectar_colunas(df: pd.DataFrame, forcadas: dict) -> dict:
    mapa_norm = {normalizar(c): c for c in df.columns}
    colunas = {}
    for papel, candidatas in CANDIDATAS.items():
        if forcadas.get(papel):
            colunas[papel] = forcadas[papel]
            continue
        achada = next((mapa_norm[c] for c in candidatas if c in mapa_norm), None)
        if achada is None and papel != "rotulo":
            raise SystemExit(
                f"Não encontrei a coluna de '{papel}'. Colunas do arquivo: {list(df.columns)}\n"
                f"Informe manualmente com --col_{papel} <NOME>."
            )
        colunas[papel] = achada
    return colunas


def carregar(caminho: str, forcadas: dict, arquivo_rotulos: str | None):
    # sep=None detecta automaticamente ',' ou ';' (o DataHub costuma exportar com ';').
    df = pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8-sig")
    col = detectar_colunas(df, forcadas)

    base = pd.DataFrame({
        "indicador": df[col["indicador"]].astype(str).str.strip(),
        "pais": df[col["pais"]].astype(str).str.strip(),
        "ano": pd.to_numeric(df[col["ano"]], errors="coerce"),
        "valor": pd.to_numeric(df[col["valor"]], errors="coerce"),
    }).dropna(subset=["ano", "valor"])

    rotulos = {}
    if col.get("rotulo"):
        rotulos = (df[[col["indicador"], col["rotulo"]]].drop_duplicates()
                   .set_index(col["indicador"])[col["rotulo"]].astype(str).to_dict())
    elif arquivo_rotulos:
        lab = pd.read_csv(arquivo_rotulos, sep=None, engine="python")
        norm = {normalizar(c): c for c in lab.columns}
        c_ind = next((norm[c] for c in CANDIDATAS["indicador"] if c in norm), lab.columns[0])
        c_rot = next((norm[c] for c in CANDIDATAS["rotulo"] if c in norm), lab.columns[-1])
        rotulos = lab.set_index(c_ind)[c_rot].astype(str).to_dict()
    return base, rotulos, col


def listar(base, rotulos, ano_min):
    recente = base[base["ano"] >= ano_min]
    cobertura = (recente.groupby("indicador")["pais"].nunique()
                 .sort_values(ascending=False).rename("paises_com_dado"))
    tabela = cobertura.to_frame()
    tabela["descricao"] = [rotulos.get(i, "") for i in tabela.index]
    pd.set_option("display.max_colwidth", 110)
    pd.set_option("display.width", 200)
    print(f"\nIndicadores com dados a partir de {ano_min} (ordenados por nº de países):\n")
    print(tabela.head(80).to_string())
    print(f"\nTotal de indicadores: {len(tabela)} | Total de países: {recente['pais'].nunique()}")


def classificar(valor, limites, nomes):
    for limite, nome in zip(limites, nomes):
        if valor < limite:
            return nome
    return nomes[-1]


def gerar(base, rotulos, args):
    recente = base[base["ano"] >= args.ano_min].sort_values("ano")
    # Valor mais recente de cada indicador em cada país -> evita misturar anos repetidos.
    ultimo = recente.groupby(["pais", "indicador"])["valor"].last().unstack()

    if args.alvo not in ultimo.columns:
        raise SystemExit(f"Indicador-alvo '{args.alvo}' não encontrado. Rode com --listar.")

    # Seleção das variáveis de entrada
    def suspeito(ind):
        texto = (ind + " " + rotulos.get(ind, "")).lower()
        return any(p in texto for p in PALAVRAS_VAZAMENTO)

    if args.entradas:
        entradas = args.entradas
    else:
        candidatos = [c for c in ultimo.columns if c != args.alvo and not suspeito(c)]
        cobertura = ultimo.loc[ultimo[args.alvo].notna(), candidatos].notna().mean()
        entradas = cobertura[cobertura >= args.cobertura_min].sort_values(
            ascending=False).head(args.max_entradas).index.tolist()
    if not entradas:
        raise SystemExit("Nenhuma variável de entrada passou no filtro de cobertura. "
                         "Reduza --cobertura_min.")

    dados = ultimo[[args.alvo] + entradas].dropna(subset=[args.alvo]).copy()
    dados["classe"] = dados[args.alvo].apply(
        lambda v: classificar(v, args.limites, args.nomes_classes))
    dados = dados.drop(columns=[args.alvo]).reset_index()

    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    dados.to_csv(args.saida, index=False)

    resumo = {
        "fonte": "UNESCO DataHub – R&D Other Policy Relevant Indicators (uis005)",
        "arquivo_origem": args.entrada,
        "ano_minimo": args.ano_min,
        "regra_ano": "valor mais recente de cada país a partir do ano mínimo",
        "indicador_alvo": args.alvo,
        "descricao_alvo": rotulos.get(args.alvo, ""),
        "limites_classes": args.limites,
        "nomes_classes": args.nomes_classes,
        "entradas": {e: rotulos.get(e, "") for e in entradas},
        "selecao_entradas": ("manual (--entradas): " + (args.justificativa or "sem justificativa informada"))
                            if args.entradas else
                            "automática, excluindo indicadores com termos relacionados ao alvo "
                            f"({', '.join(PALAVRAS_VAZAMENTO)})",
        "n_paises": int(len(dados)),
        "distribuicao_classes": dados["classe"].value_counts().to_dict(),
        "faltantes_por_entrada_%": (dados[entradas].isna().mean() * 100).round(1).to_dict(),
        "observacao": "Imputação (mediana) e padronização são ajustadas só no treino, em treinar.py.",
    }
    Path(args.resumo).write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dataset salvo em {args.saida}: {len(dados)} países, {len(entradas)} entradas.")
    print("Distribuição das classes:", resumo["distribuicao_classes"])
    if min(resumo["distribuicao_classes"].values()) < 5:
        print("ATENÇÃO: alguma classe tem menos de 5 países. Ajuste --limites para equilibrar.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entrada", required=True, help="CSV baixado do DataHub/BDDS")
    p.add_argument("--rotulos", help="CSV opcional com rótulos dos indicadores (ex.: OPRI_LABEL.csv)")
    p.add_argument("--listar", action="store_true", help="Só lista indicadores e cobertura")
    p.add_argument("--alvo", help="ID do indicador usado para criar as classes")
    p.add_argument("--entradas", nargs="*", help="IDs das entradas (opcional; senão, seleção automática)")
    p.add_argument("--justificativa", help="Texto registrado no resumo explicando a escolha manual das entradas")
    p.add_argument("--ano_min", type=int, default=2010)
    p.add_argument("--limites", nargs="+", type=float, default=[30.0, 45.0],
                   help="Cortes entre classes (ex.: 30 45 -> 3 classes)")
    p.add_argument("--nomes_classes", nargs="+", default=["baixa", "intermediaria", "paridade"])
    p.add_argument("--cobertura_min", type=float, default=0.6,
                   help="Fração mínima de países com dado para aceitar uma entrada")
    p.add_argument("--max_entradas", type=int, default=8)
    p.add_argument("--saida", default="data/processed/dataset.csv")
    p.add_argument("--resumo", default="data/processed/resumo_dados.json")
    for papel in CANDIDATAS:
        p.add_argument(f"--col_{papel}", help=f"Nome da coluna de {papel}, se a detecção falhar")
    args = p.parse_args()

    if len(args.nomes_classes) != len(args.limites) + 1:
        raise SystemExit("--nomes_classes precisa ter um nome a mais que --limites.")

    forcadas = {papel: getattr(args, f"col_{papel}") for papel in CANDIDATAS}
    base, rotulos, col = carregar(args.entrada, forcadas, args.rotulos)
    print(f"Colunas detectadas: {col}")

    if args.listar or not args.alvo:
        listar(base, rotulos, args.ano_min)
        if not args.alvo:
            print("\nEscolha o ID do indicador-alvo e rode de novo com --alvo <ID>.")
        return
    gerar(base, rotulos, args)


if __name__ == "__main__":
    main()
