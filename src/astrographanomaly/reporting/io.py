from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import networkx as nx

def write_outputs(out_dir: str, df_raw: pd.DataFrame, df_scored: pd.DataFrame, df_top: pd.DataFrame, G_full: nx.Graph, G_top: nx.Graph):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_raw.to_csv(out/"raw.csv", index=False)
    df_scored.to_csv(out/"scored.csv", index=False)
    df_top.to_csv(out/"top_anomalies.csv", index=False)

    nx.write_graphml(G_full, out/"graph_full.graphml")
    nx.write_graphml(G_top, out/"graph_topk.graphml")
