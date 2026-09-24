import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

st.set_page_config(page_title="Exploration du dataset", layout="wide")

# colonnes des emotions one-hot, fusionnees en une seule colonne "emotion"
EMOTIONS_ALL = [
    "anger", "boredom", "calm", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise",
]

# colonnes jugees utiles pour l'exploration (axe X et filtres)
USEFUL_COLUMNS = [
    "dataset", "emotion", "gender", "language", "speaker", "transcript", "age",
    "chunk", "emotion.confidence", "emotion.naturalness",
]


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, na_values=["None"])
    present = [e for e in EMOTIONS_ALL if e in df.columns]
    if present:
        df["emotion"] = df[present].fillna(0).idxmax(axis=1)
    return df


st.sidebar.header("Fichier")
csv_path = st.sidebar.text_input("Chemin du CSV", "data/spectrograms/labels.csv")

try:
    df = load_data(csv_path)
except FileNotFoundError:
    st.error(f"Fichier introuvable : {csv_path}")
    st.stop()

st.title("Exploration du dataset")
st.caption(f"{len(df)} lignes chargees depuis {csv_path}")

columns = [c for c in USEFUL_COLUMNS if c in df.columns]

MISSING = "(manquant)"

CONFIG_PATH = Path("data/split_config.yaml")
USABLE_DIR = Path("usable_data")
TRAIN_DIR = USABLE_DIR / "train"
TEST_DIR = USABLE_DIR / "test"


def load_config():
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def run_split(subset, target_column, train_pct, seed):
    """Mode proportionnel : X% des LIGNES de chaque valeur de target_column vont en train."""
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, group in subset.groupby(target_column, dropna=False):
        group = group.sample(frac=1, random_state=rng)
        n_train = int(round(len(group) * train_pct / 100))
        train_parts.append(group.iloc[:n_train])
        test_parts.append(group.iloc[n_train:])
    if train_parts:
        train_df = pd.concat(train_parts).reset_index(drop=True)
        test_df = pd.concat(test_parts).reset_index(drop=True)
    else:
        train_df = subset.iloc[0:0]
        test_df = subset.iloc[0:0]
    return train_df, test_df


def group_split(subset, target_column, train_pct, seed):
    """Mode par groupe : X% des VALEURS DISTINCTES de target_column vont entierement en train,
    quel que soit leur nombre de lignes (ex. 80% des speakers, chacun avec toutes ses lignes)."""
    key = subset[target_column].fillna("__MISSING__")
    groups = key.unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(groups)
    n_train_groups = int(round(len(shuffled) * train_pct / 100))
    train_groups = set(shuffled[:n_train_groups])
    train_mask = key.isin(train_groups)
    return subset[train_mask], subset[~train_mask]


def cascade_split(pool, stage_configs, seed):
    """Applique chaque etape (colonne cible + pourcentage + mode) successivement sur ce qui reste dans le train."""
    stages = []
    moved_parts = []
    train_pool = pool
    for cfg in stage_configs:
        split_fn = group_split if cfg["mode"] == "group" else run_split
        train_after, moved_to_test = split_fn(train_pool, cfg["column"], cfg["pct"], seed)
        mode_label = "par groupe" if cfg["mode"] == "group" else "par ligne"
        stages.append({
            "etape": f"{cfg['column']} @ {cfg['pct']}% ({mode_label})",
            "train avant": len(train_pool),
            "deplace vers test": len(moved_to_test),
            "train apres": len(train_after),
        })
        moved_parts.append(moved_to_test)
        train_pool = train_after
    return train_pool, moved_parts, stages

def copy_split(split_df, dest_dir, base_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n = len(split_df)
    progress = st.progress(0.0) if n else None
    for i, (_, row) in enumerate(split_df.iterrows()):
        src = base_dir / row["spectrogram"]
        dest_path = dest_dir / row["spectrogram"]
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists():
            st.warning(f"Collision de nom (deja copie ou meme nom qu'un autre dataset) : {dest_path}")
        try:
            shutil.copy2(src, dest_path)
        except FileNotFoundError:
            st.warning(f"Fichier introuvable, ignore : {src}")
            continue
        rows.append(row.to_dict())
        if progress is not None:
            progress.progress((i + 1) / n)
    return pd.DataFrame(rows)


st.sidebar.header("Filtres")
filtered = df.copy()
for col in columns:
    values = df[col]
    options = sorted(values.dropna().unique().tolist(), key=str)
    if values.isna().any():
        options.append(MISSING)
    selected = st.sidebar.multiselect(col, options, default=[])
    if selected:
        wanted = [v for v in selected if v != MISSING]
        mask = filtered[col].isin(wanted)
        if MISSING in selected:
            mask = mask | filtered[col].isna()
        filtered = filtered[mask]

st.sidebar.header("Graphique")
x_axis = st.sidebar.selectbox("Colonne (axe X)", columns)

st.subheader(f"Nombre de fichiers par {x_axis}")
counts = filtered[x_axis].value_counts(dropna=False).sort_index()
counts.index = counts.index.astype(str)
fig = px.bar(x=counts.index, y=counts.values, labels={"x": x_axis, "y": "nombre de fichiers"})
st.plotly_chart(fig, use_container_width=True)

st.caption(f"{len(filtered)} fichiers apres filtrage (sur {len(df)} au total)")

st.divider()
st.header("Decoupage train / test")

saved = load_config()


def rule_widgets(section_key, label, saved_rules):
    if not isinstance(saved_rules, list):
        saved_rules = []
    n = st.number_input(f"Nombre de regles - {label}", min_value=0, value=len(saved_rules), step=1, key=f"n_{section_key}")
    rules = []
    for i in range(int(n)):
        prev = saved_rules[i] if i < len(saved_rules) else {}
        c1, c2 = st.columns(2)
        with c1:
            default_col = prev.get("column") if prev.get("column") in columns else columns[0]
            col_choice = st.selectbox(
                f"Colonne ({label} #{i + 1})", columns,
                index=columns.index(default_col), key=f"{section_key}_col_{i}",
            )
        with c2:
            values = df[col_choice]
            options = sorted(values.dropna().unique().tolist(), key=str)
            if values.isna().any():
                options.append(MISSING)
            default_vals = [v for v in prev.get("values", []) if v in options]
            val_choice = st.multiselect(
                f"Valeurs ({label} #{i + 1})", options,
                default=default_vals, key=f"{section_key}_val_{i}",
            )
        rules.append({"column": col_choice, "values": val_choice})
    return rules


def apply_rules(data, rules):
    mask = pd.Series(False, index=data.index)
    for rule in rules:
        col, vals = rule["column"], rule["values"]
        if not vals:
            continue
        wanted = [v for v in vals if v != MISSING]
        m = data[col].isin(wanted)
        if MISSING in vals:
            m = m | data[col].isna()
        mask = mask | m
    return mask


st.subheader("Exclusions (retirees du train et du test)")
exclusion_rules = rule_widgets("excl", "exclusion", saved.get("exclusions", []))
subset = df[~apply_rules(df, exclusion_rules)]

st.subheader("Forcer entierement dans le test")
force_test_rules = rule_widgets("force", "forcage test", saved.get("force_test", []))
forced_mask = apply_rules(subset, force_test_rules)
forced_test_df = subset[forced_mask]
remaining = subset[~forced_mask]

st.subheader("Etapes (en cascade, chacune sur ce qui reste du train)")
saved_stages = saved.get("stages", [])
if not isinstance(saved_stages, list):
    saved_stages = []
n_stages = st.number_input("Nombre d'etapes", min_value=1, value=len(saved_stages) or 1, step=1)
stage_configs = []
for i in range(int(n_stages)):
    prev = saved_stages[i] if i < len(saved_stages) else {}
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        default_col = prev.get("column") if prev.get("column") in columns else columns[0]
        stage_col = st.selectbox(
            f"Etape {i + 1} : colonne cible", columns,
            index=columns.index(default_col), key=f"stage_col_{i}",
        )
    with c2:
        default_pct = prev.get("pct", 80)
        stage_pct = st.slider(
            f"Etape {i + 1} : % garde dans train", 50, 95,
            value=int(default_pct), key=f"stage_pct_{i}",
        )
    with c3:
        mode_options = {"Proportionnel (par ligne)": "row", "Groupe entier (par valeur)": "group"}
        default_mode_label = "Groupe entier (par valeur)" if prev.get("mode") == "group" else "Proportionnel (par ligne)"
        mode_label = st.radio(
            f"Etape {i + 1} : mode", list(mode_options),
            index=list(mode_options).index(default_mode_label), key=f"stage_mode_{i}",
        )
    stage_configs.append({"column": stage_col, "pct": stage_pct, "mode": mode_options[mode_label]})
seed = st.number_input("Seed (fixe, pour reproduire le meme split)", value=int(saved.get("seed", 42)), step=1)

final_train_df, moved_parts, stages = cascade_split(remaining, stage_configs, seed)
test_df = pd.concat([forced_test_df] + moved_parts, ignore_index=True)
train_df = final_train_df

st.table(pd.DataFrame(stages))

st.subheader("Apercu du split")
preview_x = st.selectbox("Colonne pour l'apercu", columns, index=columns.index(stage_configs[0]["column"]))

preview = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")], ignore_index=True)
counts = preview.groupby([preview_x, "split"], dropna=False).size().reset_index(name="count")
fig2 = px.bar(
    counts, x=preview_x, y="count", color="split",
    color_discrete_map={"train": "#1f77b4", "test": "#2ca02c"},
    labels={"count": "nombre de fichiers"},
)
st.plotly_chart(fig2, use_container_width=True)

st.caption(
    f"{len(df)} au total, {len(df) - len(subset)} exclus, "
    f"{len(forced_test_df)} forces dans le test, "
    f"{len(train_df)} train / {len(test_df)} test au final"
)

if st.button("Sauvegarder la config (sans lancer le split)"):
    save_config({
        "exclusions": exclusion_rules,
        "force_test": force_test_rules,
        "stages": stage_configs,
        "seed": int(seed),
    })
    st.success(f"Parametres sauvegardes dans {CONFIG_PATH}.")

if st.button("Lancer le split train/test"):
    base_dir = Path(csv_path).parent
    with st.spinner("Copie des fichiers train..."):
        train_out = copy_split(train_df, TRAIN_DIR, base_dir)
    with st.spinner("Copie des fichiers test..."):
        test_out = copy_split(test_df, TEST_DIR, base_dir)

    train_out.to_csv(TRAIN_DIR / "labels.csv", index=False, na_rep="None")
    test_out.to_csv(TEST_DIR / "labels.csv", index=False, na_rep="None")

    save_config({
        "exclusions": exclusion_rules,
        "force_test": force_test_rules,
        "stages": stage_configs,
        "seed": int(seed),
    })

    st.success(
        f"{len(train_out)} fichiers dans {TRAIN_DIR}, {len(test_out)} fichiers dans {TEST_DIR}. "
        f"Parametres sauvegardes dans {CONFIG_PATH}."
    )

st.sidebar.divider()
train_share = len(train_df) / len(subset) * 100 if len(subset) else 0
test_share = len(test_df) / len(subset) * 100 if len(subset) else 0
c1, c2 = st.sidebar.columns(2)
c1.metric("Part train (hors exclus)", f"{train_share:.1f}%")
c2.metric("Part test (hors exclus)", f"{test_share:.1f}%")