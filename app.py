import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

st.set_page_config(page_title="Dataset exploration", layout="wide")

# one-hot emotion columns, merged into a single "emotion" column
EMOTIONS_ALL = [
    "anger", "boredom", "calm", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise",
]

# columns deemed useful for exploration (X axis and filters)
USEFUL_COLUMNS = [
    "dataset", "emotion", "gender", "language", "speaker", "transcript", "age",
    "chunk", "emotion.confidence", "emotion.naturalness", "duration"
]


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, na_values=["None"])
    present = [e for e in EMOTIONS_ALL if e in df.columns]
    if present:
        df["emotion"] = df[present].fillna(0).idxmax(axis=1)
    return df


st.sidebar.header("File")
csv_path = st.sidebar.text_input("CSV path", "data/spectrograms/labels.csv")

try:
    df = load_data(csv_path)
except FileNotFoundError:
    st.error(f"File not found: {csv_path}")
    st.stop()

st.title("Dataset exploration")
st.caption(f"{len(df)} rows loaded from {csv_path}")

columns = [c for c in USEFUL_COLUMNS if c in df.columns]

MISSING = "(missing)"

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
    """Proportional mode: X% of the ROWS of each target_column value go to train."""
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
    """Group mode: X% of the DISTINCT VALUES of target_column go entirely to train,
    regardless of their row count (e.g. 80% of speakers, each with all their rows)."""
    key = subset[target_column].fillna("__MISSING__")
    groups = key.unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(groups)
    n_train_groups = int(round(len(shuffled) * train_pct / 100))
    train_groups = set(shuffled[:n_train_groups])
    train_mask = key.isin(train_groups)
    return subset[train_mask], subset[~train_mask]


def cascade_split(pool, stage_configs, seed):
    """Applies each stage (target column + percentage + mode) successively on what remains in train."""
    stages = []
    moved_parts = []
    train_pool = pool
    for cfg in stage_configs:
        split_fn = group_split if cfg["mode"] == "group" else run_split
        train_after, moved_to_test = split_fn(train_pool, cfg["column"], cfg["pct"], seed)
        mode_label = "by group" if cfg["mode"] == "group" else "by row"
        stages.append({
            "stage": f"{cfg['column']} @ {cfg['pct']}% ({mode_label})",
            "train before": len(train_pool),
            "moved to test": len(moved_to_test),
            "train after": len(train_after),
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
            st.warning(f"Name collision (already copied or same name as another dataset): {dest_path}")
        try:
            shutil.copy2(src, dest_path)
        except FileNotFoundError:
            st.warning(f"File not found, skipped: {src}")
            continue
        rows.append(row.to_dict())
        if progress is not None:
            progress.progress((i + 1) / n)
    return pd.DataFrame(rows)


st.sidebar.header("Filters")
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

st.sidebar.header("Chart")
x_axis = st.sidebar.selectbox("Column (X axis)", columns)
use_histogram = st.sidebar.toggle(
    "Distribution view (histogram, for continuous columns like emotion.naturalness, duration, age)"
)

if use_histogram:
    nbins = st.sidebar.slider("Number of bins", 5, 200, value=30)
    st.subheader(f"Distribution of {x_axis}")
    fig = px.histogram(filtered, x=x_axis, nbins=nbins, labels={x_axis: x_axis})
else:
    st.subheader(f"Number of files per {x_axis}")
    counts = filtered[x_axis].value_counts(dropna=False).sort_index()
    counts.index = counts.index.astype(str)
    fig = px.bar(x=counts.index, y=counts.values, labels={"x": x_axis, "y": "number of files"})
st.plotly_chart(fig, use_container_width=True)

st.caption(f"{len(filtered)} files after filtering (out of {len(df)} total)")

st.divider()
st.header("Train / test split")

saved = load_config()


def rule_widgets(section_key, label, saved_rules):
    if not isinstance(saved_rules, list):
        saved_rules = []
    n = st.number_input(f"Number of rules ({label})", min_value=0, value=len(saved_rules), step=1, key=f"n_{section_key}")
    rules = []
    for i in range(int(n)):
        prev = saved_rules[i] if i < len(saved_rules) else {}
        c1, c2 = st.columns(2)
        with c1:
            default_col = prev.get("column") if prev.get("column") in columns else columns[0]
            col_choice = st.selectbox(
                f"Column ({label} #{i + 1})", columns,
                index=columns.index(default_col), key=f"{section_key}_col_{i}",
            )
        with c2:
            values = df[col_choice]
            options = sorted(values.dropna().unique().tolist(), key=str)
            if values.isna().any():
                options.append(MISSING)
            default_vals = [v for v in prev.get("values", []) if v in options]
            val_choice = st.multiselect(
                f"Values ({label} #{i + 1})", options,
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


st.subheader("Exclusions (removed from train and test)")
exclusion_rules = rule_widgets("excl", "exclusion", saved.get("exclusions", []))
subset = df[~apply_rules(df, exclusion_rules)]

st.subheader("Force entirely into test")
force_test_rules = rule_widgets("force", "test forcing", saved.get("force_test", []))
forced_mask = apply_rules(subset, force_test_rules)
forced_test_df = subset[forced_mask]
remaining = subset[~forced_mask]

st.subheader("Stages (cascading, each on what remains of train)")
saved_stages = saved.get("stages", [])
if not isinstance(saved_stages, list):
    saved_stages = []
n_stages = st.number_input("Number of stages", min_value=1, value=len(saved_stages) or 1, step=1)
stage_configs = []
for i in range(int(n_stages)):
    prev = saved_stages[i] if i < len(saved_stages) else {}
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        default_col = prev.get("column") if prev.get("column") in columns else columns[0]
        stage_col = st.selectbox(
            f"Stage {i + 1}: target column", columns,
            index=columns.index(default_col), key=f"stage_col_{i}",
        )
    with c2:
        default_pct = prev.get("pct", 80)
        stage_pct = st.slider(
            f"Stage {i + 1}: % kept in train", 50, 95,
            value=int(default_pct), key=f"stage_pct_{i}",
        )
    with c3:
        mode_options = {"Proportional (by row)": "row", "Whole group (by value)": "group"}
        default_mode_label = "Whole group (by value)" if prev.get("mode") == "group" else "Proportional (by row)"
        mode_label = st.radio(
            f"Stage {i + 1}: mode", list(mode_options),
            index=list(mode_options).index(default_mode_label), key=f"stage_mode_{i}",
        )
    stage_configs.append({"column": stage_col, "pct": stage_pct, "mode": mode_options[mode_label]})
seed = st.number_input("Seed (fixed, to reproduce the same split)", value=int(saved.get("seed", 42)), step=1)

final_train_df, moved_parts, stages = cascade_split(remaining, stage_configs, seed)
test_df = pd.concat([forced_test_df] + moved_parts, ignore_index=True)
train_df = final_train_df

st.table(pd.DataFrame(stages))

st.subheader("Split preview")
preview_x = st.selectbox("Column for preview", columns, index=columns.index(stage_configs[0]["column"]))

preview = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")], ignore_index=True)
counts = preview.groupby([preview_x, "split"], dropna=False).size().reset_index(name="count")
fig2 = px.bar(
    counts, x=preview_x, y="count", color="split",
    color_discrete_map={"train": "#1f77b4", "test": "#2ca02c"},
    labels={"count": "number of files"},
)
st.plotly_chart(fig2, use_container_width=True)

st.caption(
    f"{len(df)} total, {len(df) - len(subset)} excluded, "
    f"{len(forced_test_df)} forced into test, "
    f"{len(train_df)} train / {len(test_df)} test in the end"
)

if st.button("Save config (without running the split)"):
    save_config({
        "exclusions": exclusion_rules,
        "force_test": force_test_rules,
        "stages": stage_configs,
        "seed": int(seed),
    })
    st.success(f"Settings saved to {CONFIG_PATH}.")

if st.button("Run train/test split"):
    base_dir = Path(csv_path).parent
    with st.spinner("Copying train files..."):
        train_out = copy_split(train_df, TRAIN_DIR, base_dir)
    with st.spinner("Copying test files..."):
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
        f"{len(train_out)} files in {TRAIN_DIR}, {len(test_out)} files in {TEST_DIR}. "
        f"Settings saved to {CONFIG_PATH}."
    )

st.sidebar.divider()
train_share = len(train_df) / len(subset) * 100 if len(subset) else 0
test_share = len(test_df) / len(subset) * 100 if len(subset) else 0
c1, c2 = st.sidebar.columns(2)
c1.metric("Train share (excluding excluded)", f"{train_share:.1f}%")
c2.metric("Test share (excluding excluded)", f"{test_share:.1f}%")