"""
決定木テキスト解析 - 手術フォルダ一括処理
解法B: 全行DataFrame + join アプローチ

想定ディレクトリ構造:
  base_dir/
    K0001_虫垂切除術/
      tree.txt
      feature_master.csv
    K0002_胆嚢摘出術/
      ...
"""

import os
import re
import pandas as pd

# ── 定数 ────────────────────────────────────────────────
INDENT_UNIT   = 4          # |--- の1階層あたりの文字数
MIN_WEIGHT    = 10         # 対象とする最小 weight（class:1 の件数）
TARGET_CLASS  = 1          # 対象クラス
COND_SEP      = " / "      # 出力CSV の条件区切り文字

# 正規表現
RE_BRANCH = re.compile(
    r"^(?:\|   )*\|--- (?P<feature>\S+) (?P<op><=|>) +(?P<threshold>[\d.]+)"
)
RE_LEAF = re.compile(
    r"^(?:\|   )*\|--- weights: \[(?P<w0>\d+), (?P<w1>\d+)\]\s+class: (?P<cls>\d+)"
)

def get_depth(line: str) -> int:
    """
    '|   |   |--- ...' のようなインデントから深度を返す。
    '|   ' の繰り返し数 = 深度。
    例: '|--- ...'        → 0
        '|   |--- ...'   → 1
        '|   |   |--- ...' → 2
    """
    count = 0
    i = 0
    while line[i:i+4] == "|   ":
        count += 1
        i += 4
    return count


# ── Step 1: フォルダ名パース ─────────────────────────────
def parse_folder_name(folder_name: str) -> dict:
    """
    'K0001_虫垂切除術' → {'surgery_code': 'K0001', 'surgery_name': '虫垂切除術'}
    アンダースコアが複数ある場合は最初の _ で分割
    """
    parts = folder_name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"フォルダ名の形式が不正です: {folder_name}")
    return {"surgery_code": parts[0], "surgery_name": parts[1]}


# ── Step 2: tree.txt → 全行 DataFrame ───────────────────
def load_tree_lines(tree_path: str) -> pd.DataFrame:
    """
    各行を (lineno, depth, type, feature, op, threshold, w0, w1, cls) に展開
    type: 'branch' | 'leaf' | 'other'
    """
    rows = []
    with open(tree_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f):
            line = raw.rstrip("\n")

            depth = get_depth(line)

            m_branch = RE_BRANCH.match(line)
            m_leaf   = RE_LEAF.match(line)

            if m_branch:
                rows.append({
                    "lineno":    lineno,
                    "depth":     depth,
                    "type":      "branch",
                    "feature":   m_branch.group("feature"),
                    "op":        m_branch.group("op"),
                    "threshold": m_branch.group("threshold"),
                    "w0": None, "w1": None, "cls": None,
                })
            elif m_leaf:
                rows.append({
                    "lineno":    lineno,
                    "depth":     depth,
                    "type":      "leaf",
                    "feature":   None,
                    "op":        None,
                    "threshold": None,
                    "w0":  int(m_leaf.group("w0")),
                    "w1":  int(m_leaf.group("w1")),
                    "cls": int(m_leaf.group("cls")),
                })
            # それ以外（空行など）は無視

    return pd.DataFrame(rows)


# ── Step 3: 葉 × 分岐 の cross join → パス構築 ──────────
def build_leaf_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    各葉ノードに対して、
    「自分より行番号が小さく、かつ深度が自分より浅い分岐行」
    を全て取得し、条件チェーンを組み立てる。

    返り値: leaf_id, lineno_leaf, depth_leaf, w0, w1, cls,
            conditions（リスト）
    """
    df_leaves  = df[df["type"] == "leaf"].copy().reset_index(drop=True)
    df_branches = df[df["type"] == "branch"].copy()

    results = []
    for leaf_id, leaf in df_leaves.iterrows():
        # 条件: branch の行番号 < 葉の行番号 かつ depth < 葉の depth
        path_rows = df_branches[
            (df_branches["lineno"] < leaf["lineno"]) &
            (df_branches["depth"]  < leaf["depth"])
        ].copy()

        # 深度ごとに最後に出現した分岐行だけを残す
        # （同じ深度で複数の分岐がある場合、直近のものが有効）
        path_rows = (
            path_rows
            .sort_values("lineno")
            .drop_duplicates(subset="depth", keep="last")
            .sort_values("depth")
        )

        # 条件文字列リストを生成（日本語化は後段で行う）
        conditions = [
            f"{r['feature']} {r['op']} {r['threshold']}"
            for _, r in path_rows.iterrows()
        ]

        results.append({
            "leaf_id":     leaf_id + 1,
            "lineno_leaf": leaf["lineno"],
            "w0":          leaf["w0"],
            "w1":          leaf["w1"],
            "cls":         leaf["cls"],
            "conditions":  conditions,
        })

    return pd.DataFrame(results)


# ── Step 4: フィルタ（class==1 かつ weight >= MIN_WEIGHT）──
def filter_leaves(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["cls"] == TARGET_CLASS) &
        (df["w1"]  >= MIN_WEIGHT)
    ].copy()


# ── Step 5: feature_master で条件を日本語化 ─────────────
def apply_feature_master(df: pd.DataFrame, feature_master_path: str) -> pd.DataFrame:
    """
    conditions リスト中のコードを feature_name に置換。
    マスタにないコードはそのまま残す。
    """
    fm = pd.read_csv(feature_master_path, dtype=str)
    code_to_name = dict(zip(fm["feature_code"], fm["feature_name"]))

    def translate(conditions: list) -> str:
        translated = []
        for cond in conditions:
            # 先頭のコード部分だけ置換（"ds01 <= 0.50" → "在院日数 <= 0.50"）
            for code, name in code_to_name.items():
                cond = re.sub(rf"^{re.escape(code)}\b", name, cond)
            translated.append(cond)
        return COND_SEP.join(translated)

    df["conditions_ja"] = df["conditions"].apply(translate)
    return df


# ── Step 6: 1フォルダ分の処理をまとめる ─────────────────
def process_folder(folder_path: str) -> pd.DataFrame | None:
    folder_name = os.path.basename(folder_path)

    try:
        meta = parse_folder_name(folder_name)
    except ValueError as e:
        print(f"[SKIP] {e}")
        return None

    tree_path   = os.path.join(folder_path, "tree.txt")
    fm_path     = os.path.join(folder_path, "feature_master.csv")

    if not os.path.exists(tree_path):
        print(f"[SKIP] tree.txt が見つかりません: {folder_path}")
        return None
    if not os.path.exists(fm_path):
        print(f"[SKIP] feature_master.csv が見つかりません: {folder_path}")
        return None

    # パイプライン
    df_lines  = load_tree_lines(tree_path)
    df_paths  = build_leaf_paths(df_lines)
    df_filtered = filter_leaves(df_paths)

    if df_filtered.empty:
        print(f"[INFO] 対象葉なし: {folder_name}")
        return None

    df_result = apply_feature_master(df_filtered, fm_path)

    # メタ情報を付与
    df_result["surgery_code"] = meta["surgery_code"]
    df_result["surgery_name"] = meta["surgery_name"]

    return df_result[[
        "surgery_code", "surgery_name",
        "leaf_id", "conditions_ja", "cls", "w1",
    ]].rename(columns={
        "conditions_ja": "conditions",
        "cls":           "predicted_class",
        "w1":            "weight",
    })


# ── Step 7: 全フォルダ一括処理 → CSV ────────────────────
def run(base_dir: str, output_path: str) -> pd.DataFrame:
    """
    base_dir 直下の全サブフォルダを処理して output_path に CSV 出力。
    """
    all_results = []

    folders = sorted([
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])

    print(f"対象フォルダ数: {len(folders)}")

    for folder_path in folders:
        folder_name = os.path.basename(folder_path)
        print(f"処理中: {folder_name}")
        result = process_folder(folder_path)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("出力対象の葉が1件もありませんでした。")
        return pd.DataFrame()

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n完了: {len(df_all)} 件 → {output_path}")
    return df_all


# ── エントリポイント ─────────────────────────────────────
if __name__ == "__main__":
    # Databricks では以下を実際のパスに変更
    BASE_DIR    = "/dbfs/mnt/surgery_trees"
    OUTPUT_PATH = "/dbfs/mnt/output/surgery_tree_rules.csv"

    df = run(BASE_DIR, OUTPUT_PATH)
    print(df.head(10).to_string(index=False))
