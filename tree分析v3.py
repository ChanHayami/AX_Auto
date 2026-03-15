import os
import re
import ast
import copy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =========================================================
# 設定
# =========================================================

BASE_DIR = "/dbfs/path/to/base_dir"   # 例: "/dbfs/FileStore/trees/base_dir"
OUTPUT_CSV_PATH = "/dbfs/path/to/output/result.csv"  # 例: "/dbfs/FileStore/trees/result.csv"

MIN_NON_ZERO_VALUE = 10.0
CSV_ENCODING = "utf-8-sig"


# =========================================================
# 1. フォルダ・ファイル探索系
# =========================================================

def list_disease_folders(base_dir: str) -> List[str]:
    """
    base_dir 配下の病名フォルダ一覧を返す。
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"base_dir が存在しません: {base_dir}")

    folder_paths = []
    for name in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path):
            folder_paths.append(full_path)
    return folder_paths


def parse_disease_folder_name(folder_name: str) -> Dict[str, str]:
    """
    フォルダ名 'K000001_うなうな' を分解する。
    """
    if "_" not in folder_name:
        raise ValueError(f"フォルダ名が想定形式ではありません: {folder_name}")

    disease_code, disease_name = folder_name.split("_", 1)
    if not disease_code or not disease_name:
        raise ValueError(f"フォルダ名の分解に失敗しました: {folder_name}")

    return {
        "disease_code": disease_code,
        "disease_name": disease_name,
    }


def build_target_file_paths(folder_path: str, disease_code: str) -> Dict[str, str]:
    """
    病名フォルダから対象ファイルパスを組み立てる。
    """
    return {
        "tree_file_path": os.path.join(folder_path, f"{disease_code}_tree.txt"),
        "feature_dict_path": os.path.join(folder_path, f"{disease_code}_features_dict.txt"),
    }


def validate_target_files(tree_file_path: str, feature_dict_path: str) -> Dict[str, Any]:
    """
    tree.txt / features_dict.txt の存在確認。
    """
    missing_files = []

    if not os.path.isfile(tree_file_path):
        missing_files.append(tree_file_path)
    if not os.path.isfile(feature_dict_path):
        missing_files.append(feature_dict_path)

    return {
        "is_valid": len(missing_files) == 0,
        "missing_files": missing_files,
    }


# =========================================================
# 2. 入力ファイル読み込み系
# =========================================================

def read_text_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    テキストファイルを文字列として読む。
    """
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def read_tree_lines(tree_file_path: str, encoding: str = "utf-8") -> List[str]:
    """
    tree.txt を行単位リストで読む。
    改行コードのみ除去し、空行は保持する。
    """
    with open(tree_file_path, "r", encoding=encoding) as f:
        return [line.rstrip("\n\r") for line in f]


def read_feature_dict(feature_dict_path: str, encoding: str = "utf-8") -> Dict[str, str]:
    """
    features_dict.txt を辞書として読む。
    失敗時は空辞書を返す。
    """
    text = read_text_file(feature_dict_path, encoding=encoding).strip()
    if not text:
        return {}

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            # key/value を文字列化しておく
            return {str(k): str(v) for k, v in parsed.items()}
        return {}
    except Exception as e:
        print(f"[WARN] feature_dict 読み込み失敗: {feature_dict_path} / {e}")
        return {}


# =========================================================
# 3. tree 行解析系
# =========================================================

def is_blank_line(line: str) -> bool:
    """
    空行または空白のみの行か判定。
    """
    return line.strip() == ""


def get_line_depth(line: str) -> int:
    """
    export_text() 風テキストの depth を取得する。

    例:
    |--- di0 <= 0.50              -> 0
    |   |--- co0 > 0.50           -> 1
    |   |   |--- weights: [...]   -> 2
    """
    # 行頭の "|   " の繰り返し数を数える
    count = 0
    pos = 0
    unit = "|   "
    while line.startswith(unit, pos):
        count += 1
        pos += len(unit)
    return count


def classify_tree_line(line: str) -> str:
    """
    行を分類する:
    - blank
    - leaf
    - branch
    - unknown
    """
    if is_blank_line(line):
        return "blank"

    stripped = line.strip()

    if "weights" in stripped:
        return "leaf"

    # "|---" の後に条件式らしきものがある場合を branch とみなす
    # 例: "|--- di0 <= 0.50"
    if "|---" in line:
        # weights でなければ branch 扱い
        return "branch"

    return "unknown"


def extract_condition_text(line: str) -> str:
    """
    行から '|--- ' 以降の条件/内容テキストを抜く。
    """
    idx = line.rfind("|---")
    if idx == -1:
        return line.strip()
    return line[idx + 4 :].strip()


def extract_branch_condition(line: str, depth: int) -> Dict[str, Any]:
    """
    分岐条件行から条件情報を抽出する。
    """
    raw_condition = extract_condition_text(line)

    # 典型形: di0 <= 0.50 / co0 > 1.5
    m = re.match(r"^([^\s]+)\s*(<=|>=|<|>|==|!=)\s*(.+)$", raw_condition)

    feature_code = None
    operator = None
    threshold = None

    if m:
        feature_code = m.group(1).strip()
        operator = m.group(2).strip()
        threshold = m.group(3).strip()

    return {
        "depth": depth,
        "raw_condition": raw_condition,
        "feature_code": feature_code,
        "operator": operator,
        "threshold": threshold,
    }


def _parse_weights_from_text(weights_raw: str) -> List[float]:
    """
    '[0.0, 12.0]' を [0.0, 12.0] にする。
    """
    try:
        parsed = ast.literal_eval(weights_raw)
        if isinstance(parsed, (list, tuple)):
            return [float(x) for x in parsed]
    except Exception:
        pass

    # フォールバック
    inner = weights_raw.strip().strip("[]").strip()
    if not inner:
        return []

    parts = [p.strip() for p in inner.split(",")]
    values = []
    for p in parts:
        if p == "":
            continue
        try:
            values.append(float(p))
        except ValueError:
            pass
    return values


def extract_leaf_info(line: str, depth: int, line_no: int) -> Dict[str, Any]:
    """
    葉ノード行から葉情報を抽出する。
    """
    text = extract_condition_text(line)

    # 例:
    # weights: [0.0, 12.0] class: 1
    # weights = [0.0, 12.0], class = 1
    weights_raw = ""
    class_raw: Optional[Any] = None

    m_weights = re.search(r"weights\s*[:=]\s*(\[[^\]]*\])", text)
    if m_weights:
        weights_raw = m_weights.group(1).strip()

    m_class = re.search(r"class\s*[:=]\s*([^\s,]+)", text)
    if m_class:
        class_token = m_class.group(1).strip()
        # 数値なら int 化を試みる
        if re.fullmatch(r"-?\d+", class_token):
            class_raw = int(class_token)
        else:
            class_raw = class_token

    weights = _parse_weights_from_text(weights_raw)

    return {
        "depth": depth,
        "line_no": line_no,
        "weights_raw": weights_raw,
        "weights": weights,
        "class_raw": class_raw,
        "raw_leaf_text": text,
    }


def trim_stack_to_depth(stack: List[Dict[str, Any]], depth: int) -> List[Dict[str, Any]]:
    """
    stack を指定 depth に対応する親条件数まで切り詰める。

    仕様:
    - depth 0 の branch/leaf の親条件数は 0
    - depth 1 の親条件数は 1
    - つまり stack 長は最大 depth に揃える
    """
    return stack[:depth]


def update_stack_with_branch(stack: List[Dict[str, Any]], branch_condition: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    分岐条件を stack に反映した新しい stack を返す。
    """
    depth = int(branch_condition["depth"])
    new_stack = trim_stack_to_depth(stack, depth)
    new_stack.append(branch_condition)
    return new_stack


def snapshot_path_conditions(stack: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    葉ノード用に現在の path 条件を deep copy して返す。
    """
    return copy.deepcopy(stack)


# =========================================================
# 4. 葉ノード判定・フィルタ系
# =========================================================

def analyze_weights(weights: List[float], eps: float = 1e-12) -> Dict[str, Any]:
    """
    weights を解析する。
    """
    non_zero_indices = []
    non_zero_values = []

    for i, x in enumerate(weights):
        if abs(x) > eps:
            non_zero_indices.append(i)
            non_zero_values.append(x)

    return {
        "non_zero_count": len(non_zero_indices),
        "non_zero_indices": non_zero_indices,
        "non_zero_values": non_zero_values,
    }


def is_target_leaf(weights: List[float], min_non_zero_value: float = 10.0) -> bool:
    """
    対象葉か判定する。
    条件:
    - 0以外が1つだけ
    - その値が10以上
    """
    summary = analyze_weights(weights)
    if summary["non_zero_count"] != 1:
        return False
    return summary["non_zero_values"][0] >= min_non_zero_value


def build_leaf_judgement_summary(weights: List[float], min_non_zero_value: float = 10.0) -> Dict[str, Any]:
    """
    weights 判定のサマリを返す。
    """
    summary = analyze_weights(weights)
    non_zero_index = summary["non_zero_indices"][0] if summary["non_zero_count"] == 1 else None
    non_zero_value = summary["non_zero_values"][0] if summary["non_zero_count"] == 1 else None

    return {
        "non_zero_count": summary["non_zero_count"],
        "non_zero_index": non_zero_index,
        "non_zero_value": non_zero_value,
        "is_target": is_target_leaf(weights, min_non_zero_value=min_non_zero_value),
    }


# =========================================================
# 5. 日本語化・整形系
# =========================================================

def translate_feature_code(feature_code: Optional[str], feature_dict: Dict[str, str]) -> Optional[str]:
    """
    feature code を日本語名に変換する。
    """
    if feature_code is None:
        return None
    return feature_dict.get(feature_code, feature_code)


def translate_branch_condition(branch_condition: Dict[str, Any], feature_dict: Dict[str, str]) -> Dict[str, Any]:
    """
    分岐条件1件を日本語化する。
    """
    translated_feature = translate_feature_code(branch_condition.get("feature_code"), feature_dict)

    if (
        branch_condition.get("feature_code") is not None
        and branch_condition.get("operator") is not None
        and branch_condition.get("threshold") is not None
    ):
        translated_condition = f"{translated_feature} {branch_condition['operator']} {branch_condition['threshold']}"
    else:
        translated_condition = branch_condition.get("raw_condition", "")

    return {
        "depth": branch_condition.get("depth"),
        "raw_condition": branch_condition.get("raw_condition"),
        "translated_condition": translated_condition,
        "feature_code": branch_condition.get("feature_code"),
        "translated_feature": translated_feature,
        "operator": branch_condition.get("operator"),
        "threshold": branch_condition.get("threshold"),
    }


def translate_path_conditions(path_conditions: List[Dict[str, Any]], feature_dict: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    パス全体を日本語化する。
    """
    return [translate_branch_condition(cond, feature_dict) for cond in path_conditions]


def convert_class_label(class_raw: Any, disease_name: str) -> str:
    """
    class の値を要件どおり変換する。
    - 1 -> disease_name
    - 0 -> "0"
    - その他 -> 文字列化
    """
    if class_raw == 1 or str(class_raw) == "1":
        return disease_name
    if class_raw == 0 or str(class_raw) == "0":
        return "0"
    return "" if class_raw is None else str(class_raw)


def join_conditions_text(
    conditions: List[Dict[str, Any]],
    field_name: str,
    separator: str = " AND "
) -> str:
    """
    条件一覧を1文字列に連結する。
    """
    parts = []
    for cond in conditions:
        value = cond.get(field_name)
        if value is not None and str(value).strip() != "":
            parts.append(str(value))
    return separator.join(parts)


def build_output_record(
    folder_info: Dict[str, str],
    tree_file_path: str,
    leaf_info: Dict[str, Any],
    path_conditions: List[Dict[str, Any]],
    translated_path_conditions: List[Dict[str, Any]],
    leaf_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    最終出力レコードを作る。
    """
    class_label = convert_class_label(leaf_info.get("class_raw"), folder_info["disease_name"])

    return {
        "disease_folder": folder_info["disease_folder"],
        "disease_code": folder_info["disease_code"],
        "disease_name": folder_info["disease_name"],
        "tree_file": os.path.basename(tree_file_path),
        "tree_file_path": tree_file_path,
        "leaf_line_no": leaf_info.get("line_no"),
        "class_raw": leaf_info.get("class_raw"),
        "class_label": class_label,
        "weights_raw": leaf_info.get("weights_raw"),
        "non_zero_count": leaf_summary.get("non_zero_count"),
        "non_zero_index": leaf_summary.get("non_zero_index"),
        "non_zero_value": leaf_summary.get("non_zero_value"),
        "conditions_raw": " || ".join([c.get("raw_condition", "") for c in path_conditions]),
        "conditions_ja": " || ".join([c.get("translated_condition", "") for c in translated_path_conditions]),
        "condition_path_text": join_conditions_text(path_conditions, "raw_condition", separator=" AND "),
        "condition_path_text_ja": join_conditions_text(translated_path_conditions, "translated_condition", separator=" AND "),
        "raw_leaf_text": leaf_info.get("raw_leaf_text"),
    }


# =========================================================
# 6. tree 全体処理系
# =========================================================

def parse_tree_file(tree_lines: List[str]) -> List[Dict[str, Any]]:
    """
    tree.txt を解析し、葉 + そこまでの path 条件を抽出する。
    ここではまだ対象葉フィルタはしない。
    """
    stack: List[Dict[str, Any]] = []
    parsed_leaf_paths: List[Dict[str, Any]] = []

    for zero_based_idx, line in enumerate(tree_lines):
        line_no = zero_based_idx + 1
        line_type = classify_tree_line(line)

        if line_type == "blank":
            continue

        depth = get_line_depth(line)

        if line_type == "branch":
            branch_condition = extract_branch_condition(line, depth)
            stack = update_stack_with_branch(stack, branch_condition)

        elif line_type == "leaf":
            # 葉に到達した時点の親条件を保持
            stack_for_leaf = trim_stack_to_depth(stack, depth)
            leaf_info = extract_leaf_info(line, depth, line_no)
            path_conditions = snapshot_path_conditions(stack_for_leaf)

            parsed_leaf_paths.append({
                "leaf_info": leaf_info,
                "path_conditions": path_conditions,
            })

        else:
            # unknown は無視。ただし必要ならログしてもよい
            continue

    return parsed_leaf_paths


def filter_target_leaf_paths(
    parsed_leaf_paths: List[Dict[str, Any]],
    min_non_zero_value: float = 10.0
) -> List[Dict[str, Any]]:
    """
    要件に合う葉だけ残す。
    """
    filtered = []
    for item in parsed_leaf_paths:
        leaf_info = item["leaf_info"]
        weights = leaf_info.get("weights", [])
        if is_target_leaf(weights, min_non_zero_value=min_non_zero_value):
            filtered.append(item)
    return filtered


def process_single_disease_folder(
    folder_path: str,
    min_non_zero_value: float = 10.0
) -> Dict[str, Any]:
    """
    1病名フォルダを処理する。
    """
    folder_name = os.path.basename(folder_path)

    try:
        parsed_folder = parse_disease_folder_name(folder_name)
        folder_info = {
            "disease_folder": folder_name,
            "disease_code": parsed_folder["disease_code"],
            "disease_name": parsed_folder["disease_name"],
        }

        paths = build_target_file_paths(folder_path, folder_info["disease_code"])
        validation = validate_target_files(paths["tree_file_path"], paths["feature_dict_path"])

        if not validation["is_valid"]:
            return {
                "records": [],
                "status": "skipped",
                "message": f"missing files: {validation['missing_files']}",
            }

        tree_lines = read_tree_lines(paths["tree_file_path"])
        feature_dict = read_feature_dict(paths["feature_dict_path"])

        parsed_leaf_paths = parse_tree_file(tree_lines)
        target_leaf_paths = filter_target_leaf_paths(
            parsed_leaf_paths,
            min_non_zero_value=min_non_zero_value
        )

        records = []
        for item in target_leaf_paths:
            leaf_info = item["leaf_info"]
            path_conditions = item["path_conditions"]
            translated_path_conditions = translate_path_conditions(path_conditions, feature_dict)
            leaf_summary = build_leaf_judgement_summary(
                leaf_info.get("weights", []),
                min_non_zero_value=min_non_zero_value
            )

            record = build_output_record(
                folder_info=folder_info,
                tree_file_path=paths["tree_file_path"],
                leaf_info=leaf_info,
                path_conditions=path_conditions,
                translated_path_conditions=translated_path_conditions,
                leaf_summary=leaf_summary,
            )
            records.append(record)

        return {
            "records": records,
            "status": "success",
            "message": "",
        }

    except Exception as e:
        return {
            "records": [],
            "status": "error",
            "message": str(e),
        }


# =========================================================
# 7. 全体実行・出力系
# =========================================================

def process_all_disease_folders(
    base_dir: str,
    min_non_zero_value: float = 10.0
) -> Dict[str, Any]:
    """
    base_dir 配下の全病名フォルダを処理する。
    """
    folder_paths = list_disease_folders(base_dir)

    all_records: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []

    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path)
        result = process_single_disease_folder(
            folder_path,
            min_non_zero_value=min_non_zero_value
        )

        records = result["records"]
        status = result["status"]
        message = result["message"]

        all_records.extend(records)
        logs.append({
            "folder": folder_name,
            "status": status,
            "message": message,
            "record_count": len(records),
        })

        print(f"[{status.upper()}] {folder_name} / record_count={len(records)} / {message}")

    return {
        "records": all_records,
        "logs": logs,
    }


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    records を DataFrame に変換する。
    """
    columns = [
        "disease_folder",
        "disease_code",
        "disease_name",
        "tree_file",
        "tree_file_path",
        "leaf_line_no",
        "class_raw",
        "class_label",
        "weights_raw",
        "non_zero_count",
        "non_zero_index",
        "non_zero_value",
        "conditions_raw",
        "conditions_ja",
        "condition_path_text",
        "condition_path_text_ja",
        "raw_leaf_text",
    ]

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=columns)

    # 列順を揃える。足りない列があれば作る。
    for col in columns:
        if col not in df.columns:
            df[col] = None

    return df[columns]


def save_dataframe_to_csv(
    df: pd.DataFrame,
    output_path: str,
    encoding: str = "utf-8-sig",
    index: bool = False
) -> None:
    """
    DataFrame を CSV 保存する。
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=index, encoding=encoding)


def run_pipeline(
    base_dir: str,
    output_csv_path: str,
    min_non_zero_value: float = 10.0
) -> Dict[str, Any]:
    """
    全体処理のエントリーポイント。
    """
    result = process_all_disease_folders(
        base_dir=base_dir,
        min_non_zero_value=min_non_zero_value
    )

    records = result["records"]
    logs = result["logs"]

    df = records_to_dataframe(records)
    save_dataframe_to_csv(df, output_csv_path, encoding=CSV_ENCODING, index=False)

    success_count = sum(1 for x in logs if x["status"] == "success")
    skip_count = sum(1 for x in logs if x["status"] == "skipped")
    error_count = sum(1 for x in logs if x["status"] == "error")

    summary = {
        "record_count": len(df),
        "folder_count": len(logs),
        "success_count": success_count,
        "skip_count": skip_count,
        "error_count": error_count,
        "output_csv_path": output_csv_path,
        "logs": logs,
    }

    print("=" * 80)
    print("処理完了")
    print(f"folder_count   : {summary['folder_count']}")
    print(f"success_count  : {summary['success_count']}")
    print(f"skip_count     : {summary['skip_count']}")
    print(f"error_count    : {summary['error_count']}")
    print(f"record_count   : {summary['record_count']}")
    print(f"output_csv_path: {summary['output_csv_path']}")
    print("=" * 80)

    return summary


# =========================================================
# 実行例
# =========================================================

if __name__ == "__main__":
    summary = run_pipeline(
        base_dir=BASE_DIR,
        output_csv_path=OUTPUT_CSV_PATH,
        min_non_zero_value=MIN_NON_ZERO_VALUE,
    )
