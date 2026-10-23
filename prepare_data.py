"""
prepare_data.py

把 Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset 轉成 BERT 多元分類的訓練資料。

流程：
    data.csv（欄位：text, emotion）
        ↓
    清理空白、移除空字串與重複句子
        ↓
    文字 label → 整數 id（建立 label2id / id2label）
        ↓
    stratified 切分 train / valid / test（80% / 10% / 10%）
        ↓
    data/bert/train.jsonl、valid.jsonl、test.jsonl
    data/bert/label_map.json

資料來源（擇一）：
    1. 手動從 Hugging Face「Files and versions」下載 data.csv，放到 data/raw/data.csv
    2. 沒有本機檔案時，自動用 pandas 從 hf:// 路徑下載（需要安裝 huggingface_hub）
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================================================
# 設定
# =============================================================================

LOCAL_CSV = Path("data/raw/data.csv")
HF_CSV = "hf://datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset/data.csv"

OUTPUT_DIR = Path("data/bert")

TEXT_COLUMN = "text"
LABEL_COLUMN = "emotion"

VALID_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42


# =============================================================================
# 輔助函式
# =============================================================================

def load_raw_dataframe() -> pd.DataFrame:
    """優先讀本機 data/raw/data.csv，沒有的話從 Hugging Face 下載並存一份到本機。"""
    if LOCAL_CSV.is_file():
        print(f"讀取本機檔案：{LOCAL_CSV}")
        return pd.read_csv(LOCAL_CSV)

    print(f"找不到 {LOCAL_CSV}，改從 Hugging Face 下載：{HF_CSV}")
    df = pd.read_csv(HF_CSV)
    LOCAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOCAL_CSV, index=False)
    print(f"已另存一份到：{LOCAL_CSV}")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """只保留 text / emotion 兩欄，去除空白、空值與重複資料。"""
    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        raise KeyError(f"CSV 缺少欄位：{missing}，目前欄位：{list(df.columns)}")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

    before = len(df)
    df = df[(df[TEXT_COLUMN] != "") & (df[TEXT_COLUMN].str.lower() != "nan")]
    df = df[(df[LABEL_COLUMN] != "") & (df[LABEL_COLUMN].str.lower() != "nan")]

    # 同一句話若出現多次，只保留第一筆，避免同句同時出現在 train 與 test 造成資料洩漏。
    df = df.drop_duplicates(subset=[TEXT_COLUMN], keep="first")
    print(f"清理前 {before} 筆 → 清理後 {len(df)} 筆")

    return df.reset_index(drop=True)


def build_label_map(labels):
    """
    文字 label → 整數 id。

    資料集卡片上的類別名稱與實際資料不完全一致（例如「驚訝」/「驚奇」），
    所以這裡不寫死類別，而是直接從資料中出現的 label 建立對照表。
    排序後再編號，確保每次執行得到相同的 id。
    """
    names = sorted(set(labels))
    label2id = {name: index for index, name in enumerate(names)}
    id2label = {index: name for name, index in label2id.items()}
    return label2id, id2label


def save_jsonl(df: pd.DataFrame, path: Path):
    with path.open("w", encoding="utf-8") as file:
        for text, label in zip(df[TEXT_COLUMN], df["label"]):
            row = {"text": text, "label": int(label)}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_distribution(name, df, id2label):
    counter = Counter(df["label"])
    print(f"\n[{name}] 共 {len(df)} 筆")
    for label_id in sorted(counter):
        print(f"  {label_id} {id2label[label_id]:<6} {counter[label_id]:>5}")


# =============================================================================
# 主程式
# =============================================================================

def main():
    df = clean_dataframe(load_raw_dataframe())

    label2id, id2label = build_label_map(df[LABEL_COLUMN])
    df["label"] = df[LABEL_COLUMN].map(label2id)

    # 先切出 test，再從剩下的資料切出 valid，兩次都用 stratify 保持各類比例。
    train_valid_df, test_df = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=df["label"],
        random_state=SEED,
    )
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=VALID_RATIO / (1 - TEST_RATIO),
        stratify=train_valid_df["label"],
        random_state=SEED,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_df, OUTPUT_DIR / "train.jsonl")
    save_jsonl(valid_df, OUTPUT_DIR / "valid.jsonl")
    save_jsonl(test_df, OUTPUT_DIR / "test.jsonl")

    with (OUTPUT_DIR / "label_map.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nLabel 對照表：")
    for name, index in label2id.items():
        print(f"  {name} → {index}")

    print_distribution("train", train_df, id2label)
    print_distribution("valid", valid_df, id2label)
    print_distribution("test", test_df, id2label)

    print(f"\n輸出完成：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
