"""
infer_bert.py

載入微調後的 Best Model（output/bert_emotion_clf/），預測作業指定的句子。

輸出格式（score 取到小數點後第 4 位）：
    我每天都能跟她一起上學，我好開心！ => 開心語調 (0.9876)

也可以自己指定句子：
    python infer_bert.py "今天考試考砸了" "哇！你怎麼會在這裡？"
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_DIR = Path("output/bert_emotion_clf")
RESULT_FILE = Path("results/predictions.txt")

# 作業指定要預測的句子
TEXTS = [
    "我每天都能跟她一起上學，我好開心！",
    "最好的朋友要離開臺灣了，以後可能不容易再見面...",
    "我覺得我快不行了...",
    "剛剛收到研究所錄取的通知書！",
    "今年的冬天好像比較晚來。",
]

# 額外顯示前幾名的類別（方便觀察模型在哪些類別間猶豫）
TOP_K = 3


def main():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"找不到模型：{MODEL_DIR.resolve()}，請先執行 python train_bert.py")

    texts = sys.argv[1:] or TEXTS

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = 0 if torch.cuda.is_available() else -1
    print("Inference device：" + (torch.cuda.get_device_name(0) if device == 0 else "CPU"))

    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=TOP_K,  # 回傳前 K 名；第一名就是預測結果
    )
    results = classifier(texts, truncation=True, max_length=128)

    lines = []
    print("=" * 72)
    for text, ranking in zip(texts, results):
        best = ranking[0]
        line = f"{text} => {best['label']} ({best['score']:.4f})"
        lines.append(line)
        print(line)
        others = "、".join(f"{r['label']} {r['score']:.4f}" for r in ranking[1:])
        print(f"    其他候選：{others}")
    print("=" * 72)

    # 只有預測作業指定句子時才寫檔，方便直接貼到 README
    if not sys.argv[1:]:
        RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        RESULT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"預測結果已存到：{RESULT_FILE.resolve()}")


if __name__ == "__main__":
    main()
