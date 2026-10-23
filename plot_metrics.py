"""
plot_metrics.py

讀取 train_bert.py 產生的檔案，畫出：
    images/loss_curve.png        Training / Validation Loss
    images/metric_curve.png      Validation Accuracy / Macro F1
    images/confusion_matrix.png  Test set 混淆矩陣

中文字型：
    依序尋找 fonts/NotoSansTC-Regular.otf、Windows 的微軟正黑體、macOS 的蘋方，
    找不到時圖上的中文類別會改用數字 id 顯示。
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

STATE_FILE = Path("output/bert_emotion_clf/trainer_state.json")
CONFUSION_FILE = Path("results/confusion_matrix.json")
IMAGE_DIR = Path("images")

FONT_CANDIDATES = [
    Path("fonts/NotoSansTC-Regular.otf"),
    Path("C:/Windows/Fonts/msjh.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
]


def setup_chinese_font():
    for path in FONT_CANDIDATES:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            print(f"使用中文字型：{path}")
            return True
    print("找不到中文字型，混淆矩陣將以數字 id 顯示類別。")
    return False


def points(logs, key):
    return [(item["step"], item[key]) for item in logs if "step" in item and key in item]


def plot_lines(series, title, ylabel, output_path):
    plt.figure(figsize=(10, 6))
    for label, data, style in series:
        if data:
            x, y = zip(*data)
            plt.plot(x, y, style, label=label, linewidth=1.8)
    plt.xlabel("Optimizer Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已輸出：{output_path.resolve()}")


def plot_confusion_matrix(has_font):
    if not CONFUSION_FILE.is_file():
        print(f"找不到 {CONFUSION_FILE}，略過混淆矩陣。")
        return

    data = json.loads(CONFUSION_FILE.read_text(encoding="utf-8"))
    labels = data["labels"] if has_font else [str(i) for i in range(len(data["labels"]))]
    matrix = data["matrix"]
    size = len(labels)

    fig, ax = plt.subplots(figsize=(1.1 * size + 3, 1.1 * size + 2))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(size), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(size), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test set)")

    peak = max(max(row) for row in matrix) or 1
    for i in range(size):
        for j in range(size):
            value = matrix[i][j]
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > peak * 0.6 else "black", fontsize=10)

    fig.tight_layout()
    output_path = IMAGE_DIR / "confusion_matrix.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已輸出：{output_path.resolve()}")
    if not has_font:
        for index, name in enumerate(data["labels"]):
            print(f"  {index} = {name}")


def main():
    if not STATE_FILE.is_file():
        raise FileNotFoundError(f"找不到 {STATE_FILE}，請先執行 python train_bert.py")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    has_font = setup_chinese_font()

    logs = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("log_history", [])

    plot_lines(
        [
            ("Training Loss", points(logs, "loss"), "-"),
            ("Validation Loss", points(logs, "eval_loss"), "o-"),
        ],
        "BERT Training and Validation Loss",
        "Loss",
        IMAGE_DIR / "loss_curve.png",
    )
    plot_lines(
        [
            ("Validation Accuracy", points(logs, "eval_accuracy"), "o-"),
            ("Validation Macro F1", points(logs, "eval_macro_f1"), "s-"),
        ],
        "BERT Validation Metrics",
        "Score",
        IMAGE_DIR / "metric_curve.png",
    )
    plot_confusion_matrix(has_font)


if __name__ == "__main__":
    main()
