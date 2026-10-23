"""
train_bert.py

使用 google-bert/bert-base-chinese 微調「中文句子多元情緒分類」模型。

跟課堂二元分類範例的差異：
    1. NUM_LABELS 不寫死，從 data/bert/label_map.json 讀出類別數與名稱。
    2. 評估指標從 binary F1 改成 accuracy、macro F1、weighted F1。
    3. 資料各類別數量不平均，預設使用 class weight 的 Cross-Entropy Loss，
       讓少數類別（例如厭惡、恐懼）也被重視。
    4. 以 validation macro F1 選出 Best Model，存到 output/bert_emotion_clf/，
       checkpoint-* 資料夾則放在 checkpoints/（不上傳 GitHub）。
    5. 訓練完在 test set 上評估，輸出 classification report 與 confusion matrix。

流程：
    data/bert/*.jsonl
        ↓
    Tokenizer（truncation，不預先 padding）
        ↓
    DataCollatorWithPadding（dynamic padding）
        ↓
    BERT + Sequence Classification Head（num_labels = 類別數）
        ↓
    Trainer（evaluation / early stopping / best model）
        ↓
    output/bert_emotion_clf/
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# =============================================================================
# 設定
# =============================================================================

MODEL_NAME = "google-bert/bert-base-chinese"

DATA_DIR = Path("data/bert")
LABEL_MAP_FILE = DATA_DIR / "label_map.json"

# checkpoint-* 放這裡（.gitignore 已排除）
CHECKPOINT_DIR = Path("checkpoints")
# Best Model 放這裡（要上傳 GitHub）
OUTPUT_DIR = Path("output/bert_emotion_clf")
# 測試集評估結果
RESULT_DIR = Path("results")

# 對話句子都很短，128 個 token 綽綽有餘，比 512 省很多 GPU 記憶體與時間。
MAX_SEQ_LENGTH = 128

SEED = 42
NUM_TRAIN_EPOCHS = 6
PER_DEVICE_TRAIN_BATCH_SIZE = 32
PER_DEVICE_EVAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"

EVAL_STEPS = 50
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 2
EARLY_STOPPING_PATIENCE = 4
LOGGING_STEPS = 10

# 以哪個指標挑 Best Model
METRIC_FOR_BEST_MODEL = "macro_f1"

# 是否使用 class weight 處理類別不平衡
USE_CLASS_WEIGHTS = True


# =============================================================================
# 輔助函式
# =============================================================================

def print_env_info():
    print("=" * 72)
    print(f"PyTorch：{torch.__version__}")
    print(f"Transformers：{transformers.__version__}")
    print(f"CUDA available：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(f"BF16 supported：{torch.cuda.is_bf16_supported()}")
    print("=" * 72)


def pick_precision():
    """RTX 30/40/50 系列（Ampere 之後）用 BF16；較舊的 GPU 用 FP16；沒有 GPU 就用 FP32。"""
    if not torch.cuda.is_available():
        return False, False
    if torch.cuda.is_bf16_supported():
        return False, True
    return True, False


def load_label_map():
    if not LABEL_MAP_FILE.is_file():
        raise FileNotFoundError(f"找不到 {LABEL_MAP_FILE}，請先執行 python prepare_data.py")
    with LABEL_MAP_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    label2id = {name: int(index) for name, index in data["label2id"].items()}
    id2label = {int(index): name for index, name in data["id2label"].items()}
    return label2id, id2label


def compute_metrics(eval_pred):
    """logits → argmax → 計算多元分類指標。"""
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = np.argmax(predictions, axis=-1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def compute_class_weights(labels, num_labels):
    """
    weight_c = N / (K × n_c)

    N：總筆數、K：類別數、n_c：第 c 類的筆數。
    數量越少的類別權重越大；全部類別一樣多時，權重都等於 1。
    """
    counts = np.bincount(np.asarray(labels), minlength=num_labels).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    """把 Trainer 預設的 Cross-Entropy 換成加上 class weight 的版本。"""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def evaluate_on_test(trainer, test_dataset, id2label):
    """在 test set 上評估，輸出指標、分類報告、混淆矩陣。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    output = trainer.predict(test_dataset)
    preds = np.argmax(output.predictions, axis=-1)
    labels = output.label_ids
    names = [id2label[i] for i in range(len(id2label))]

    metrics = {
        "test_accuracy": float(accuracy_score(labels, preds)),
        "test_macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "test_size": int(len(labels)),
    }
    report = classification_report(
        labels, preds, labels=list(range(len(names))), target_names=names, digits=4, zero_division=0
    )
    matrix = confusion_matrix(labels, preds, labels=list(range(len(names))))

    with (RESULT_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    (RESULT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    with (RESULT_DIR / "confusion_matrix.json").open("w", encoding="utf-8") as file:
        json.dump({"labels": names, "matrix": matrix.tolist()}, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("Test set 評估結果")
    print("=" * 72)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print()
    print(report)


# =============================================================================
# 主程式
# =============================================================================

def main():
    print_env_info()
    transformers.set_seed(SEED)

    label2id, id2label = load_label_map()
    num_labels = len(label2id)
    print(f"類別數：{num_labels}")
    print(f"id2label：{id2label}")

    data_files = {
        "train": str(DATA_DIR / "train.jsonl"),
        "validation": str(DATA_DIR / "valid.jsonl"),
        "test": str(DATA_DIR / "test.jsonl"),
    }
    for split, path in data_files.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"找不到 {split} 資料：{path}，請先執行 python prepare_data.py")

    dataset = load_dataset("json", data_files=data_files)
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(list(dataset["train"]["label"]), num_labels)
        print("Class weights：")
        for index, weight in enumerate(class_weights.tolist()):
            print(f"  {index} {id2label[index]:<6} {weight:.3f}")

    # warmup 步數 = 總 optimizer steps × WARMUP_RATIO（直接算成整數，避免不同版本對 warmup_ratio 的支援差異）
    steps_per_epoch = math.ceil(
        len(tokenized["train"]) / (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    )
    warmup_steps = int(steps_per_epoch * NUM_TRAIN_EPOCHS * WARMUP_RATIO)
    print(f"每個 epoch {steps_per_epoch} steps，warmup {warmup_steps} steps")

    use_fp16, use_bf16 = pick_precision()
    print(f"fp16={use_fp16}, bf16={use_bf16}")

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        class_weights=class_weights,
    )

    trainer.train()

    print(f"最佳 checkpoint：{trainer.state.best_model_checkpoint}")
    print(f"最佳 eval_{METRIC_FOR_BEST_MODEL}：{trainer.state.best_metric}")

    # load_best_model_at_end=True → 目前 trainer.model 就是 Best Model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    # trainer_state.json 存到 output/ 一份，plot_metrics.py 會讀它畫圖
    trainer.state.save_to_json(str(OUTPUT_DIR / "trainer_state.json"))

    evaluate_on_test(trainer, tokenized["test"], id2label)

    print(f"Best Model 已儲存：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
