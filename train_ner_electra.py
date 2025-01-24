# 安装依赖
# pip install transformers torch datasets seqeval

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import  ElectraForTokenClassification,ElectraTokenizerFast
import evaluate

import json

with open("datasets/train_data.json", encoding='utf-8') as f:
    train_data = json.load(f)


with open("datasets/test_data.json", encoding='utf-8') as f:
    test_data = json.load(f)

# 参数设置
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 1e-5
MODEL_NAME ='hfl/chinese-electra-180g-small-ex-discriminator'

# 创建标签映射
label_list = sorted(list(set([label for sample in train_data for label in sample["label"]])))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}


# 自定义数据集类
class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["input"]
        labels = item["label"]

        # Tokenize 输入并对齐标签
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels_ids = []
        word_ids = encoding.word_ids()
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels_ids.append(-100)
            elif word_idx != previous_word_idx:
                labels_ids.append(label2id[labels[word_idx]])
            else:
                labels_ids.append(-100)  # 子词标记设为 -100（在 CrossEntropy 中忽略）
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels_ids, dtype=torch.long)
        }

from transformers import get_linear_schedule_with_warmup

# 初始化模型和组件
tokenizer = ElectraTokenizerFast.from_pretrained(MODEL_NAME)
model = ElectraForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

train_dataset = NERDataset(train_data, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

valid_dataset = NERDataset(test_data, tokenizer, MAX_LEN)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

t_total = len(train_loader) * EPOCHS

warmup_steps = int(0.1 * t_total)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
)



seqeval = evaluate.load("seqeval")

def evaluate(model, dataloader):
    model.eval()
    true_predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # 转换为预测结果
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
            p= [[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)]
            true_predictions.extend(p)
            t=[  [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)]
            true_labels.extend(t )

    print(len(true_predictions), len(true_labels))
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    print(results)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 新增参数
OUTPUT_DIR = "./ner_model_electra"  # 模型保存路径
SAVE_BEST_MODEL = True      # 是否保存最佳模型
best_f1 = 0                 # 用于跟踪最佳分数


# 训练循环
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    count=0
    print(len(train_loader),BATCH_SIZE)
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        count+=1
        if count % 100 == 0:
            print(f"Epoch Count {count} - Average loss: {total_loss /count:.4f}")

    print(f"Epoch {epoch + 1} - Average loss: {total_loss / len(train_loader):.4f}")
    # 每个 epoch 结束后保存
    epoch_dir = f"{OUTPUT_DIR}_epoch{epoch + 1}"
    model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)
    print(f"模型已保存至 {epoch_dir}")

    # 如果有验证集，可以添加最佳模型保存逻辑
    if SAVE_BEST_MODEL:
        # 假设使用验证集进行评估
        eval_results = evaluate(model, valid_loader)  # 需要实现 valid_loader
        current_f1 = eval_results["f1"]

        if current_f1 > best_f1:
            best_f1 = current_f1
            model.save_pretrained(f"{OUTPUT_DIR}_best")
            tokenizer.save_pretrained(f"{OUTPUT_DIR}_best")
            print(f"★ 最佳模型已更新 (F1: {best_f1:.4f})")

# 最终模型保存（训练结束后）
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"最终模型已保存至 {OUTPUT_DIR}")

