# from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import ElectraTokenizerFast, ElectraForTokenClassification
import torch


def decode_entities(tokens, labels):
    """
    将B/I/E标签序列转换为实体列表
    Args:
        tokens: List[str] 输入token列表（长度N）
        labels: List[str] 对应标签列表（长度N）
    Returns:
        List[Dict] 实体列表，每个元素包含start、end、text、type信息
    """
    entities = []
    current_entity = None

    for idx, (token, label) in enumerate(zip(tokens, labels)):
        # 处理B标签：开始新实体
        if label.startswith('B-'):
            if current_entity is not None:
                # 保存未闭合的实体（可能单个B标签）
                entities.append(current_entity)
            entity_type = label.split('-')[1]
            current_entity = {
                'start': idx,
                'end': idx + 1,
                'text': [token],
                'type': entity_type
            }

        # 处理I/E标签：延续当前实体
        elif label.startswith('I-') or label.startswith('E-'):
            if current_entity is None:
                continue  # 跳过非法I/E开头

            # 检查类型是否一致
            expected_type = current_entity['type']
            current_type = label.split('-')[1]

            if current_type == expected_type:
                current_entity['text'].append(token)
                current_entity['end'] = idx + 1

                # E标签表示实体结束
                if label.startswith('E-'):
                    entities.append(current_entity)
                    current_entity = None
            else:
                # 类型不匹配时结束当前实体
                entities.append(current_entity)
                current_entity = None

        # 处理O标签：结束当前实体
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    # 处理最后一个未闭合的实体
    if current_entity is not None:
        entities.append(current_entity)

    # 合并token并格式化输出
    formatted = []
    for entity in entities:
        start_pos = entity['start']
        end_pos = entity['end']
        merged_text = merge_tokens(tokens[start_pos:end_pos])
        formatted.append({
            'text': merged_text,
            'type': entity['type'],
            'start': start_pos,
            'end': end_pos - 1  # 转换为闭区间索引
        })

    return formatted


def merge_tokens(token_list):
    """
    合并子词token（处理BERT等模型的分词结果）
    示例：["New", "York"] -> "New York"
           ["un", "##able"] -> "unable"
    """
    merged = []
    for token in token_list:
        if token.startswith("##"):
            merged[-1] += token[2:]
        else:
            merged.append(token)
    return "".join(merged)


class NerPredict:

    def __init__(self):
        MODEL_NAME = "ner_model_electra_best"
        self.tokenizer = ElectraTokenizerFast.from_pretrained(MODEL_NAME)
        self.model = ElectraForTokenClassification.from_pretrained(MODEL_NAME)
        self.MAX_LEN = 128
        self.id2label = {
            0: "B-disease",
            1: "E-disease",
            2: "I-disease",
            3: "O"
        }

    def predict(self,text):
        self.model.eval()
        tokens = list(text)
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
        # 对齐标签
        word_ids = encoding.word_ids()
        previous_word_idx = None
        labels = []
        for i, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue
            labels.append(self.id2label[predictions[i]])
            previous_word_idx = word_idx
        return decode_entities(tokens,labels)


if __name__ == "__main__":
    ner_predict = NerPredict()
    sentences = ["瘦脸针、水光针和玻尿酸详解！", "半月板钙化的病因有哪些？"]
    for s in sentences:
        print(ner_predict.predict(s))


