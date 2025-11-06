import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.contrains import MODEL_NAME, MODEL_DIR, ID2LABEL


class LoadModel:
    def __init__(self, model_path: str = MODEL_DIR):
        self.model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path)
        self.id2label = ID2LABEL
        self.model.eval()

    def predict(self, text: str):
        words = text.split()
        tokenized_word = []
        word_map = []
        for i, word in enumerate(words):
            tokens = self.tokenizer.tokenize(word)
            tokenized_word.extend(tokens)
            word_map.extend([i] * len(tokens))
        encoded = self.tokenizer.encode_plus(
            tokenized_word, 
            is_split_into_words=False, 
            truncation=True, 
            padding="max_length", 
            max_length=64, 
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**encoded)
            preds = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        
        word_preds = []
        last_word = -1
        for idx, word_idx in enumerate(word_map):
            if word_idx != last_word:
                word_preds.append(preds[idx])
                last_word = word_idx        


        # return self.id2label[preds.item()]
        return self.id2label(word_preds)


text = "Con nhỏ đó ngu thật, nói chuyện mất dạy ghê"
model = LoadModel()
result = model.predict(text)

print(result)
