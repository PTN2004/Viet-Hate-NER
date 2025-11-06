import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from typing import List, Dict
from datasets import Dataset

def align_predictions(predictions, label_ids, id2label):

    preds = np.argmax(predictions, axis=2)

    preds_list = []
    labels_list = []

    for p, l in zip(preds, label_ids):
        preds_i = []
        labels_i = []
        for pred_id, label_id in zip(p, l):
            if label_id == -100:
                continue
            preds_i.append(id2label[int(pred_id)])
            labels_i.append(id2label[int(label_id)])
        preds_list.append(preds_i)
        labels_list.append(labels_i)

    return preds_list, labels_list

def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)
    true_labels, true_preds = [], []
    for pred, lab in zip(preds, labels):
        for p_, l_ in zip(pred, lab):
            if l_ != -100:
                true_labels.append(l_)
                true_preds.append(p_)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_preds, average="weighted"
    )
    accuracy = accuracy_score(true_labels, true_preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def read_file_dataset(file_path:str):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["Word", "Tag", "sentence_id"])

    df = df.groupby(df["sentence_id"]).agg({
        "Word" : list,
        "Tag" : list
    }).reset_index()

    dataset = Dataset.from_dict({
        "tokens": df["Word"].tolist(),
        "labels": df["Tag"].tolist()
    })

    return dataset
