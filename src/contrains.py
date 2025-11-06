LABELS_LIST = ["O", "B-LOC", "I-LOC"]
LABEL2ID = {l: i for i, l in enumerate(LABELS_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


MODEL_NAME = "vinai/phobert-large"
MODEL_DIR = "./models/train_models"
