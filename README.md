PhoBERT NER training template
=============================

Files:
- pho_ner_dataset.py     : NERDataset class (word-to-subword alignment, -100 mask)
- utils.py               : helper metrics and alignment
- train_phobert.py       : training script using Hugging Face Trainer
- requirements.txt       : python dependencies

Quick start:
1. Create a virtualenv and install dependencies:
   pip install -r requirements.txt

2. Edit `train_phobert.py`:
   - replace demo `texts` and `labels` with your dataset loader
   - set OUTPUT_DIR, NUM_EPOCHS, BATCH_SIZE, MAX_LEN as needed

3. Run training:
   python train_phobert.py

Notes:
- This template uses `use_fast=False` tokenizer for PhoBERT (vinai/phobert-base).
  If you prefer fast tokenizer, change `use_fast=True` and adapt dataset to use tokenizer.word_ids().
- Labels use -100 for ignored positions (subwords, special tokens, pads).
- To change optimizer, either:
    * pass (optimizer, scheduler) to Trainer via the `optimizers` argument, or
    * subclass Trainer and override `create_optimizer`.
