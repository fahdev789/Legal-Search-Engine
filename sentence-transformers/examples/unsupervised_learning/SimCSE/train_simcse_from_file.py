"""
This file loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.

SimCSE will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.

Usage:
python train_simcse_from_file.py path/to/sentences.txt

"""

import gzip
import logging
import math
import sys
from datetime import datetime

import tqdm
from torch.utils.data import DataLoader

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

# Training parameters
model_name = "distilroberta-base"
train_batch_size = 128
max_seq_length = 32
num_epochs = 1

# Input file path (a text file, each line a sentence)
if len(sys.argv) < 2:
    print(f"Run this script with: python {sys.argv[0]} path/to/sentences.txt")
    exit()

filepath = sys.argv[1]

# Save path to store our model
output_name = ""
if len(sys.argv) >= 3:
    output_name = "-" + sys.argv[2].replace(" ", "_").replace("/", "_").replace("\\", "_")

model_output_path = "output/train_simcse{}-{}".format(output_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


# Use Hugging Face/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

################# Read the train corpus  #################
train_samples = []
with (
    gzip.open(filepath, "rt", encoding="utf8") if filepath.endswith(".gz") else open(filepath, encoding="utf8") as fIn
):
    for line in tqdm.tqdm(fIn, desc="Read file"):
        line = line.strip()
        if len(line) >= 10:
            train_samples.append(InputExample(texts=[line, line]))


logging.info(f"Train sentences: {len(train_samples)}")

# We train our model using the MultipleNegativesRankingLoss
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info(f"Warmup-steps: {warmup_steps}")

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": 5e-5},
    checkpoint_path=model_output_path,
    show_progress_bar=True,
    use_amp=False,  # Set to True, if your GPU supports FP16 cores
)
