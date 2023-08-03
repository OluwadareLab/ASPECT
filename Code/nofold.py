#########################################
### Importing the necessary libraries ###
#########################################

import torch
import numpy as np
import pandas as pd
import wandb
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback, AdamW
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils.model_utils import load_model, compute_metrics
from utils.data_utils import return_kmer, val_dataset_gene, HF_dataset, divide_dataset_into_folds
from utils.viz_utils import count_plot
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2500"


KMER = 6
NUM_FOLDS = 5  # Number of folds for stratified k-fold cross-validation
RANDOM_SEED = 42  # Random seed for reproducibility
SEQ_MAX_LEN = 512  # max len of BERT
EPOCHS = 5
BATCH_SIZE = 16

#############################################
## Initializing variables and reading data ##
#############################################

# Initialize wandb
wandb.init(project="DBertFolds", name=f"DNABERT_{KMER}_NOFOLDS")
wandb_config = {
	"model_path": f"DBertFolds_{KMER}",
}
wandb.config.update(wandb_config)
# breakpoint()

results_dir = os.path.join(".", "results", "ASP")
os.makedirs(results_dir, exist_ok=True)
file_count = len(os.listdir(results_dir))
results_dir = Path(f"./results")/ "ASP" / f"ASP_RUN-{file_count}"
# results_dir = Path(f"./results"/f"ASPrun_{runm}|{file_count}")

validation_results = []
sum_acc, sum_f1, test_folds = [], [], []                                  # List to store evaluation results for each fold
train_set = pd.read_csv("../tNt/train_est.csv")
# train_set = pd.read_csv("../dsc/train_data.csv")
# test_set = pd.read_csv("../dsc/test/test_data.csv")                                              # Load 20% subset of training data to split

# Split the data into small fraction, maintaining the same label distribution
# train_set, test_set = train_test_split(train_set, test_size=0.20, stratify=train_set["CLASS"])      # split train 80 and test 20
train_set, val_set = train_test_split(train_set, test_size=0.20, stratify=train_set["CLASS"])   # split train 70 and val 10

NUM_CLASSES = len(np.unique(train_set["CLASS"]))
model_config = {
    "model_path": f"zhihan1996/DNA_bert_{KMER}",
    "num_classes": NUM_CLASSES,
}
# model_config = {
#     "model_path": f"zhihan1996/DNABERT-2-117M",
#     "num_classes": NUM_CLASSES,
# }

f1_list, acc_list = [], []
# Create a WandbCallback object
class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"
    "A callback that logs the evaluation metrics to WandB"
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(metrics["eval_f1"])
        print("\n")
        f1_list.append(metrics["eval_f1"])
        acc_list.append(metrics["eval_accuracy"])
    def on_train_begin(self, args, state, control, **kwargs):
        print("*********************************Starting training*********************************")
        
class MyEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience=25, early_stopping_threshold=0.0):
        super().__init__(early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold)

model, tokenizer, device = load_model(model_config, return_model=True)     

train_dataset = val_dataset_gene(tokenizer, kmer_size=KMER, test_data=train_set)
val_dataset = val_dataset_gene(tokenizer, kmer_size=KMER, test_data=val_set)
# test_dataset = val_dataset_gene(tokenizer, kmer_size=KMER, test_data=test_set)

        ############################################
        ### Training and evaluating the model #####
        ############################################
        
eval_results = []
results_dir.mkdir(parents=True, exist_ok=True)
# Set up the Trainer
training_args = TrainingArguments(
    output_dir=results_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    learning_rate=0.001,
    weight_decay=0.01,
    logging_dir=results_dir / "logs",
    logging_steps=60,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=False,
	evaluation_strategy="epoch",
	save_strategy="epoch",
    fp16=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,
    report_to="wandb",  # enable logging to W&B
)

# training_args = training_args.set_optimizer(
#     name="adamw_torch",
#     learning_rate=1e-3,
#     weight_decay=0.01,
#     beta1=0.8,
#     beta2=0.999,
#     epsilon=1e-08,
# )

trainer = Trainer(
model=model,                         # the instantiated 🤗 Transformers model to be trained
args=training_args,                  # training arguments, defined above
train_dataset=train_dataset,         # training dataset
eval_dataset=val_dataset,            # validation dataset
compute_metrics=compute_metrics,     # computing metrics for evaluation in wandb
tokenizer=tokenizer,
callbacks=[MyCallback()]
)
#Train the model
trainer.train()

# save the model and tokenizer
model_path = results_dir
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(acc_list)
print(f1_list)
#Evauating on test data of fold
# res = trainer.evaluate(test_dataset)
# eval_results.append(res)     
# wandb.log({"test_acc": eval_results, "test_f1": eval_results})
# # average over the eval_accuracy and eval_f1 from the dic items in eval_results
avg_acc = np.mean(acc_list)
avg_f1 = np.mean(f1_list)
# print(f"\n##############################################\nAverage accuracy: {avg_acc}")
# print(f"Average F1: {avg_f1}")

wandb.log({"avg_acc": avg_acc, "avg_f1": avg_f1})
