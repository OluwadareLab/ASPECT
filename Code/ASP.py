#########################################
### Importing the necessary libraries ###
#########################################

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, WandbCallback
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils.model_utils import load_model, compute_metrics
from utils.data_utils import return_kmer, val_dataset_gene, HF_dataset, divide_dataset_into_folds
from utils.viz_utils import count_plot
from pathlib import Path
import numpy as np
import pandas as pd
import wandb
import pdb
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


KMER = 6
NUM_FOLDS = 5  # Number of folds for stratified k-fold cross-validation
RANDOM_SEED = 42  # Random seed for reproducibility
SEQ_MAX_LEN = 512  # max len of BERT
EPOCHS = 10
BATCH_SIZE = 4 

#############################################
## Initializing variables and reading data ##
#############################################

# Initialize wandb
wandb.init(project="DBertFolds", name=f"DNABERT_{KMER}_F{NUM_FOLDS}")
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

f1_flog, acc_flog = {}, {}
validation_results = []
sum_acc, sum_f1, test_folds = [], [], []                                 # List to store evaluation results for each fold
tr_set = pd.read_csv("../tNt/train_data.csv")                           
test_set = pd.read_csv("../tNt/test/test_data.csv")

# # Split the data into small fraction, maintaining the same label distribution
# train_set, test_set = train_test_split(tr_set, test_size=0.5, stratify=tr_set["CLASS"], random_state=42)
# test_folds = divide_dataset_into_folds(tes_set, NUM_FOLDS)

# ds_kmer, ds_labels = [], []
# for seq, label in zip(train_set["SEQ"], train_set["CLASS"]):
#     kmer_seq = return_kmer(seq, K=KMER)          
#     ds_kmer.append(kmer_seq)                
#     ds_labels.append(label - 1)                  

ds_kmer, ds_labels = [], []
for seq, label in zip(tr_set["SEQ"], tr_set["CLASS"]):
    kmer_seq = return_kmer(seq, K=KMER)                         # Convert sequence to k-mer representation
    ds_kmer.append(kmer_seq)                                    # Append k-mer sequence to training data
    ds_labels.append(label - 1)                                 # Adjust label indexing

df_kmers = np.array(ds_kmer)
df_labels = np.array(ds_labels)

# labels = data["CLASS"].values                                 # Isolate the label columns in dataframe
                                                                # not used as top block does this and more.
                                                                # need to implement a more original method
NUM_CLASSES = len(np.unique(ds_labels))
model_config = {
    "model_path": f"zhihan1996/DNA_bert_{KMER}",
    "num_classes": NUM_CLASSES,
}
# model_config = {
#     "model_path": f"zhihan1996/DNABERT-2-117M",
#     "num_classes": NUM_CLASSES,
# }
model, tokenizer, device = load_model(model_config, return_model=True)
# breakpoint()
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle = True)       # Setting up skf fold count
count = 0
# for train_idx, eval_idx in skf.split(                           # Splitting data into k-folds
#     ds_kmer, ds_labels):                                        # to isolate the train and test pairs
#         count+=1
#         train_kmers = [ds_kmer[idx] for idx in train_idx]
#         train_labels = [ds_labels[idx] for idx in train_idx]
#         eval_kmers = [ds_kmer[idx] for idx in eval_idx]
#         eval_labels = [ds_labels[idx] for idx in eval_idx]

for train_idx, eval_idx in skf.split(                           # Splitting data into k-folds
    df_kmers, df_labels):                                        # to isolate the train and test pairs
        count+=1
        # print("Train:",train_idx,'Test:',eval_idx)
        train_kmers, eval_kmers = [df_kmers[train_idx], df_kmers[eval_idx]]
        train_labels, eval_labels = [df_labels[train_idx], df_labels[eval_idx]]
        
        count_plot(train_labels, f"Training Class Distribution Fold {count}", results_dir)
        count_plot(eval_labels, f"Evaluation Class Distribution Fold {count}", results_dir)
        
        # breakpoint()
        # train_kmers = train_kmers.tolist()
        # train_labels = train_labels.tolist()
        # eval_kmers = eval_kmers.tolist()
        # eval_labels = eval_labels.tolist()
                
        # Tokenize the two seperate data
        train_encodings = tokenizer.batch_encode_plus(
            train_kmers.tolist(),
            max_length=SEQ_MAX_LEN,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",  # return pytorch tensors
        )
  
        eval_encodings = tokenizer.batch_encode_plus(
            eval_kmers.tolist(),
            max_length=SEQ_MAX_LEN,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",  # return pytorch tensors
        )
        # breakpoint()
        train_dataset = HF_dataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels.tolist())       #worked
        val_dataset = HF_dataset(eval_encodings["input_ids"], eval_encodings["attention_mask"], eval_labels.tolist())           #worked
        
        # Create DataLoader for the training dataset
        # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
        # breakpoint()
        
        ############################################
        ### Training and evaluating the model #####
        ############################################
        
        eval_results = []
        results_dir.mkdir(parents=True, exist_ok=True)

        # Set up the Trainer
        training_args = TrainingArguments(
        output_dir=results_dir / f"fold_{count}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=results_dir / f"fold_{count}" / "logs",
        logging_steps=60,
        load_best_model_at_end=True,
	    evaluation_strategy="epoch",
	    save_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=16
        )
        
        trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # validation dataset
        compute_metrics=compute_metrics,     # computing metrics for evaluation in wandb
        tokenizer=tokenizer,
        )
        
        # breakpoint()
        # Train and evaluate
        result = trainer.train()
        # breakpoint()
        # save the model and tokenizer
        model_path = results_dir / f"modelfold{count}"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        
        # Get the validation metric from the validation result and store it in the validation_results list
        vmetr = result.metrics["eval_accuracy"]
        sum_acc.append(vmetr)
        vetr = result.metrics["eval_f1"]
        sum_f1.append(vmetr)

        wandb.log({
        f"Fold {count} Acc" : sum_acc[count-1],
        f"Fold {count} F1" : sum_f1[count-1]
        })
        
afold_acc = np.mean(sum_acc)
afold_f1 = np.mean(sum_f1)
wandb.log({"avg_acc": afold_acc, "avg_f1": afold_f1})
# average over the eval_accuracy and eval_f1 from the items in sum
# fold_acc = np.mean([res["eval_accuracy"] for res in eval_results])
# fold_f1 = np.mean([res["eval_f1"] for res in eval_results])
print(f"Average Fold Accuracy : {afold_acc}")
print(f"Average Fold F1 : {afold_f1}")

      

test_dataset = val_dataset_gene(tokenizer, kmer_size=KMER, test_data=test_set)         #test splitting 

res = trainer.evaluate(test_dataset)
eval_results.append(res)
wandb.log({"eval_acc": eval_results["eval_accuracy"], "avg_f1": eval_results["eval_f1"]})


wandb.finish()