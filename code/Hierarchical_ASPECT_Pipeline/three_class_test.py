import os
import sys

os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["USE_FLASH_ATTENTION_2"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "TRUE"


import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


class FlashAttnStub:
    """Stub class to replace flash_attn functions"""
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Flash Attention is disabled. Use standard attention instead.")
    def __getattr__(self, name):
        return FlashAttnStub()

_stub_module = type(sys)('flash_attn')
_stub_module.flash_attn_func = FlashAttnStub()
_stub_module.flash_attn_qkvpacked_func = FlashAttnStub()
_stub_module.flash_attn_varlen_func = FlashAttnStub()
_stub_triton = type(sys)('triton')
_stub_triton.jit = FlashAttnStub()
_stub_triton.cdiv = lambda x, y: (x + y - 1) // y  
sys.modules['flash_attn'] = _stub_module
sys.modules['flash_attn.flash_attn_triton'] = _stub_module
sys.modules['flash_attn.flash_attn_interface'] = _stub_module

try:
    import triton
    pass
except ImportError:
    sys.modules['triton'] = _stub_triton

class FlashAttnTritonImportHook:
    """Import hook to replace flash_attn_triton with a stub"""
    def find_spec(self, name, path, target=None):
        if name.endswith('flash_attn_triton') or 'flash_attn_triton' in name:
            from importlib.util import spec_from_loader, ModuleSpec
            from importlib.machinery import ModuleLoader
            
            class StubLoader:
                def create_module(self, spec):
                    stub = type(sys)('flash_attn_triton_stub')
                    stub._flash_attn_forward = FlashAttnStub()
                    return stub
                def exec_module(self, module):
                    pass
            
            return spec_from_loader(name, StubLoader())
        return None

# Install the import hook
sys.meta_path.insert(0, FlashAttnTritonImportHook())

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Run three-class classification test (cassette vs alt_three vs alt_five)')
parser.add_argument('--dataset', type=str, default='dataset1', 
                    choices=['dataset1', 'dataset2', 'dataset3'],
                    help='Evaluation dataset to use (default: dataset1)')
parser.add_argument('--input-csv', type=str, default=None,
                    help='Custom input CSV path (overrides --dataset if provided)')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Output directory (default: ./test_result/result_three_class_{dataset})')
args = parser.parse_args()

dataset_names = {
    "dataset1": "dataset1_constitutive_cassette_dominant.csv",
    "dataset2": "dataset2_alt_three_alt_five_dominant.csv",
    "dataset3": "dataset3_constitutive_alt_three_dominant.csv"
}

if args.input_csv:
    test_csv_path = args.input_csv
    EVALUATION_DATASET = os.path.basename(args.input_csv).replace('.csv', '')
else:
    EVALUATION_DATASET = args.dataset
    test_csv_path = f'../evaluation_datasets/{dataset_names[EVALUATION_DATASET]}'

model_path = "../three_class_model_training/result_11/DB2_balanced_three_class_from_multiclass/best_model"

if args.output_dir:
    output_dir = args.output_dir
else:
    output_dir = f"./test_result/result_three_class_{EVALUATION_DATASET}"

prediction_csv_path = os.path.join(output_dir, "predictions_with_probabilities.csv")
os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
test_df = pd.read_csv(test_csv_path)
has_labels = "label" in test_df.columns

def preprocess_test_data(dataframe, tokenizer, max_length=1024, has_labels=True):
    texts = dataframe["sequence"].tolist()
    
    if has_labels:
        labels = dataframe["label"].tolist()
        # Map string labels to numeric values (alphabetical order: alt_five=0, alt_three=1, cassette=2)
        label_mapping = {
            "alt_five": 0,
            "alt_three": 1,
            "cassette": 2
        }
        numeric_labels = []
        for label in labels:
            if label in label_mapping:
                numeric_labels.append(label_mapping[label])
            else:
                numeric_labels.append(-1)
        labels_tensor = torch.tensor(numeric_labels)
    else:
        labels_tensor = torch.zeros(len(texts), dtype=torch.long)

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings, labels_tensor

test_encodings, test_labels = preprocess_test_data(test_df, tokenizer, has_labels=has_labels)
model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
if hasattr(model.config, 'use_flash_attention_2'):
    model.config.use_flash_attention_2 = False
if hasattr(model.config, 'use_flash_attention'):
    model.config.use_flash_attention = False
if hasattr(model.config, '_flash_attn_2_enabled'):
    model.config._flash_attn_2_enabled = False

def patch_model_attention(model):
    """Patch all attention layers to disable Flash Attention"""
    for name, module in model.named_modules():
        if hasattr(module, '_flash_attn_enabled'):
            module._flash_attn_enabled = False
        if hasattr(module, 'use_flash_attention_2'):
            module.use_flash_attention_2 = False
        if hasattr(module, 'use_flash_attention'):
            module.use_flash_attention = False
        if hasattr(module, 'self') and hasattr(module.self, '_flash_attn_enabled'):
            module.self._flash_attn_enabled = False
        if hasattr(module, 'self') and hasattr(module.self, 'use_flash_attention_2'):
            module.self.use_flash_attention_2 = False

patch_model_attention(model)

model.to(device)
model.eval()

test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"], test_labels
)
test_loader = DataLoader(test_dataset, batch_size=16)

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

class_names = ["alt_five", "alt_three", "cassette"]

if has_labels:
    valid_indices = [i for i, label in enumerate(all_labels) if label != -1]
    
    if len(valid_indices) > 0:
        valid_preds = [all_preds[i] for i in valid_indices]
        valid_labels = [all_labels[i] for i in valid_indices]
        valid_probs = [all_probs[i] for i in valid_indices]
        
        report = classification_report(
            valid_labels, valid_preds, target_names=class_names, output_dict=True
        )
        conf_matrix = confusion_matrix(valid_labels, valid_preds)

        report_df = pd.DataFrame(report).transpose()
        report_csv_path = os.path.join(output_dir, "classification_report.csv")
        report_df.to_csv(report_csv_path, index=True)

        plt.figure(figsize=(8, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Three Class (cassette vs alt_three vs alt_five)")
        conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(conf_matrix_path)
        plt.close()

        roc_auc_scores = {}
        for i, class_name in enumerate(class_names):
            try:
                roc_auc = roc_auc_score((np.array(valid_labels) == i).astype(int), [p[i] for p in valid_probs])
                roc_auc_scores[f"roc_auc_{class_name}"] = roc_auc
            except ValueError:
                roc_auc_scores[f"roc_auc_{class_name}"] = "N/A"
        roc_data = {
            "roc_auc_scores": roc_auc_scores,
            "confusion_matrix": conf_matrix.tolist(),
        }
        roc_json_path = os.path.join(output_dir, "roc_auc.json")
        with open(roc_json_path, "w") as f:
            json.dump(roc_data, f)

label_mapping = {0: "alt_five", 1: "alt_three", 2: "cassette"}
predicted_events = [label_mapping[p] for p in all_preds]

if has_labels:
    true_events = []
    for label in all_labels:
        if label == -1:
            true_events.append("N/A")
        else:
            true_events.append(label_mapping[label])
else:
    true_events = ["N/A"] * len(predicted_events)

predicted_probabilities = [prob[pred] for prob, pred in zip(all_probs, all_preds)]
prob_alt_five = [prob[0] for prob in all_probs]
prob_alt_three = [prob[1] for prob in all_probs]
prob_cassette = [prob[2] for prob in all_probs]

output_df = pd.DataFrame({
    "sequence": test_df["sequence"],
    "true_event_class": true_events,
    "predicted_event_class": predicted_events,
    "predicted_probability": predicted_probabilities,
    "prob_cassette": prob_cassette,
    "prob_alt_three": prob_alt_three,
    "prob_alt_five": prob_alt_five
})
output_df.to_csv(prediction_csv_path, index=False)

all_events = set(output_df['true_event_class'].unique()) | set(output_df['predicted_event_class'].unique())
all_events = sorted([e for e in all_events if pd.notna(e)])

event_counts = []
for event in all_events:
    actual_count = len(output_df[output_df['true_event_class'] == event])
    predicted_count = len(output_df[output_df['predicted_event_class'] == event])
    true_predicted_count = len(output_df[(output_df['true_event_class'] == event) & (output_df['predicted_event_class'] == event)])
    event_counts.append({
        'event': event,
        'actual_count': actual_count,
        'predicted_count': predicted_count,
        'true_predicted_count': true_predicted_count
    })

counts_df = pd.DataFrame(event_counts)
counts_csv_path = os.path.join(output_dir, "event_counts.csv")
counts_df.to_csv(counts_csv_path, index=False)

