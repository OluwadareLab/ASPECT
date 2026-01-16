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

# Block flash_attn import by creating a stub module in sys.modules
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
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Run binary classification test (cassette vs alt3, cassette vs alt5, alt3 vs alt5)')
parser.add_argument('--input', type=str,
                    default='./test_result/result_three_class_dataset1/predictions_with_probabilities.csv',
                    help='Path to three-class predictions CSV (default: ./test_result/result_three_class_dataset1/predictions_with_probabilities.csv)')
parser.add_argument('--output-dir', type=str,
                    default='./test_result/result_binary_class_dataset1',
                    help='Output directory (default: ./test_result/result_binary_class_dataset1)')
args = parser.parse_args()

three_class_results_path = args.input
output_dir = args.output_dir
prediction_csv_path = os.path.join(output_dir, "predictions_with_probabilities.csv")
os.makedirs(output_dir, exist_ok=True)

binary_model_overrides = {
    tuple(sorted(["cassette", "alt_three"])): "../binary_model_training/result_8/DB2_cassette_vs_alt_three/best_model",
    tuple(sorted(["cassette", "alt_five"])): "../binary_model_training/result_8/DB2_cassette_vs_alt_five/best_model",
    tuple(sorted(["alt_three", "alt_five"])): "../binary_model_training/result_13/DB2_alt_three_vs_alt_five/best_model",
}


ENSEMBLE_STRATEGY = 'optimized_hybrid'
THREE_CLASS_CONFIDENCE_THRESHOLD = 0.60
CASSETTE_ENSEMBLE_WEIGHT_3CLASS = 0.30


binary_dir = "../result_binary_class"
all_binary_models = {}

if os.path.exists(binary_dir):
    for model_dir_name in os.listdir(binary_dir):
        if model_dir_name.startswith("DB2_") and os.path.isdir(os.path.join(binary_dir, model_dir_name)):
            best_model_path = os.path.join(binary_dir, model_dir_name, "best_model")
            if os.path.exists(best_model_path):
                name = model_dir_name.replace("DB2_", "")
                if "_vs_" in name:
                    base_name = name.replace("_retry", "")
                    classes = base_name.split("_vs_")
                    if len(classes) == 2:
                        classes_tuple = tuple(sorted(classes))
                        is_retry = "_retry" in name
                        
                        if classes_tuple not in all_binary_models:
                            all_binary_models[classes_tuple] = []
                        all_binary_models[classes_tuple].append({
                            "model_path": best_model_path,
                            "classes": classes,
                            "name": name,
                            "is_retry": is_retry
                        })


binary_models = {}
for classes_tuple, models in all_binary_models.items():
    if all(cls in ["cassette", "alt_three", "alt_five"] for cls in classes_tuple):
        models.sort(key=lambda x: (not x["is_retry"], x["name"]))
        preferred_model = models[0]
        binary_models[classes_tuple] = preferred_model["model_path"]

for classes_tuple, model_path in binary_model_overrides.items():
    if os.path.exists(model_path):
        binary_models[classes_tuple] = model_path

three_class_df = pd.read_csv(three_class_results_path)
sequences_by_model = defaultdict(list)

for idx, row in three_class_df.iterrows():
    sequence = row['sequence']
    true_event = row['true_event_class']
    prob_cassette = row['prob_cassette']
    prob_alt_three = row['prob_alt_three']
    prob_alt_five = row['prob_alt_five']
    
    class_probs = [
        ("cassette", prob_cassette),
        ("alt_three", prob_alt_three),
        ("alt_five", prob_alt_five)
    ]
    
    class_probs.sort(key=lambda x: x[1], reverse=True)
    if len(class_probs) >= 2:
        top_2_classes = tuple(sorted([cls for cls, prob in class_probs[:2]]))
        
        if top_2_classes == ('alt_five', 'alt_three'):
            if ENSEMBLE_STRATEGY == 'optimized_hybrid':
                sequences_by_model[top_2_classes].append({
                    'sequence': sequence,
                    'true_event': true_event,
                    'three_class_probs': {
                        'cassette': prob_cassette,
                        'alt_three': prob_alt_three,
                        'alt_five': prob_alt_five
                    },
                    'original_index': idx
                })
            else:
                sequences_by_model[('USE_3CLASS_DIRECT',)].append({
                    'sequence': sequence,
                    'true_event': true_event,
                    'three_class_probs': {
                        'cassette': prob_cassette,
                        'alt_three': prob_alt_three,
                        'alt_five': prob_alt_five
                    },
                    'original_index': idx,
                    'three_class_prediction': class_probs[0][0]
                })
        else:
            sequences_by_model[top_2_classes].append({
                'sequence': sequence,
                'true_event': true_event,
                'three_class_probs': {
                    'cassette': prob_cassette,
                    'alt_three': prob_alt_three,
                    'alt_five': prob_alt_five
                },
                'original_index': idx
                })

all_results = []

if ('USE_3CLASS_DIRECT',) in sequences_by_model:
    for seq_info in sequences_by_model[('USE_3CLASS_DIRECT',)]:
        all_3class_probs = [
            ('cassette', seq_info['three_class_probs']['cassette']),
            ('alt_three', seq_info['three_class_probs']['alt_three']),
            ('alt_five', seq_info['three_class_probs']['alt_five'])
        ]
        all_3class_probs.sort(key=lambda x: x[1], reverse=True)
        top_3class_pred = all_3class_probs[0][0]
        top_3class_prob = all_3class_probs[0][1]
        
        result = {
            'sequence': seq_info['sequence'],
            'true_event_class': seq_info['true_event'],
            'two_class_model': 'USE_3CLASS_DIRECT',
            'predicted_event_class': top_3class_pred,
            'predicted_probability': top_3class_prob,
            'prediction_source': 'three_class_direct',
            'true_event_in_model': seq_info['true_event'] if seq_info['true_event'] != "N/A" else "N/A",
            'original_index': seq_info['original_index']
        }
        
        result['prob_alt_three'] = seq_info['three_class_probs']['alt_three']
        result['prob_alt_five'] = seq_info['three_class_probs']['alt_five']
        result['three_class_prob_cassette'] = seq_info['three_class_probs']['cassette']
        result['three_class_prob_alt_three'] = seq_info['three_class_probs']['alt_three']
        result['three_class_prob_alt_five'] = seq_info['three_class_probs']['alt_five']
        
        all_results.append(result)

for classes, model_path in binary_models.items():
    if classes not in sequences_by_model or not os.path.exists(model_path):
        continue
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    
    # Disable Flash Attention in model config
    if hasattr(model.config, 'use_flash_attention_2'):
        model.config.use_flash_attention_2 = False
    if hasattr(model.config, 'use_flash_attention'):
        model.config.use_flash_attention = False
    if hasattr(model.config, '_flash_attn_2_enabled'):
        model.config._flash_attn_2_enabled = False
    
    def patch_model_attention(model):
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
    
    sequences = [seq_info['sequence'] for seq_info in sequences_by_model[classes]]
    true_labels = [seq_info['true_event'] for seq_info in sequences_by_model[classes]]
    label_mapping = {cls: idx for idx, cls in enumerate(classes)}
    numeric_labels = []
    has_valid_labels = False
    for label in true_labels:
        if label != "N/A" and label in label_mapping:
            numeric_labels.append(label_mapping[label])
            has_valid_labels = True
        else:
            numeric_labels.append(-1)
    
    encodings = tokenizer(
        sequences,
        truncation=True,
        padding=True,
        max_length=1024,
        return_tensors="pt",
    )
    
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = torch.tensor(numeric_labels)
    valid_indices = [i for i, label in enumerate(numeric_labels) if label != -1]
    
    test_dataset = TensorDataset(input_ids, attention_mask, labels)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_input_ids, batch_attention_mask, batch_labels = [x.to(device) for x in batch]
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    reverse_label_mapping = {idx: cls for cls, idx in label_mapping.items()}
    predicted_classes = [reverse_label_mapping[p] for p in all_preds]
    
    model_output_dir = os.path.join(output_dir, f"model_{'_'.join(classes)}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    if has_valid_labels and len(valid_indices) > 0:
        valid_preds = [all_preds[i] for i in valid_indices]
        valid_probs = [all_probs[i] for i in valid_indices]
        valid_true_labels = [all_labels[i] for i in valid_indices]
        valid_true_classes = [reverse_label_mapping[l] for l in valid_true_labels]
        
        unique_labels = set(valid_true_labels)
        unique_preds = set(valid_preds)
        all_unique = unique_labels | unique_preds
        
        if len(all_unique) < 2:
            print(f"Warning: Only {len(all_unique)} unique class(es) found in predictions/labels. Skipping metrics calculation.")
            report_csv_path = None
            conf_matrix_path = None
            roc_json_path = None
        else:
            report = classification_report(
                valid_true_labels, valid_preds, target_names=list(classes), 
                labels=list(range(len(classes))), output_dict=True
            )
            report_df = pd.DataFrame(report).transpose()
            report_csv_path = os.path.join(model_output_dir, "classification_report.csv")
            report_df.to_csv(report_csv_path, index=True)
            
            conf_matrix = confusion_matrix(valid_true_labels, valid_preds, labels=list(range(len(classes))))
            plt.figure(figsize=(8, 8))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - {'_'.join(classes)}")
            conf_matrix_path = os.path.join(model_output_dir, "confusion_matrix.png")
            plt.savefig(conf_matrix_path)
            plt.close()
            
            roc_auc_scores = {}
            for i, class_name in enumerate(classes):
                try:
                    roc_auc = roc_auc_score((np.array(valid_true_labels) == i).astype(int), [p[i] for p in valid_probs])
                    roc_auc_scores[f"roc_auc_{class_name}"] = roc_auc
                except ValueError:
                    roc_auc_scores[f"roc_auc_{class_name}"] = "N/A"
            roc_data = {
                "roc_auc_scores": roc_auc_scores,
                "confusion_matrix": conf_matrix.tolist(),
            }
            roc_json_path = os.path.join(model_output_dir, "roc_auc.json")
            with open(roc_json_path, "w") as f:
                json.dump(roc_data, f)
    
    for i, seq_info in enumerate(sequences_by_model[classes]):
        true_event_in_model = None
        if seq_info['true_event'] != "N/A" and seq_info['true_event'] in classes:
            true_event_in_model = seq_info['true_event']
        elif seq_info['true_event'] == "N/A":
            true_event_in_model = "N/A"
        
        all_3class_probs = [
            ('cassette', seq_info['three_class_probs']['cassette']),
            ('alt_three', seq_info['three_class_probs']['alt_three']),
            ('alt_five', seq_info['three_class_probs']['alt_five'])
        ]
        all_3class_probs.sort(key=lambda x: x[1], reverse=True)
        top_3class_overall = all_3class_probs[0][0]
        top_3class_prob = all_3class_probs[0][1]
        
        binary_probs = all_probs[i]
        max_prob_idx = int(np.argmax(binary_probs))
        binary_pred = reverse_label_mapping[max_prob_idx]
        binary_prob = float(binary_probs[max_prob_idx])
        
        if ENSEMBLE_STRATEGY == 'optimized_hybrid':
            is_alt3_alt5 = 'alt_three' in classes and 'alt_five' in classes
            
            if is_alt3_alt5:
                three_class_probs_in_model = {
                    cls: seq_info['three_class_probs'][cls] for cls in classes
                }
                max_3class_class = max(three_class_probs_in_model.items(), key=lambda x: x[1])[0]
                max_3class_prob = three_class_probs_in_model[max_3class_class]
                
                final_prediction = max_3class_class
                final_probability = max_3class_prob
                prediction_source = "optimized_3class_alt3_alt5"
            else:
                combined_probs = {}
                for cls in classes:
                    prob_3class = seq_info['three_class_probs'][cls]
                    prob_binary = all_probs[i][classes.index(cls)]
                    combined_probs[cls] = CASSETTE_ENSEMBLE_WEIGHT_3CLASS * prob_3class + (1 - CASSETTE_ENSEMBLE_WEIGHT_3CLASS) * prob_binary
                
                final_prediction = max(combined_probs.items(), key=lambda x: x[1])[0]
                final_probability = combined_probs[final_prediction]
                prediction_source = "optimized_ensemble_cassette"
        
        else:
            final_prediction = binary_pred
            final_probability = binary_prob
            prediction_source = "binary"
        
        result = {
            'sequence': seq_info['sequence'],
            'true_event_class': seq_info['true_event'],
            'two_class_model': '_'.join(classes),
            'predicted_event_class': final_prediction,
            'predicted_probability': final_probability,
            'prediction_source': prediction_source,
            'true_event_in_model': true_event_in_model,
            'original_index': seq_info['original_index']
        }
        
        for j, cls in enumerate(classes):
            result[f'prob_{cls}'] = float(all_probs[i][j])
        
        result['three_class_prob_cassette'] = seq_info['three_class_probs']['cassette']
        result['three_class_prob_alt_three'] = seq_info['three_class_probs']['alt_three']
        result['three_class_prob_alt_five'] = seq_info['three_class_probs']['alt_five']
        
        all_results.append(result)

if all_results:
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('original_index').reset_index(drop=True)
    results_df = results_df.drop(columns=['original_index'])
    results_df.to_csv(prediction_csv_path, index=False)
    
    prediction_no_seq_path = os.path.join(output_dir, "predictions_with_probabilities_no_sequence.csv")
    results_df_no_seq = results_df.drop(columns=['sequence'])
    results_df_no_seq.to_csv(prediction_no_seq_path, index=False)
    
    all_events = set(results_df_no_seq['true_event_class'].unique()) | set(results_df_no_seq['predicted_event_class'].unique())
    all_events = sorted([e for e in all_events if pd.notna(e) and e != "N/A"])
    
    event_counts = []
    for event in all_events:
        actual_count = len(results_df_no_seq[results_df_no_seq['true_event_class'] == event])
        predicted_count = len(results_df_no_seq[results_df_no_seq['predicted_event_class'] == event])
        true_predicted_count = len(results_df_no_seq[(results_df_no_seq['true_event_class'] == event) & (results_df_no_seq['predicted_event_class'] == event)])
        event_counts.append({
            'event': event,
            'actual_count': actual_count,
            'predicted_count': predicted_count,
            'true_predicted_count': true_predicted_count
        })
    
    counts_df = pd.DataFrame(event_counts)
    counts_csv_path = os.path.join(output_dir, "event_counts.csv")
    counts_df.to_csv(counts_csv_path, index=False)

