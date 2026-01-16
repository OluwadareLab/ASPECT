
import os
import sys
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["USE_FLASH_ATTENTION_2"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

class FlashAttnStub:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Flash Attention is disabled.")
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
except ImportError:
    sys.modules['triton'] = _stub_triton

class FlashAttnTritonImportHook:
    def find_spec(self, name, path, target=None):
        if name.endswith('flash_attn_triton') or 'flash_attn_triton' in name:
            from importlib.util import spec_from_loader
            class StubLoader:
                def create_module(self, spec):
                    stub = type(sys)('flash_attn_triton_stub')
                    stub._flash_attn_forward = FlashAttnStub()
                    return stub
                def exec_module(self, module):
                    pass
            return spec_from_loader(name, StubLoader())
        return None

sys.meta_path.insert(0, FlashAttnTritonImportHook())

import csv
import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List
import numpy as np
import transformers
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with 2 classes (binary classification)."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        kmer: int = -1,
    ):
        super(SupervisedDataset, self).__init__()
        
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        
        if len(data) == 0:
            raise ValueError(f"No data found in {data_path}")
        
        if len(data[0]) == 2:
            logger.info("Perform single sequence classification...")
            labels_str = [d[0].strip() for d in data]
            texts = [d[1].strip() for d in data]
        elif len(data[0]) == 3:
            logger.info("Perform sequence-pair classification...")
            labels_str = [d[0].strip() for d in data]
            texts = [[d[1], d[2]] for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        unique_classes = sorted(list(set(labels_str)))
        if len(unique_classes) != 2:
            raise ValueError(f"Expected 2 classes for binary classification, but found {len(unique_classes)}: {unique_classes}")
        
        label_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.class_names = unique_classes
        self.label_mapping = label_mapping
        labels = [label_mapping[label] for label in labels_str]

        try:
            tokenizer_max = getattr(tokenizer, 'model_max_length', 256)
            if tokenizer_max is None or tokenizer_max > 10000:
                tokenizer_max = 256
        except:
            tokenizer_max = 256
        
        max_len = min(256, tokenizer_max)
        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            attention_mask=self.attention_mask[i],
            labels=self.labels[i],
        )


def softmax(logits: np.ndarray):
    """Softmax for numpy array of shape [batch_size, num_classes]."""
    exp_vals = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def load_model_and_tokenizer(model_path: str, base_model_name: str = None):
    """Load trained LoRA model and tokenizer."""
    logger.info(f"Loading model from {model_path}")
    
    adapter_config_path = Path(model_path) / "adapter_config.json"
    is_lora = adapter_config_path.exists()
    
    if is_lora:
        logger.info("Detected LoRA adapter. Loading base model and adapter...")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        if base_model_name is None:
            base_model_name = adapter_config.get("base_model_name_or_path", "zhihan1996/DNABERT-2-117M")
        
        logger.info(f"Base model: {base_model_name}")
        logger.info(f"LoRA adapter: {model_path}")
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=True,
            trust_remote_code=True,
            model_max_length=256,
        )
        tokenizer.model_max_length = 256
        
        base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=2,
            trust_remote_code=True,
        )
        
        if hasattr(base_model.config, 'use_flash_attention_2'):
            base_model.config.use_flash_attention_2 = False
        if hasattr(base_model.config, 'use_flash_attention'):
            base_model.config.use_flash_attention = False
        if hasattr(base_model.config, '_flash_attn_2_enabled'):
            base_model.config._flash_attn_2_enabled = False
        
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        logger.info("LoRA adapter merged into base model")
    else:
        logger.info("Loading regular model (not LoRA)...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        if hasattr(model.config, 'use_flash_attention_2'):
            model.config.use_flash_attention_2 = False
        if hasattr(model.config, 'use_flash_attention'):
            model.config.use_flash_attention = False
        if hasattr(model.config, '_flash_attn_2_enabled'):
            model.config._flash_attn_2_enabled = False
    
    model.to(device)
    model.eval()
    
    return model, tokenizer


def predict(model, dataset, batch_size: int = 32):
    """Make predictions on the dataset."""
    logger.info(f"Making predictions on {len(dataset)} samples...")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            
            pred_classes = np.argmax(logits, axis=1)
            probs = softmax(logits)
            
            all_predictions.extend(pred_classes)
            all_labels.extend(labels)
            all_probabilities.extend(probs)
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def compute_metrics(y_true, y_pred, y_proba, class_names):
    """Compute all classification metrics."""
    metrics = {}
    
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    metrics["macro_precision"] = np.mean(precision)
    metrics["macro_recall"] = np.mean(recall)
    metrics["macro_f1"] = np.mean(f1)
    
    metrics["weighted_precision"] = np.average(precision, weights=support)
    metrics["weighted_recall"] = np.average(recall, weights=support)
    metrics["weighted_f1"] = np.average(f1, weights=support)
    
    for i, class_name in enumerate(class_names):
        metrics[f"{class_name}_precision"] = float(precision[i])
        metrics[f"{class_name}_recall"] = float(recall[i])
        metrics[f"{class_name}_f1"] = float(f1[i])
        metrics[f"{class_name}_support"] = int(support[i])
    
    try:
        auc = roc_auc_score(y_true, y_proba[:, 1])
        metrics["auc"] = float(auc)
    except ValueError as e:
        logger.warning(f"Could not compute AUC: {e}")
        metrics["auc"] = None
    
    auc_scores = []
    for i in range(len(class_names)):
        try:
            auc = roc_auc_score((y_true == i).astype(int), y_proba[:, i])
            metrics[f"{class_names[i]}_auc"] = float(auc)
            auc_scores.append(auc)
        except ValueError as e:
            logger.warning(f"Could not compute AUC for class {class_names[i]}: {e}")
            metrics[f"{class_names[i]}_auc"] = None
    
    metrics["macro_auc"] = float(np.mean(auc_scores)) if auc_scores else None
    
    return metrics


def generate_classification_report(y_true, y_pred, class_names, output_path: str):
    """Generate and save classification report."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=False
    )
    
    with open(output_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Classification report saved to {output_path}")
    print("\n" + "=" * 80)
    print("Classification Report")
    print("=" * 80)
    print(report)
    print("=" * 80 + "\n")


def plot_confusion_matrix(y_true, y_pred, class_names, output_path: str, title: str = "Confusion Matrix"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(y_true, y_proba, class_names, output_path: str, title: str = "ROC Curve"):
    """Plot ROC curve for binary classification with shaded AUC area."""
    if len(class_names) != 2:
        logger.warning("ROC curve plotting is only supported for binary classification")
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
    auc_score = roc_auc_score(y_true, y_proba[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.fill_between(fpr, tpr, color="#E5E5FD")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"ROC curve saved to {output_path}")


def save_metrics_json(metrics: Dict, output_path: str):
    """Save metrics as JSON file."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {output_path}")


def test_model(
    model_path: str,
    test_data_path: str,
    output_dir: str,
    batch_size: int = 32,
    base_model_name: str = None,
    model_name: str = None,
):
    """Test a single model on test data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing model: {model_path}")
    logger.info(f"Test data: {test_data_path}")
    logger.info(f"{'='*80}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer = load_model_and_tokenizer(model_path, base_model_name=base_model_name)
    
    logger.info(f"Loading test dataset from {test_data_path}")
    test_dataset = SupervisedDataset(
        data_path=test_data_path,
        tokenizer=tokenizer,
        kmer=-1,
    )
    
    class_names = test_dataset.class_names
    logger.info(f"Classes: {class_names}")
    logger.info(f"Label mapping: Class 0={class_names[0]}, Class 1={class_names[1]}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    y_pred, y_true, y_proba = predict(model, test_dataset, batch_size=batch_size)
    metrics = compute_metrics(y_true, y_pred, y_proba, class_names)
    
    if model_name is None:
        model_name = Path(model_path).parent.name.replace("DB2_", "")
    
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {model_output_dir}")
    
    report_path = model_output_dir / "classification_report.txt"
    generate_classification_report(y_true, y_pred, class_names, str(report_path))
    
    cm_path = model_output_dir / "confusion_matrix_test.png"
    title_model_name = model_name.replace("_continuation", "")
    title_model_name = re.sub(r'_trial_\d+', '', title_model_name)
    title_model_name = re.sub(r'trial_\d+_', '', title_model_name)
    plot_confusion_matrix(y_true, y_pred, class_names, str(cm_path), 
                         title=f"Confusion Matrix - {title_model_name} (Test Set)")
    
    if len(class_names) == 2:
        roc_path = model_output_dir / "roc_curve_test.png"
        plot_roc_curve(y_true, y_proba, class_names, str(roc_path))
    
    metrics_path = model_output_dir / "test_metrics.json"
    save_metrics_json(metrics, str(metrics_path))
    
    logger.info("\n" + "=" * 80)
    logger.info("Test Results Summary")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    logger.info(f"AUC: {metrics['auc']:.4f}" if metrics['auc'] else "AUC: N/A")
    logger.info("\nPer-class metrics:")
    for class_name in class_names:
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {metrics[f'{class_name}_precision']:.4f}")
        logger.info(f"    Recall: {metrics[f'{class_name}_recall']:.4f}")
        logger.info(f"    F1: {metrics[f'{class_name}_f1']:.4f}")
        logger.info(f"    AUC: {metrics[f'{class_name}_auc']:.4f}" if metrics[f'{class_name}_auc'] else f"    AUC: N/A")
        logger.info(f"    Support: {metrics[f'{class_name}_support']}")
    logger.info("=" * 80 + "\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test binary LoRA classification models")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory (should contain best_model/ with adapter files)"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to test.csv file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Output directory for test results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=None,
        help="Base model name (auto-detected from adapter_config.json if not provided)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for output files (default: inferred from model_path)"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if model_path.name != "best_model" and (model_path / "best_model").exists():
        model_path = model_path / "best_model"
    
    test_model(
        model_path=str(model_path),
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        base_model_name=args.base_model_name,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()

