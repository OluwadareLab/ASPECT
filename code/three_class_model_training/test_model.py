import os
import sys
# Set environment variables BEFORE any other imports to disable Flash Attention
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["USE_FLASH_ATTENTION_2"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix for PyTorch 2.6+ weights_only issue with checkpoint loading
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Block flash_attn import by creating a stub module
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
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# LoRA support
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with 3 classes."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        kmer: int = -1,
    ):
        super(SupervisedDataset, self).__init__()
        
        # Load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        
        if len(data) == 0:
            raise ValueError(f"No data found in {data_path}")
        
        if len(data[0]) == 2:
            # Data is in the format of [label, text]
            logger.info("Perform single sequence classification...")
            labels_str = [d[0].strip() for d in data]
            texts = [d[1].strip() for d in data]
        elif len(data[0]) == 3:
            # Data is in the format of [label, text1, text2]
            logger.info("Perform sequence-pair classification...")
            labels_str = [d[0].strip() for d in data]
            texts = [[d[1], d[2]] for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        # Get unique classes and create label mapping (3 classes)
        unique_classes = sorted(list(set(labels_str)))
        if len(unique_classes) != 3:
            raise ValueError(f"Expected 3 classes, but found {len(unique_classes)}: {unique_classes}")
        
        # Create label mapping dynamically
        label_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.class_names = unique_classes  # Store class names for later use
        self.label_mapping = label_mapping
        
        # Convert string labels to integers
        labels = [label_mapping[label] for label in labels_str]

        # Use max_length=256 to match training configuration
        # Training uses model_max_length=256, so we must use the same here
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
            padding="longest",
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
    
    # Check if this is a LoRA adapter (has adapter_config.json)
    adapter_config_path = Path(model_path) / "adapter_config.json"
    is_lora = adapter_config_path.exists()
    
    if is_lora:
        logger.info("Detected LoRA adapter. Loading base model and adapter...")
        # Load adapter config to get base model name
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        if base_model_name is None:
            base_model_name = adapter_config.get("base_model_name_or_path", "zhihan1996/DNABERT-2-117M")
        
        logger.info(f"Base model: {base_model_name}")
        logger.info(f"LoRA adapter: {model_path}")
        
        # Load tokenizer from base model
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=True,
            trust_remote_code=True,
            model_max_length=256,  # Match training configuration
        )
        tokenizer.model_max_length = 256  # Ensure it's set correctly
        
        # Load base model
        base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=3,  # Three-class classification
            trust_remote_code=True,
        )
        
        # Disable Flash Attention
        if hasattr(base_model.config, 'use_flash_attention_2'):
            base_model.config.use_flash_attention_2 = False
        if hasattr(base_model.config, 'use_flash_attention'):
            base_model.config.use_flash_attention = False
        if hasattr(base_model.config, '_flash_attn_2_enabled'):
            base_model.config._flash_attn_2_enabled = False
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge adapter into base model for inference
        logger.info("LoRA adapter merged into base model")
    else:
        # Regular model (not LoRA)
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
        
        # Disable Flash Attention
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
            
            # Get predicted classes
            pred_classes = np.argmax(logits, axis=1)
            
            # Get probabilities
            probs = softmax(logits)
            
            all_predictions.extend(pred_classes)
            all_labels.extend(labels)
            all_probabilities.extend(probs)
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def compute_metrics(y_true, y_pred, y_proba, class_names):
    """Compute all classification metrics."""
    metrics = {}
    
    # Overall accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Per-class precision, recall, F1, support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    # Macro averages
    metrics["macro_precision"] = np.mean(precision)
    metrics["macro_recall"] = np.mean(recall)
    metrics["macro_f1"] = np.mean(f1)
    
    # Weighted averages
    metrics["weighted_precision"] = np.average(precision, weights=support)
    metrics["weighted_recall"] = np.average(recall, weights=support)
    metrics["weighted_f1"] = np.average(f1, weights=support)
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        metrics[f"{class_name}_precision"] = float(precision[i])
        metrics[f"{class_name}_recall"] = float(recall[i])
        metrics[f"{class_name}_f1"] = float(f1[i])
        metrics[f"{class_name}_support"] = int(support[i])
    
    # AUC scores (one-vs-rest)
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
    # Verify class order matches expected mapping
    logger.info(f"Generating classification report with class order: {class_names}")
    logger.info(f"Expected mapping: Class 0={class_names[0]}, Class 1={class_names[1]}, Class 2={class_names[2]}")
    
    # Verify label ranges
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    logger.info(f"Unique true labels: {unique_true}")
    logger.info(f"Unique predicted labels: {unique_pred}")
    logger.info(f"Expected labels: {list(range(len(class_names)))}")
    
    if not np.array_equal(np.sort(unique_true), np.arange(len(class_names))):
        logger.warning(f"WARNING: True labels don't match expected range: {unique_true} vs {list(range(len(class_names)))}")
    if not np.array_equal(np.sort(unique_pred), np.arange(len(class_names))):
        logger.warning(f"WARNING: Predicted labels don't match expected range: {unique_pred} vs {list(range(len(class_names)))}")
    
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=False, labels=range(len(class_names))
    )
    
    # Also get dict version for verification
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, labels=range(len(class_names))
    )
    
    # Verify the report matches class names
    logger.info("Classification report per class:")
    for i, class_name in enumerate(class_names):
        if str(i) in report_dict:
            metrics = report_dict[str(i)]
            logger.info(f"  Class {i} ({class_name}): Precision={metrics.get('precision', 0):.4f}, "
                       f"Recall={metrics.get('recall', 0):.4f}, F1={metrics.get('f1-score', 0):.4f}, "
                       f"Support={metrics.get('support', 0)}")
    
    # Save as text file
    with open(output_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Class Mapping: Class 0={class_names[0]}, Class 1={class_names[1]}, Class 2={class_names[2]}\n\n")
        f.write(report)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Classification report saved to {output_path}")
    print("\n" + "=" * 80)
    print("Classification Report")
    print("=" * 80)
    print(f"Class Mapping: Class 0={class_names[0]}, Class 1={class_names[1]}, Class 2={class_names[2]}\n")
    print(report)
    print("=" * 80 + "\n")


def plot_confusion_matrix(y_true, y_pred, class_names, output_path: str, title: str = "Confusion Matrix"):
    """Plot and save confusion matrix."""
    # Ensure labels are in the correct order: [0, 1, 2, ...] matching class_names
    # class_names should be in alphabetical order: ['alt_five', 'alt_three', 'cassette']
    # This matches training: Class 0=alt_five, Class 1=alt_three, Class 2=cassette
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    # Verify the confusion matrix matches expected class order
    logger.info(f"Confusion matrix shape: {cm.shape}")
    logger.info(f"Class names order: {class_names}")
    logger.info(f"Expected mapping: Class 0={class_names[0]}, Class 1={class_names[1]}, Class 2={class_names[2]}")
    logger.info(f"Confusion matrix (rows=true, cols=predicted):\n{cm}")
    
    # Verify row sums match true label counts
    for i, class_name in enumerate(class_names):
        true_count = (y_true == i).sum()
        row_sum = cm[i].sum()
        logger.info(f"Class {i} ({class_name}): {true_count} true samples, {row_sum} in confusion matrix row")
        if true_count != row_sum:
            logger.warning(f"WARNING: Mismatch for class {i} ({class_name}): {true_count} vs {row_sum}")
    
    # Verify column sums match predicted label counts
    for j, class_name in enumerate(class_names):
        pred_count = (y_pred == j).sum()
        col_sum = cm[:, j].sum()
        logger.info(f"Class {j} ({class_name}): {pred_count} predicted samples, {col_sum} in confusion matrix column")
        if pred_count != col_sum:
            logger.warning(f"WARNING: Mismatch for class {j} ({class_name}): {pred_count} vs {col_sum}")
    
    # Create display with correct labels
    # Rows = true labels, Columns = predicted labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_path}")


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
    
    # Create base output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, base_model_name=base_model_name)
    
    # Load test dataset
    logger.info(f"Loading test dataset from {test_data_path}")
    test_dataset = SupervisedDataset(
        data_path=test_data_path,
        tokenizer=tokenizer,
        kmer=-1,
    )
    
    class_names = test_dataset.class_names
    logger.info(f"Classes: {class_names}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Make predictions
    y_pred, y_true, y_proba = predict(model, test_dataset, batch_size=batch_size)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_proba, class_names)
    
    # Determine model name for output files
    if model_name is None:
        model_name = Path(model_path).parent.name.replace("DB2_", "")
    
    # Create model-specific output directory
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {model_output_dir}")
    
    # Generate outputs in model-specific directory
    # 1. Classification report (text)
    report_path = model_output_dir / "classification_report.txt"
    generate_classification_report(y_true, y_pred, class_names, str(report_path))
    
    # 2. Confusion matrix
    cm_path = model_output_dir / "confusion_matrix_test.png"
    plot_confusion_matrix(y_true, y_pred, class_names, str(cm_path), 
                         title=f"Confusion Matrix - {model_name} (Test Set)")
    
    # 3. Metrics JSON
    metrics_path = model_output_dir / "test_metrics.json"
    save_metrics_json(metrics, str(metrics_path))
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Results Summary")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    logger.info(f"Macro AUC: {metrics['macro_auc']:.4f}" if metrics['macro_auc'] else "Macro AUC: N/A")
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
    parser = argparse.ArgumentParser(description="Test three-class LoRA classification models")
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
    
    # Check if model_path points to best_model or parent directory
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

