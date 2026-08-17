import os
import sys
# Set environment variables BEFORE any other imports to disable Flash Attention
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["USE_FLASH_ATTENTION_2"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix for PyTorch 2.6+ weights_only issue with checkpoint loading
# Patch torch.load to use weights_only=False for checkpoint loading
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # If weights_only is not explicitly set, default to False for checkpoint compatibility
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Block flash_attn import by creating a stub module in sys.modules
# This prevents the actual flash_attn module from being imported
class FlashAttnStub:
    """Stub class to replace flash_attn functions"""
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Flash Attention is disabled. Use standard attention instead.")
    def __getattr__(self, name):
        return FlashAttnStub()

# Create stub modules to prevent flash_attn from being imported
# This must happen BEFORE any model code tries to import flash_attn
_stub_module = type(sys)('flash_attn')
_stub_module.flash_attn_func = FlashAttnStub()
_stub_module.flash_attn_qkvpacked_func = FlashAttnStub()
_stub_module.flash_attn_varlen_func = FlashAttnStub()
# Create a mock triton module to prevent compilation errors
_stub_triton = type(sys)('triton')
_stub_triton.jit = FlashAttnStub()
_stub_triton.cdiv = lambda x, y: (x + y - 1) // y  # Simple integer division
sys.modules['flash_attn'] = _stub_module
sys.modules['flash_attn.flash_attn_triton'] = _stub_module
sys.modules['flash_attn.flash_attn_interface'] = _stub_module
# Also stub out any potential triton imports from flash_attn
try:
    import triton
    # Keep triton but make sure it doesn't cause issues
    pass
except ImportError:
    sys.modules['triton'] = _stub_triton

# CRITICAL: Install import hook to intercept flash_attn_triton imports
# This catches imports from model code files (e.g., from HuggingFace cache)
class FlashAttnTritonImportHook:
    """Import hook to replace flash_attn_triton with a stub"""
    def find_spec(self, name, path, target=None):
        if name.endswith('flash_attn_triton') or 'flash_attn_triton' in name:
            # Return a spec that loads our stub module
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

import csv
import copy
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
import optuna.visualization as vis

import torch
import transformers
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# LoRA
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# ADASYN
from imblearn.over_sampling import ADASYN
from collections import Counter

# WandB
import wandb

# Optuna
import optuna

# Metrics
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score, 
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

logger = logging.getLogger(__name__)

# Don't set CUDA_VISIBLE_DEVICES here - let Docker handle GPU assignment
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Commented out - use Docker --gpus flag instead
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()                                                                                                    

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M") 
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})

    # New flags
    use_wandb: bool = field(default=False, metadata={"help": "Whether to log to Weights & Biases"})
    use_optuna: bool = field(default=True, metadata={"help": "Whether to run hyperparameter search with Optuna"})
    use_class_weights: bool = field(default=False, metadata={"help": "Whether to apply class weights"})
    apply_adasyn: bool = field(default=False, metadata={"help": "Whether to apply ADASYN oversampling"})
    optuna_target_metric: str = field(
        default="auto",
        metadata={"help": "Optuna objective metric: auto, weighted, macro, or blend"}
    )
    stabilize_training: bool = field(
        default=False,
        metadata={
            "help": (
                "Opt-in stability tweaks for imbalanced/collapsing datasets. "
                "Default False to preserve previous behavior/results."
            )
        },
    )


@dataclass
class DataArguments:                                                                
    data_path: str = field(default='./data/512_split_cons_cass', metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer (use BPE tokenization for DNABERT-2)."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="./output")
    logging_dir: str = field(default="./logs")
    cache_dir: Optional[str] = field(default=None)
    report_to: List[str] = field(default_factory=lambda: [])  # Empty list by default, will be set to ["wandb"] if use_wandb is True
    run_name: str = field(default="DB2")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=256, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=32)#change 32
    per_device_eval_batch_size: int = field(default=32)#change 32
    num_train_epochs: int = field(default=15)
    fp16: bool = field(default=False) # Set to False by default, will be enabled if CUDA is available
    logging_steps: int = field(default=50)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    warmup_steps: int = field(default=500)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    load_best_model_at_end: bool = field(default=True)
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=10)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_weighted_f1")
    #metric_for_best_model: str = field(default="eval_class_1_f1")
    greater_is_better: bool = field(default=True)
    save_total_limit: int = field(default=1)
    seed: int = field(default=42)
    # For Optuna
    optuna_trials: int = field(default=15, metadata={"help": "Number of trials for Optuna HPO"})


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def get_last_checkpoint(output_dir):
    # Looks for subdirectories named 'checkpoint-NUMBER' and returns the highest one
    checkpoint_paths = []
    for folder_name in os.listdir(output_dir):
        full_path = os.path.join(output_dir, folder_name)
        if (
            os.path.isdir(full_path)
            and folder_name.startswith("checkpoint-")
            and folder_name[len("checkpoint-"):].isdigit()
        ):
            checkpoint_paths.append((int(folder_name.split("-")[1]), full_path))

    if not checkpoint_paths:
        return None
    # Return path with largest checkpoint number
    checkpoint_paths.sort(key=lambda x: x[0], reverse=True)
    return checkpoint_paths[0][1]

def get_alter_of_dna_sequence(sequence: str):
    """Get the reversed complement of the original DNA sequence."""
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([MAP[c] for c in sequence])


def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i : i + k] for i in range(len(sequence) - k + 1)])


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
    return kmer


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        kmer: int = -1,
        label_mapping: Optional[Dict[str, int]] = None,
        # apply_adasyn: bool = False,
    ):
        super(SupervisedDataset, self).__init__()
        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [label, sequence]
            logging.warning("Perform single sequence classification...")
            texts = [d[1].strip() for d in data]
            observed_labels = [d[0].strip() for d in data]
            if label_mapping is None:
                # Build mapping from this split (typically training split).
                unique_labels = sorted(set(observed_labels))
                label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            else:
                # Reuse provided mapping for consistent label ids across splits.
                missing_labels = sorted(set(observed_labels) - set(label_mapping.keys()))
                if missing_labels:
                    raise ValueError(
                        f"Found labels not present in provided mapping for {data_path}: {missing_labels}"
                    )
                unique_labels = sorted(label_mapping, key=label_mapping.get)

            self.label_mapping = label_mapping
            self.class_names = unique_labels
            labels = [label_mapping[d[0].strip()] for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[1], d[1]] for d in data]
            labels = [int(d[0]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        # # Optionally apply ADASYN oversampling (binary or multi-class)
        # if apply_adasyn:
        #     logging.warning("Applying ADASYN oversampling on the minority class...")
        #     # Flatten 'texts' if it is list of lists:
        #     if isinstance(texts[0], list):
        #         # If 2-sequence input, join them for oversampling. 
        #         # Or handle with caution for multi-seq scenarios
        #         texts_flat = [" ".join(t) for t in texts]
        #     else:
        #         texts_flat = texts

        #     # Convert text -> numeric features for ADASYN
        #     from sklearn.feature_extraction.text import CountVectorizer
        #     vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,1))
        #     X = vectorizer.fit_transform(texts_flat).toarray()
        #     y = np.array(labels)

        #     class_counts = Counter(y)
        #     logging.warning(f"Class distribution before ADASYN: {class_counts}")

        #     adasyn = ADASYN(random_state=42)
        #     X_res, y_res = adasyn.fit_resample(X, y)
        #     # Convert back to strings
        #     X_res_text = vectorizer.inverse_transform(X_res)
        #     X_res_text = ["".join(tokens) for tokens in X_res_text]

        #     # Reassign
        #     texts = X_res_text
        #     labels = y_res.tolist()

        # DNABERT-2 uses BPE tokenization, so by default (kmer=-1) we pass raw sequences
        # to the tokenizer. Only use k-mer if explicitly specified (for other models).
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.is_initialized() and torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            # If texts is list of lists, we do k-mer for each piece, 
            # but often DNABERT expects single seq -> adapt as needed
            if isinstance(texts[0], list):
                # For pair classification, generate k-mer separately
                texts = [
                    " ".join([load_or_generate_kmer(data_path, [t], kmer)[0] for t in pair]) 
                    for pair in texts
                ]
            else:
                texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        else:
            # kmer=-1: Use BPE tokenization (DNABERT-2 default)
            # Raw DNA sequences are passed directly to the tokenizer
            logging.warning("Using raw DNA sequences with BPE tokenization (DNABERT-2 default)")

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))
        # label_mapping and class_names are already set above for 2-column format

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            attention_mask=self.attention_mask[i],
            labels=self.labels[i],
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "attention_mask", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.tensor(labels, dtype=torch.long)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

# Not used in the current pipeline - Function below is used to compute metrics
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    """
    Example of extended metrics: accuracy, f1, matthews, precision, recall,
    plus AUC and PR-AUC if binary classification.
    """
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    # Basic metrics
    acc = accuracy_score(valid_labels, valid_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels, valid_predictions, average="macro", zero_division=0
    )
    # For your script's existing metrics
    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
    return metrics


def extended_compute_metrics(eval_pred):
    """
    Extended compute_metrics with AUC and PR-AUC if binary classification.
    """
    predictions, labels = eval_pred

    # Convert logits to probabilities
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    # Argmax for predicted class
    pred_class = np.argmax(predictions, axis=1)
    
    # Classification report for detailed metrics
    class_report = classification_report(labels, pred_class, output_dict=True)

    # Extract F1-score for class "1" (use integer key, fallback to string)
    # Handle both string and integer keys in classification_report
    class_1_f1 = 0.0
    if "1" in class_report:
        class_1_f1 = class_report["1"]["f1-score"]
    elif 1 in class_report:
        class_1_f1 = class_report[1]["f1-score"]
    elif len(class_report) > 2:  # More than just accuracy and macro avg
        # Get the last class (assuming binary classification)
        numeric_keys = [k for k in class_report.keys() if isinstance(k, (int, str)) and str(k).isdigit()]
        if numeric_keys:
            last_class_key = max(numeric_keys, key=lambda x: int(x) if isinstance(x, str) else x)
            class_1_f1 = class_report[last_class_key]["f1-score"]
    
    weighted_f1 = class_report.get("weighted avg", {}).get("f1-score", 0.0)
    macro_f1 = class_report.get("macro avg", {}).get("f1-score", 0.0)
    
    # Optional: Include other metrics for logging
    return {
        "eval_class_1_f1": class_1_f1,  # Key metric for model selection
        "accuracy": class_report.get("accuracy", 0.0),
        "eval_weighted_f1": weighted_f1,
        "eval_macro_f1": macro_f1,
    }



def softmax(logits: np.ndarray):
    """Softmax for numpy array of shape [batch_size, num_classes]."""
    exp_vals = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return logits.detach().cpu()  # Return tensor, not numpy

# Define function to plot confusion matrix
def plot_confusion_matrix(trainer, eval_dataset, results_dir, runname_label, class_names=None):
    results_dir = Path(results_dir)
    
    predictions, labels, _ = trainer.predict(eval_dataset)
    preds = np.argmax(predictions, axis=-1)
    cm = confusion_matrix(labels, preds)
    
    # Use provided class names or default
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(labels)))]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {runname_label}")
    
    plt.savefig(results_dir / f"confusion_matrix_{runname_label}.png")
    plt.close()
    

# Define the FocalLoss Class
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        
        Args:
            alpha (torch.Tensor, optional): Class weights. Shape [num_classes].
            gamma (float, optional): Focusing parameter.
            reduction (str, optional): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=self.alpha, reduction='none')  # We'll handle reduction

    def forward(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)  # Shape [batch_size]
        pt = torch.exp(-ce_loss)  # Probability of the true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Compute the inverse frequency of each class
def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Compute class weights as the inverse frequency of each class.
    
    Args:
        labels (List[int]): List of class labels.
        num_classes (int): Number of classes.
        
    Returns:
        torch.Tensor: Class weights tensor.
    """
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
    weights = torch.tensor(class_weights, dtype=torch.float32)
    return weights

# Create a Custom Trainer
class CustomTrainer(transformers.Trainer):
    """
    Custom Trainer to use Focal Loss instead of CrossEntropyLoss.
    """
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Check CUDA availability and adjust FP16 accordingly
    if training_args.fp16 and not torch.cuda.is_available():
        logging.warning("FP16 requested but CUDA not available. Disabling FP16.")
        training_args.fp16 = False
    elif not training_args.fp16 and torch.cuda.is_available():
        # Auto-enable FP16 if CUDA is available and not explicitly disabled
        logging.info("CUDA available. Enabling FP16 for faster training.")
        training_args.fp16 = True

    # Extract the actual dataset name (e.g., "constitutive_vs_cassette" from the full path)
    dataset_name = os.path.basename(os.path.normpath(data_args.data_path))

    # Apply stability tweaks ONLY for known-collapsing datasets.
    # This keeps all other datasets' behavior/results unchanged by default.
    _STABILITY_ALLOWLIST = {
        "AA_vs_ES_vs_RI",
        "AD_vs_ES",
        "ES_vs_ME",
    }
    stabilize_for_this_dataset = bool(model_args.stabilize_training) or (dataset_name in _STABILITY_ALLOWLIST)

    # Base results directory - can be overridden with environment variable
    base_results_root = Path(os.environ.get("RESULTS_DIR", "./result_4"))
    default_results_dir = base_results_root / f"DB2_{dataset_name}"
    results_dir = default_results_dir

    # For datasets that previously crashed and left inconsistent checkpoints
    # (e.g., DB2_constitutive_vs_alt_three with missing checkpoint-592),
    # use a clean "retry" directory so HuggingFace Trainer does not see
    # stale checkpoints / trainer_state from an earlier run.
    if dataset_name == "constitutive_vs_alt_three" and default_results_dir.exists():
        retry_results_dir = base_results_root / f"DB2_{dataset_name}_retry"
        logging.warning(
            "Existing results directory detected for %s at %s. "
            "Using clean retry directory instead: %s",
            dataset_name,
            default_results_dir,
            retry_results_dir,
        )
        results_dir = retry_results_dir

    # results_dir = Path("./results") / f"DB2_{dataset_name}_OverSampled" if model_args.apply_adasyn else f"DB2_{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Update output directory and logging directory
    training_args.output_dir = str(results_dir / "model_output")
    training_args.logging_dir = str(results_dir / "logs")
    
    # Handle WandB initialization
    if not model_args.use_wandb:
        training_args.report_to = []  # Empty list = no reporting
        run_name = "no_wandb_run"  # Default run name when WandB is not used
        logging.info("WandB disabled. Training logs will not be sent to WandB.")
    else:
        # Enable wandb reporting and initialize
        training_args.report_to = ["wandb"]  # List with "wandb"
        run_name = f"DB22_{dataset_name}"
        try:
            wandb.init(
                entity=os.environ.get("WANDB_ENTITY") or None,
                project=os.environ.get("WANDB_PROJECT", "hierarchical-aspect"),
                name=run_name,
            )
            wandb.config.update({**vars(model_args), **vars(data_args), **vars(training_args)})
            logging.info("WandB initialized successfully.")
        except Exception as e:
            logging.warning(f"Failed to initialize WandB: {e}. Continuing without WandB.")
            training_args.report_to = []  # Empty list = no reporting
            model_args.use_wandb = False


    # Set random seed
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    
    # Build CSV file to load based on the apply_adasyn flag
    train_data_path = os.path.join(
            data_args.data_path, 
            "train_oversampled.csv" if model_args.apply_adasyn else "train.csv"
        )

    logging.warning(f"Loading dataset: {train_data_path}")

    # Load dataset
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=train_data_path,
        kmer=data_args.kmer,
        label_mapping=None,
        # apply_adasyn=False,  # Make sure we do NOT run ADASYN inside this constructor
    )

    val_data_path = os.path.join(data_args.data_path, "dev.csv")
    if not os.path.exists(val_data_path):
        alt_val_data_path = os.path.join(data_args.data_path, "val.csv")
        if os.path.exists(alt_val_data_path):
            logging.warning(f"'dev.csv' not found. Using '{alt_val_data_path}' as validation split.")
            val_data_path = alt_val_data_path
        else:
            raise FileNotFoundError(
                f"Neither dev.csv nor val.csv found in data_path={data_args.data_path}"
            )

    val_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=val_data_path,
        kmer=data_args.kmer,
        label_mapping=getattr(train_dataset, "label_mapping", None),
        # apply_adasyn=False,  # Only oversample the training set
    )
    # Keep test set separate - not used during training or evaluation
    # test_dataset = SupervisedDataset(
    #     tokenizer=tokenizer,
    #     data_path=os.path.join(data_args.data_path, "test.csv"),
    #     kmer=data_args.kmer,
    #     # apply_adasyn=False,
    # )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # 4. Define the Training Function with Focal Loss
    # ============================
    def build_and_train_model(learning_rate, weight_decay, num_train_epochs, batch_size, alpha=None, gamma=2.0):
        """Helper function for normal or Optuna-based training loops."""
        # Ensure Flash Attention is disabled before loading model
        os.environ["USE_FLASH_ATTENTION"] = "0"
        os.environ["DISABLE_FLASH_ATTENTION"] = "1"
        os.environ["USE_FLASH_ATTENTION_2"] = "0"
        # Additional environment variables that some models check
        os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
        os.environ["MAX_JOBS"] = "1"  # Limit parallel compilation
        
        # Try to load model config first and modify it
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                trust_remote_code=True,
            )
            # Set num_labels in config
            if hasattr(config, 'num_labels'):
                config.num_labels = train_dataset.num_labels
            # Disable Flash Attention in config if attributes exist
            if hasattr(config, 'use_flash_attention_2'):
                config.use_flash_attention_2 = False
            if hasattr(config, 'use_flash_attention'):
                config.use_flash_attention = False
            if hasattr(config, '_flash_attn_2_enabled'):
                config._flash_attn_2_enabled = False
            # Additional config attributes that might control Flash Attention
            for attr in ['_attn_implementation', 'attn_implementation']:
                if hasattr(config, attr):
                    setattr(config, attr, 'eager')  # Use eager (standard) attention
        except Exception as e:
            logging.warning(f"Could not modify config: {e}. Proceeding with default config.")
            config = None
        
        # load model
        model_kwargs = {
            "cache_dir": training_args.cache_dir,
            "trust_remote_code": True,
        }
        # Only add num_labels if we're not using a custom config
        if config is None:
            model_kwargs["num_labels"] = train_dataset.num_labels
        else:
            # If using custom config, num_labels should be set in config
            model_kwargs["config"] = config
            
        # CRITICAL FIX: We'll patch flash_attn_triton.py files AFTER model download
        # The patching happens in a hook that runs after files are downloaded
        def patch_flash_attn_files():
            """Find and patch all flash_attn_triton.py files in HuggingFace cache"""
            cache_dir = training_args.cache_dir or os.path.expanduser("~/.cache/huggingface")
            hub_path = os.path.join(cache_dir, "hub")
            
            if not os.path.exists(hub_path):
                return
            
            # Find all flash_attn_triton.py files
            for root, dirs, files in os.walk(hub_path):
                if "flash_attn_triton.py" in files:
                    flash_file = os.path.join(root, "flash_attn_triton.py")
                    try:
                        # Create a stub file that raises an error instead of compiling
                        stub_content = '''# PATCHED: Flash Attention disabled
# This file has been patched to prevent Triton compilation errors
# The model will fall back to standard attention

def _flash_attn_forward(*args, **kwargs):
    raise RuntimeError("Flash Attention is disabled. The model should use standard attention instead.")

# Export the function that the model code expects
__all__ = ['_flash_attn_forward']
'''
                        with open(flash_file, 'w') as f:
                            f.write(stub_content)
                        logging.info(f"Patched {flash_file} to disable Flash Attention")
                    except Exception as e:
                        logging.warning(f"Could not patch {flash_file}: {e}")
        
        # Patch files before loading (in case they already exist)
        patch_flash_attn_files()
        
        # Load model - the stub modules and environment variables should prevent Flash Attention
        # IMPORTANT: We need to prevent the model from trying to compile Flash Attention kernels
        # The model code downloads flash_attn_triton.py which tries to compile Triton kernels
        # We'll catch and suppress any Flash Attention related errors during model loading
        
        try:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                **model_kwargs
            )
            # Patch files again after download (in case new files were downloaded)
            patch_flash_attn_files()
        except Exception as e:
            error_str = str(e).lower()
            if "flash" in error_str or "triton" in error_str or "compilation" in error_str:
                # Try patching files and retrying
                logging.warning(f"Model loading failed due to Flash Attention/Triton: {e}")
                logging.info("Patching model files and retrying...")
                patch_flash_attn_files()
                # Retry once
                try:
                    model = transformers.AutoModelForSequenceClassification.from_pretrained(
                        model_args.model_name_or_path,
                        **model_kwargs
                    )
                except Exception as retry_error:
                    logging.error(f"Retry also failed: {retry_error}")
                    raise RuntimeError(
                        "Cannot load DNABERT-2 model: Flash Attention compilation error persists. "
                        "The model code tries to compile Triton kernels for Flash Attention. "
                        "Please contact the model authors or use a different model that doesn't require Flash Attention."
                    ) from retry_error
            else:
                raise
        
        # Disable Flash Attention in model config after loading (double-check)
        if hasattr(model.config, 'use_flash_attention_2'):
            model.config.use_flash_attention_2 = False
        if hasattr(model.config, 'use_flash_attention'):
            model.config.use_flash_attention = False
        if hasattr(model.config, '_flash_attn_2_enabled'):
            model.config._flash_attn_2_enabled = False
        
        # CRITICAL: Patch the model to prevent Flash Attention usage
        # DNABERT-2's code tries to use Flash Attention even if package is uninstalled
        # We need to patch the model's attention mechanism
        def patch_model_attention(model):
            """Recursively patch all attention layers to disable Flash Attention"""
            for name, module in model.named_modules():
                # Patch common Flash Attention attributes
                if hasattr(module, '_flash_attn_enabled'):
                    module._flash_attn_enabled = False
                if hasattr(module, 'use_flash_attention_2'):
                    module.use_flash_attention_2 = False
                if hasattr(module, 'use_flash_attention'):
                    module.use_flash_attention = False
                # If module has a method that tries to import flash_attn, patch it
                if hasattr(module, '_flash_attention_forward'):
                    # Replace with standard attention forward
                    original_forward = module.forward if hasattr(module, 'forward') else None
                    if original_forward:
                        def patched_forward(*args, **kwargs):
                            # Force use of standard attention
                            kwargs.pop('use_flash_attention_2', None)
                            kwargs.pop('use_flash_attention', None)
                            return original_forward(*args, **kwargs)
                        module.forward = patched_forward
        
        # Apply patches
        try:
            patch_model_attention(model)
            logging.info("Successfully patched model to disable Flash Attention")
        except Exception as e:
            logging.warning(f"Could not patch model attention: {e}. Continuing anyway.")
        
        # Additional safety: Monkey-patch any flash_attn imports in the model's namespace
        import types
        def create_noop_flash_attn():
            """Create a no-op module that raises an error if Flash Attention is called"""
            class NoOpFlashAttn:
                def __getattr__(self, name):
                    raise RuntimeError(
                        "Flash Attention is disabled. The model should use standard attention. "
                        "If you see this error, the model code is trying to use Flash Attention. "
                        "Please ensure the model uses standard attention mechanisms."
                    )
            return NoOpFlashAttn()
        
        # Try to set flash_attn in model's __dict__ if it exists
        if hasattr(model, '__dict__'):
            model.__dict__['flash_attn'] = create_noop_flash_attn()

        # configure LoRA
        if model_args.use_lora:
        # Dynamically choose target_modules based on the model's architecture
            model_str = str(model)
            if "query" in model_str and "value" in model_str:
                target_modules = ["query", "value"]
            elif "Wqkv" in model_str:
                target_modules = ["Wqkv"]
            else:
                logging.warning("Could not detect typical target modules. Using default ['Wqkv'].")
                target_modules = ["Wqkv"]

            logging.info(f"Using LoRA target_modules: {target_modules}")

            try:
                lora_config = LoraConfig(
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    target_modules=target_modules,  # Use dynamically determined modules
                    lora_dropout=model_args.lora_dropout,
                    bias="none",
                    task_type="SEQ_CLS",
                    inference_mode=False,
                )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
            except Exception as e:
                logging.warning(f"Failed to apply dynamic target_modules due to: {e}")
                logging.warning("Falling back to default target_modules=['Wqkv']")
                lora_config = LoraConfig(
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    target_modules=["Wqkv"],
                    lora_dropout=model_args.lora_dropout,
                    bias="none",
                    task_type="SEQ_CLS",
                    inference_mode=False,
                )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()



        if model_args.use_class_weights:
            # Compute class weights if alpha is not provided
            if alpha is None:
                class_weights = compute_class_weights(train_dataset.labels, train_dataset.num_labels)
            else:
                class_weights = alpha

            class_weights = class_weights.to("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize Focal Loss
            loss_fn = FocalLoss(alpha=class_weights, gamma=gamma, reduction='mean')
        
        
        # Override training_args with possible optuna suggestions
        local_training_args = copy.deepcopy(training_args)
        local_training_args.learning_rate = learning_rate
        local_training_args.weight_decay = weight_decay
        local_training_args.num_train_epochs = num_train_epochs
        local_training_args.per_device_train_batch_size = batch_size

        # Define trainer with focal loss or with cross-entropy loss
        last_checkpoint = None
        if (
            os.path.isdir(local_training_args.output_dir)
            and not local_training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(local_training_args.output_dir)
            if last_checkpoint is not None:
                logger.info(f"Resuming from checkpoint: {last_checkpoint}")
                
        if model_args.use_class_weights:
            # Define custom trainer with Focal Loss
            trainer = CustomTrainer(
                model=model,
                tokenizer=tokenizer,
                args=local_training_args,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                compute_metrics=extended_compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                loss_fn=loss_fn,  # Pass the focal loss function
                # callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)]
            )
        else:
            # Define custom trainer with standard cross-entropy loss
            trainer = transformers.Trainer(
                model=model,
                tokenizer=tokenizer,
                args=local_training_args,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                compute_metrics=extended_compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                # callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
        try:
            trainer.train(resume_from_checkpoint=last_checkpoint)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise e

        if local_training_args.save_model:
            if model_args.use_lora:
                # For a LoRA adapter, save using the PEFT method
                trainer.model.save_pretrained(local_training_args.output_dir)
            else:
                # For a full model checkpoint, use your safe_save_model_for_hf_trainer function
                trainer.save_state()
                safe_save_model_for_hf_trainer(trainer=trainer, output_dir=local_training_args.output_dir)

        # Evaluate on validation set (test set is kept separate)
        if local_training_args.eval_and_save_results:
            results_path = os.path.dirname(local_training_args.output_dir)            
            # results_path = os.path.join(local_training_args.output_dir, "results")
            results = trainer.evaluate(eval_dataset=val_dataset)
            os.makedirs(results_path, exist_ok=True)
            with open(os.path.join(results_path, f"{dataset_name}_eval_results.json"), "w") as f:
                json.dump(results, f)
                
            # Get class names from dataset
            class_names = getattr(train_dataset, 'class_names', None)
            plot_confusion_matrix(trainer, val_dataset, results_path, run_name, class_names=class_names)
            
        return trainer


    # Use optuna setting from arguments (don't override)
    if model_args.use_optuna:
# ===============================
#   Hyperparameter Search
# ===============================
        best_model_dir = str(results_dir / "best_model")  # Directory to save the best model

        train_size = len(train_dataset)
        num_classes = train_dataset.num_labels
        train_label_names = set(getattr(train_dataset, "label_mapping", {}).keys())
        requested_target_metric = str(model_args.optuna_target_metric).strip().lower()
        if requested_target_metric == "auto":
            # Preserve prior default behavior unless this dataset is in the stability allowlist.
            if stabilize_for_this_dataset and num_classes > 2:
                effective_target_metric = "macro"
            else:
                effective_target_metric = "macro" if "ME" in train_label_names else "blend"
        else:
            effective_target_metric = requested_target_metric

        if effective_target_metric not in {"weighted", "macro", "blend"}:
            raise ValueError(
                "optuna_target_metric must be one of: auto, weighted, macro, blend. "
                f"Got: {model_args.optuna_target_metric}"
            )

        if effective_target_metric == "macro":
            training_args.metric_for_best_model = "eval_macro_f1"
        else:
            training_args.metric_for_best_model = "eval_weighted_f1"

        logger.info(
            "Optuna target metric requested=%s effective=%s labels=%s metric_for_best_model=%s",
            requested_target_metric,
            effective_target_metric,
            sorted(train_label_names),
            training_args.metric_for_best_model,
        )

        # Dataset-aware Optuna search ranges (keeps training core unchanged).
        if train_size < 2000:
            epoch_min, epoch_max = 8, 20
            batch_choices = [8, 16, 32, 64]
            lr_min, lr_max = 1e-5, 2e-4
        elif train_size < 10000:
            epoch_min, epoch_max = 10, 25
            batch_choices = [16, 32, 64, 128]
            lr_min, lr_max = 1e-5, 2e-4
        else:
            epoch_min, epoch_max = 12, 30
            batch_choices = [32, 64, 128]
            lr_min, lr_max = 1e-5, 3e-4

        if stabilize_for_this_dataset:
            # Opt-in stability tweaks for collapsing/imbalanced datasets.
            # - For 3+ classes with class-weights enabled: force gamma=0.0 (i.e., weighted cross-entropy).
            # - For binary with class-weights enabled: search a safer gamma range including 0.0.
            if num_classes > 2 and model_args.use_class_weights:
                fixed_gamma = 0.0
                gamma_min, gamma_max = 0.0, 0.0
            elif num_classes == 2 and model_args.use_class_weights:
                fixed_gamma = None
                gamma_min, gamma_max = 0.0, 2.0
            else:
                fixed_gamma = None
                gamma_min, gamma_max = (2.0, 4.0) if num_classes == 2 else (1.5, 3.5)
        else:
            # Default: keep the previous search ranges/behavior for reproducibility.
            fixed_gamma = None
            gamma_min, gamma_max = (2.0, 4.0) if num_classes == 2 else (1.5, 3.5)

        def objective(trial):
            print(f"Starting trial {trial.number}")

            # Hyperparameter suggestions
            learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            num_train_epochs = trial.suggest_int("num_train_epochs", epoch_min, epoch_max)
            batch_size = trial.suggest_categorical("batch_size", batch_choices)
            gamma = fixed_gamma if fixed_gamma is not None else trial.suggest_float("gamma", gamma_min, gamma_max)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            alpha = None
            # Keep original per-class alpha search for binary setups only.
            # For multi-class, fall back to compute_class_weights() inside training.
            if model_args.use_class_weights and train_dataset.num_labels == 2:
                alpha = torch.tensor([
                    trial.suggest_float("alpha_class_0", 0.1, 1.0),
                    trial.suggest_float("alpha_class_1", 0.1, 1.0)
                ], dtype=torch.float32).to(device)

            # Train model with current trial's hyperparameters
            trainer = build_and_train_model(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                num_train_epochs=num_train_epochs,
                batch_size=batch_size,
                alpha=alpha,
                gamma=gamma
            )

            # Evaluate model on validation set (test set is kept separate)
            metrics = trainer.evaluate(val_dataset)
            logger.info(f"Computed metrics for trial {trial.number}: {metrics}")

            # Save the best model based on eval_class_1_f1
            #eval_class_1_f1 = metrics["eval_class_1_f1"]
            #eval_weighted_f1 = metrics["weighted_f1"]
            #current_score = metrics["eval_class_1_f1"]
            #combined_score = 0.4 * eval_class_1_f1 + 0.6 * eval_weighted_f1
            #current_score=combined_score
            #if trial.study.best_value is None or current_score > trial.study.best_value:
                #logger.info(f"New best trial found: {trial.number} with eval_class_1_f1: {current_score}")
            eval_class_1_f1 = metrics.get("eval_class_1_f1", 0)
            eval_weighted_f1 = metrics.get("eval_weighted_f1", 0)
            eval_macro_f1 = metrics.get("eval_macro_f1", 0)
            if eval_weighted_f1 is None or eval_class_1_f1 is None or eval_macro_f1 is None:
                raise ValueError(f"Metrics are missing required keys: {metrics}")

            # Debug missing keys
            #if "weighted_f1" not in metrics:
             #   print(f"Error: 'weighted_f1' not found in metrics for trial {trial.number}")
              #  print(f"Available keys: {metrics.keys()}")

    # Combine metrics

            target_metric = effective_target_metric
            if target_metric == "macro":
                current_score = eval_macro_f1
            elif target_metric == "blend":
                # Bias slightly toward minority-sensitive macro F1.
                current_score = 0.7 * eval_macro_f1 + 0.3 * eval_weighted_f1
            else:
                # Default: weighted F1 for overall stability on imbalanced sets.
                current_score = eval_weighted_f1
            # Check if this is the best trial so far
            # Handle first trial (no best_trial yet) or compare with best value
            is_best = False
            # Check number of completed trials first to avoid accessing best_trial/best_value
            # which raises ValueError if no trials are completed yet
            try:
                n_completed = len([t for t in trial.study.trials if t.state.name == 'COMPLETE'])
                if n_completed == 0:
                    # This is the first completed trial, so it's automatically the best
                    is_best = True
                else:
                    # Compare with best value from completed trials
                    try:
                        best_value = trial.study.best_value
                        is_best = current_score > best_value
                    except (ValueError, AttributeError):
                        # Fallback: if we can't get best_value, assume this is best
                        is_best = True
            except (ValueError, AttributeError, Exception):
                # If anything fails, assume this is the first/best trial
                is_best = True
            
            if is_best:
                logger.info(
                    "New best trial found: %s with %s score %.6f",
                    trial.number,
                    target_metric,
                    current_score,
                )
                if model_args.use_lora:
                    trainer.model.save_pretrained(best_model_dir)
                else:
                    trainer.save_state()
                    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=best_model_dir)

            # Return the metric to optimizes
            return current_score
            #return combined_score
            

        # Runs Hyper-Parameter Search Model Loop
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=training_args.optuna_trials)
        print(f"Optuna will run {training_args.optuna_trials} trials")
        #vis.plot_optimization_history(study).show()
        #vis.plot_param_importances(study).show()
        logger.info(f"Optuna finished. Best params: {study.best_params}")
        if model_args.use_wandb:
            wandb.log({"best_optuna_params": study.best_params})

    else:
# ===========================
#   Normal Single Training
# ===========================
        build_and_train_model(
            learning_rate=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            num_train_epochs=training_args.num_train_epochs,
            batch_size=training_args.per_device_train_batch_size,
        )
    
    if model_args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
