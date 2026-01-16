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
    """Stub class to replace flash_attn functions"""
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Flash Attention is disabled. Use standard attention instead.")
    def __getattr__(self, name):
        return FlashAttnStub()

_stub_module = type(sys)('flash_attn')
_stub_module.flash_attn_func = FlashAttnStub()
_stub_module.flash_attn_qkvpacked_func = FlashAttnStub()
_stub_module.flash_attn_varlen_func = FlashAttnStub()
# Create a mock triton module to prevent compilation errors
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

# CRITICAL: Install import hook to intercept flash_attn_triton imports
# This catches imports from model code files (e.g., from HuggingFace cache)
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

import csv
import copy
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from collections import Counter
import wandb
import optuna
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score, 
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()                                                                                                    

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M") 
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})

    use_wandb: bool = field(default=False, metadata={"help": "Whether to log to Weights & Biases"})
    use_optuna: bool = field(default=True, metadata={"help": "Whether to run hyperparameter search with Optuna"})
    use_class_weights: bool = field(default=False, metadata={"help": "Whether to apply class weights"})
    apply_adasyn: bool = field(default=False, metadata={"help": "Whether to apply ADASYN oversampling"})


@dataclass
class DataArguments:                                                                
    data_path: str = field(default='../data_preprocessing/balanced_three_class_datasets/cassette_alt_three_alt_five', metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer (use BPE tokenization for DNABERT-2)."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="./output")
    logging_dir: str = field(default="./logs")
    cache_dir: Optional[str] = field(default=None)
    report_to: List[str] = field(default_factory=lambda: [])
    run_name: str = field(default="DB2")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=256, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=32)
    num_train_epochs: int = field(default=10)
    fp16: bool = field(default=False)
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
    metric_for_best_model: str = field(default="eval_combined_score")
    greater_is_better: bool = field(default=True)
    save_total_limit: int = field(default=1)
    seed: int = field(default=42)
    optuna_trials: int = field(default=30, metadata={"help": "Number of trials for Optuna HPO"})


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def get_last_checkpoint(output_dir):
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
    checkpoint_paths.sort(key=lambda x: x[0], reverse=True)
    return checkpoint_paths[0][1]

def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([MAP[c] for c in sequence])


def generate_kmer_str(sequence: str, k: int) -> str:
    return " ".join([sequence[i : i + k] for i in range(len(sequence) - k + 1)])


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
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

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        kmer: int = -1,
    ):
        super(SupervisedDataset, self).__init__()
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [label, sequence]
            logging.warning("Perform single sequence classification...")
            texts = [d[1].strip() for d in data]
            # Auto-detect unique labels and create mapping
            unique_labels = sorted(set(d[0].strip() for d in data))
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
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

def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
   
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    acc = accuracy_score(valid_labels, valid_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels, valid_predictions, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
    return metrics


def extended_compute_metrics(eval_pred):
   
    predictions, labels = eval_pred

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    pred_class = np.argmax(predictions, axis=1)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, pred_class, labels=[0, 1, 2])
    
    alt_five_to_alt_three_confusion = cm[0, 1] if cm.shape == (3, 3) else 0
    alt_three_to_alt_five_confusion = cm[1, 0] if cm.shape == (3, 3) else 0
    total_confusion = alt_five_to_alt_three_confusion + alt_three_to_alt_five_confusion
    total_samples = len(labels)
    
    confusion_rate = total_confusion / total_samples if total_samples > 0 else 0.0
    confusion_penalty = confusion_rate * 0.25
    
    class_report = classification_report(labels, pred_class, output_dict=True, zero_division=0)
    class_0_f1 = class_report.get("0", {}).get("f1-score", 0.0) if "0" in class_report else 0.0
    class_1_f1 = class_report.get("1", {}).get("f1-score", 0.0) if "1" in class_report else 0.0
    class_2_f1 = class_report.get("2", {}).get("f1-score", 0.0) if "2" in class_report else 0.0
    
    if 0 in class_report:
        class_0_f1 = class_report[0].get("f1-score", class_0_f1)
    if 1 in class_report:
        class_1_f1 = class_report[1].get("f1-score", class_1_f1)
    if 2 in class_report:
        class_2_f1 = class_report[2].get("f1-score", class_2_f1)
    
    # Class mapping: class 0=alt_five, class 1=alt_three, class 2=cassette (alphabetical order)
    alt_five_f1 = class_0_f1
    alt_three_f1 = class_1_f1
    cassette_f1 = class_2_f1
    
    weighted_f1 = class_report.get("weighted avg", {}).get("f1-score", 0.0)
    macro_f1 = class_report.get("macro avg", {}).get("f1-score", 0.0)
    
    base_score = 0.40 * alt_five_f1 + 0.35 * alt_three_f1 + 0.12 * weighted_f1 + 0.05 * macro_f1 + 0.03 * cassette_f1
    combined_score = base_score - confusion_penalty
    
    return {
        "eval_class_2_f1": alt_five_f1,
        "eval_class_0_f1": cassette_f1,
        "eval_class_1_f1": alt_three_f1,
        "eval_class_0_raw": class_0_f1,
        "eval_class_1_raw": class_1_f1,
        "eval_class_2_raw": class_2_f1,
        "accuracy": class_report.get("accuracy", 0.0),
        "eval_weighted_f1": weighted_f1,
        "eval_macro_f1": macro_f1,
        "eval_combined_score": combined_score,
        "eval_confusion_penalty": confusion_penalty,
        "eval_alt_five_alt_three_confusion": total_confusion
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
    return logits.detach().cpu()

def plot_confusion_matrix(trainer, eval_dataset, results_dir, runname_label, class_names=None):
    results_dir = Path(results_dir)
    
    predictions, labels, _ = trainer.predict(eval_dataset)
    preds = np.argmax(predictions, axis=-1)
    cm = confusion_matrix(labels, preds)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(labels)))]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {runname_label}")
    
    plt.savefig(results_dir / f"confusion_matrix_{runname_label}.png")
    plt.close()


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
       
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=self.alpha, reduction='none')

    def forward(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Compute the inverse frequency of each class
def compute_class_weights(labels: List[int], num_classes: int, focus_class: Optional[int] = None) -> torch.Tensor:
    
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    
    base_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
    
    if focus_class is not None and focus_class < num_classes:
        focus_multiplier = 1.8
        base_weights[focus_class] = base_weights[focus_class] * focus_multiplier
        logger.info(f"Giving extra weight to class {focus_class}: {base_weights[focus_class]:.4f} (multiplier: {focus_multiplier}x)")
    
    weights = torch.tensor(base_weights, dtype=torch.float32)
    logger.info(f"Class weights: {dict(zip(range(num_classes), weights.tolist()))}")
    return weights

class CustomTrainer(transformers.Trainer):
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

    if training_args.fp16 and not torch.cuda.is_available():
        logging.warning("FP16 requested but CUDA not available. Disabling FP16.")
        training_args.fp16 = False
    elif not training_args.fp16 and torch.cuda.is_available():
        logging.info("CUDA available. Enabling FP16 for faster training.")
        training_args.fp16 = True

    dataset_name = os.path.basename(os.path.normpath(data_args.data_path))
    base_results_root = Path(os.environ.get("RESULTS_DIR", "./result_4"))
    default_results_dir = base_results_root / f"DB2_{dataset_name}"
    results_dir = default_results_dir

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

    results_dir.mkdir(parents=True, exist_ok=True)

    training_args.output_dir = str(results_dir / "model_output")
    training_args.logging_dir = str(results_dir / "logs")
    
    if not model_args.use_wandb:
        training_args.report_to = []
        run_name = "no_wandb_run"
        logging.info("WandB disabled. Training logs will not be sent to WandB.")
    else:
        training_args.report_to = ["wandb"]
        run_name = f"DB22_{dataset_name}"
        try:
            wandb.init(entity='sahilthapa35077-university-of-colorado', project="alt3_cass", name=run_name)
            wandb.config.update({**vars(model_args), **vars(data_args), **vars(training_args)})
            logging.info("WandB initialized successfully.")
        except Exception as e:
            logging.warning(f"Failed to initialize WandB: {e}. Continuing without WandB.")
            training_args.report_to = []
            model_args.use_wandb = False

    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)

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

    train_data_path = os.path.join(
        data_args.data_path, 
        "train_oversampled.csv" if model_args.apply_adasyn else "train.csv"
    )

    logging.warning(f"Loading dataset: {train_data_path}")

    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=train_data_path,
        kmer=data_args.kmer,
    )

    val_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "dev.csv"),
        kmer=data_args.kmer,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    def build_and_train_model(learning_rate, weight_decay, num_train_epochs, batch_size, alpha=None, gamma=2.0, trial_number=None):
        os.environ["USE_FLASH_ATTENTION"] = "0"
        os.environ["DISABLE_FLASH_ATTENTION"] = "1"
        os.environ["USE_FLASH_ATTENTION_2"] = "0"
        os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"
        os.environ["MAX_JOBS"] = "1"
        
        if trial_number is not None:
            trial_output_dir = str(results_dir / f"trial_{trial_number}_model_output")
            trial_logging_dir = str(results_dir / f"trial_{trial_number}_logs")
        else:
            trial_output_dir = str(results_dir / "model_output")
            trial_logging_dir = str(results_dir / "logs")
        
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                trust_remote_code=True,
            )
            if hasattr(config, 'num_labels'):
                config.num_labels = train_dataset.num_labels
            if hasattr(config, 'use_flash_attention_2'):
                config.use_flash_attention_2 = False
            if hasattr(config, 'use_flash_attention'):
                config.use_flash_attention = False
            if hasattr(config, '_flash_attn_2_enabled'):
                config._flash_attn_2_enabled = False
            for attr in ['_attn_implementation', 'attn_implementation']:
                if hasattr(config, attr):
                    setattr(config, attr, 'eager')
        except Exception as e:
            logging.warning(f"Could not modify config: {e}. Proceeding with default config.")
            config = None
        
        model_kwargs = {
            "cache_dir": training_args.cache_dir,
            "trust_remote_code": True,
        }
        if config is None:
            model_kwargs["num_labels"] = train_dataset.num_labels
        else:
            model_kwargs["config"] = config
            
        def patch_flash_attn_files():
            """Find and patch all flash_attn_triton.py files in HuggingFace cache"""
            cache_dir = training_args.cache_dir or os.path.expanduser("~/.cache/huggingface")
            hub_path = os.path.join(cache_dir, "hub")
            
            if not os.path.exists(hub_path):
                return
            
            for root, dirs, files in os.walk(hub_path):
                if "flash_attn_triton.py" in files:
                    flash_file = os.path.join(root, "flash_attn_triton.py")
                    try:
                        stub_content = '''# PATCHED: Flash Attention disabled
def _flash_attn_forward(*args, **kwargs):
    raise RuntimeError("Flash Attention is disabled. The model should use standard attention instead.")
__all__ = ['_flash_attn_forward']
'''
                        with open(flash_file, 'w') as f:
                            f.write(stub_content)
                        logging.info(f"Patched {flash_file} to disable Flash Attention")
                    except Exception as e:
                        logging.warning(f"Could not patch {flash_file}: {e}")
        
        patch_flash_attn_files()
        
        try:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                **model_kwargs
            )
            patch_flash_attn_files()
        except Exception as e:
            error_str = str(e).lower()
            if "flash" in error_str or "triton" in error_str or "compilation" in error_str:
                logging.warning(f"Model loading failed due to Flash Attention/Triton: {e}")
                logging.info("Patching model files and retrying...")
                patch_flash_attn_files()
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
        
        if hasattr(model.config, 'use_flash_attention_2'):
            model.config.use_flash_attention_2 = False
        if hasattr(model.config, 'use_flash_attention'):
            model.config.use_flash_attention = False
        if hasattr(model.config, '_flash_attn_2_enabled'):
            model.config._flash_attn_2_enabled = False
        
        def patch_model_attention(model):
            """Recursively patch all attention layers to disable Flash Attention"""
            for name, module in model.named_modules():
                if hasattr(module, '_flash_attn_enabled'):
                    module._flash_attn_enabled = False
                if hasattr(module, 'use_flash_attention_2'):
                    module.use_flash_attention_2 = False
                if hasattr(module, 'use_flash_attention'):
                    module.use_flash_attention = False
                if hasattr(module, '_flash_attention_forward'):
                    original_forward = module.forward if hasattr(module, 'forward') else None
                    if original_forward:
                        def patched_forward(*args, **kwargs):
                            kwargs.pop('use_flash_attention_2', None)
                            kwargs.pop('use_flash_attention', None)
                            return original_forward(*args, **kwargs)
                        module.forward = patched_forward
        
        try:
            patch_model_attention(model)
            logging.info("Successfully patched model to disable Flash Attention")
        except Exception as e:
            logging.warning(f"Could not patch model attention: {e}. Continuing anyway.")
        
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
        
        if hasattr(model, '__dict__'):
            model.__dict__['flash_attn'] = create_noop_flash_attn()

        if model_args.use_lora:
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
                    target_modules=target_modules,
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
            if alpha is None:
                class_weights = compute_class_weights(train_dataset.labels, train_dataset.num_labels, focus_class=None)
                if train_dataset.num_labels == 3:
                    class_weights[0] = class_weights[0] * 1.7
                    class_weights[1] = class_weights[1] * 1.3
                    class_weights[2] = class_weights[2] * 0.85
                    logger.info(f"Applied weights - alt_five (class 0): {class_weights[0].item():.3f}, alt_three (class 1): {class_weights[1].item():.3f}, cassette (class 2): {class_weights[2].item():.3f}")
            else:
                class_weights = alpha

            class_weights = class_weights.to("cuda" if torch.cuda.is_available() else "cpu")
            loss_fn = FocalLoss(alpha=class_weights, gamma=gamma, reduction='mean')
        
        local_training_args = copy.deepcopy(training_args)
        local_training_args.learning_rate = learning_rate
        local_training_args.weight_decay = weight_decay
        local_training_args.num_train_epochs = num_train_epochs
        local_training_args.per_device_train_batch_size = batch_size
        
        if trial_number is not None:
            local_training_args.output_dir = trial_output_dir
            local_training_args.logging_dir = trial_logging_dir
            local_training_args.overwrite_output_dir = True
            logger.info(f"Trial {trial_number}: Using unique output directory: {trial_output_dir}")

        last_checkpoint = None
        if trial_number is None:
            if (
                os.path.isdir(local_training_args.output_dir)
                and not local_training_args.overwrite_output_dir
            ):
                last_checkpoint = get_last_checkpoint(local_training_args.output_dir)
                if last_checkpoint is not None:
                    logger.info(f"Resuming from checkpoint: {last_checkpoint}")
                
        if model_args.use_class_weights:
            trainer = CustomTrainer(
                model=model,
                tokenizer=tokenizer,
                args=local_training_args,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                compute_metrics=extended_compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                loss_fn=loss_fn,
                callbacks=[transformers.EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.001
                )]
            )
        else:
            trainer = transformers.Trainer(
                model=model,
                tokenizer=tokenizer,
                args=local_training_args,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                compute_metrics=extended_compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                callbacks=[transformers.EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.001
                )]
            )
            
        try:
            trainer.train(resume_from_checkpoint=last_checkpoint)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise e

        if local_training_args.save_model:
            if model_args.use_lora:
                trainer.model.save_pretrained(local_training_args.output_dir)
            else:
                trainer.save_state()
                safe_save_model_for_hf_trainer(trainer=trainer, output_dir=local_training_args.output_dir)

        if local_training_args.eval_and_save_results:
            results_path = os.path.dirname(local_training_args.output_dir)
            results = trainer.evaluate(eval_dataset=val_dataset)
            os.makedirs(results_path, exist_ok=True)
            
            if trial_number is not None:
                eval_file = os.path.join(results_path, f"{dataset_name}_trial_{trial_number}_eval_results.json")
            else:
                eval_file = os.path.join(results_path, f"{dataset_name}_eval_results.json")
            
            with open(eval_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved trial {trial_number} eval results to {eval_file}")
            if trial_number is not None:
                logger.info(f"Trial {trial_number} metrics: weighted_f1={results.get('eval_weighted_f1', 0):.4f}, accuracy={results.get('eval_accuracy', 0):.4f}")
            
            best_checkpoint_info = None
            if hasattr(trainer.state, 'best_metric') and trainer.state.best_metric is not None:
                metric_name = getattr(trainer.args, 'metric_for_best_model', 'eval_combined_score')
                best_checkpoint_info = {
                    'best_metric': trainer.state.best_metric,
                    'best_model_checkpoint': trainer.state.best_model_checkpoint,
                    'best_metric_name': metric_name,
                }
                if trial_number is not None:
                    best_info_file = os.path.join(results_path, f"{dataset_name}_trial_{trial_number}_best_checkpoint.json")
                else:
                    best_info_file = os.path.join(results_path, f"{dataset_name}_best_checkpoint.json")
                with open(best_info_file, "w") as f:
                    json.dump(best_checkpoint_info, f, indent=2)
                logger.info(f"Best checkpoint info saved: {best_checkpoint_info}")
                
            class_names = getattr(train_dataset, 'class_names', None)
            plot_confusion_matrix(trainer, val_dataset, results_path, run_name, class_names=class_names)
        else:
            results = None
            
        return trainer, results


    if model_args.use_optuna:
        best_model_dir = str(results_dir / "best_model")

        def objective(trial):
            print(f"Starting trial {trial.number}")

            learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            num_train_epochs = trial.suggest_int("num_train_epochs", 25, 40)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            gamma = trial.suggest_float("gamma", 3.5, 5.5)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            alpha = torch.tensor([
                trial.suggest_float("alpha_class_0", 1.8, 2.8),
                trial.suggest_float("alpha_class_1", 1.8, 2.5),
                trial.suggest_float("alpha_class_2", 0.5, 1.0)
            ], dtype=torch.float32).to(device)

            trainer, build_results = build_and_train_model(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                num_train_epochs=num_train_epochs,
                batch_size=batch_size,
                alpha=alpha,
                gamma=gamma,
                trial_number=trial.number
            )

            best_checkpoint_path = None
            if hasattr(trainer.state, 'best_model_checkpoint') and trainer.state.best_model_checkpoint:
                best_checkpoint_path = trainer.state.best_model_checkpoint
                logger.info(f"Trial {trial.number}: Best checkpoint is {best_checkpoint_path}")
                if hasattr(trainer.state, 'best_metric') and trainer.state.best_metric is not None:
                    metric_name = getattr(trainer.args, 'metric_for_best_model', 'eval_combined_score')
                    logger.info(f"Trial {trial.number}: Best metric ({metric_name}) = {trainer.state.best_metric:.4f}")
            else:
                logger.warning(f"Trial {trial.number}: No best checkpoint found in trainer state. Using final model.")

            metrics = trainer.evaluate(val_dataset)
            
            logger.info("=" * 80)
            logger.info(f"TRIAL {trial.number} COMPLETED - DETAILED METRICS:")
            logger.info("=" * 80)
            logger.info(f"  Weighted F1: {metrics.get('eval_weighted_f1', 0):.4f}")
            logger.info(f"  Macro F1: {metrics.get('eval_macro_f1', 0):.4f}")
            logger.info(f"  Accuracy: {metrics.get('eval_accuracy', 0):.4f}")
            logger.info(f"  Alt_five F1 (class 2): {metrics.get('eval_class_2_f1', 0):.4f}")
            logger.info(f"  Alt_three F1 (class 1): {metrics.get('eval_class_1_f1', 0):.4f}")
            logger.info(f"  Cassette F1 (class 0): {metrics.get('eval_class_0_f1', 0):.4f}")
            logger.info(f"  Loss: {metrics.get('eval_loss', 0):.4f}")
            if hasattr(trainer.state, 'best_metric') and trainer.state.best_metric is not None:
                logger.info(f"  Best metric value: {trainer.state.best_metric:.4f}")
                logger.info(f"  Best checkpoint: {trainer.state.best_model_checkpoint}")
            logger.info("=" * 80)

            eval_class_2_f1 = metrics.get("eval_class_2_f1", 0)
            eval_class_0_f1 = metrics.get("eval_class_0_f1", 0)
            eval_class_1_f1 = metrics.get("eval_class_1_f1", 0)
            eval_weighted_f1 = metrics.get("eval_weighted_f1", 0)
            eval_macro_f1 = metrics.get("eval_macro_f1", 0)
            eval_combined_score = metrics.get("eval_combined_score", 0)
            
            if eval_combined_score is None or eval_combined_score == 0:
                raise ValueError(f"Metrics are missing eval_combined_score: {metrics}")

            current_score = eval_combined_score
            
            logger.info(f"Trial {trial.number} - Weighted F1: {current_score:.4f}, alt_five F1: {eval_class_2_f1:.4f}, alt_three F1: {eval_class_1_f1:.4f}, cassette F1: {eval_class_0_f1:.4f}")
            is_best = False
            try:
                n_completed = len([t for t in trial.study.trials if t.state.name == 'COMPLETE'])
                if n_completed == 0:
                    is_best = True
                else:
                    try:
                        best_value = trial.study.best_value
                        is_best = current_score > best_value
                    except (ValueError, AttributeError):
                        is_best = True
            except (ValueError, AttributeError, Exception):
                is_best = True
            
            if is_best:
                metric_name = "eval_weighted_f1"
                logger.info(f"New best trial found: {trial.number} with {metric_name}: {current_score:.4f}")
                
                best_checkpoint_info = {
                    'trial_number': trial.number,
                    'best_metric': current_score,
                    'best_metric_name': metric_name,
                    'best_checkpoint_path': best_checkpoint_path if best_checkpoint_path else trainer.state.best_model_checkpoint,
                    'accuracy': metrics.get('eval_accuracy', 0),
                    'weighted_f1': metrics.get('eval_weighted_f1', 0),
                    'macro_f1': metrics.get('eval_macro_f1', 0),
                    'alt_five_f1': metrics.get('eval_class_2_f1', 0),
                    'alt_three_f1': metrics.get('eval_class_1_f1', 0),
                    'cassette_f1': metrics.get('eval_class_0_f1', 0),
                }
                
                best_info_file = os.path.join(best_model_dir, "best_checkpoint_info.json")
                os.makedirs(best_model_dir, exist_ok=True)
                with open(best_info_file, "w") as f:
                    json.dump(best_checkpoint_info, f, indent=2)
                logger.info(f"Saved best checkpoint info to {best_info_file}")
                
                if best_checkpoint_path:
                    if trainer.state.best_model_checkpoint and best_checkpoint_path != trainer.state.best_model_checkpoint:
                        logger.warning(f"WARNING: best_checkpoint_path ({best_checkpoint_path}) != trainer.state.best_model_checkpoint ({trainer.state.best_model_checkpoint})")
                    else:
                        logger.info(f"Verified: trainer.model is from best checkpoint: {best_checkpoint_path}")
                elif trainer.state.best_model_checkpoint:
                    logger.info(f"Using best checkpoint from trainer.state: {trainer.state.best_model_checkpoint}")
                else:
                    logger.warning("WARNING: No best checkpoint found! Saving final epoch model instead of best checkpoint.")
                
                if model_args.use_lora:
                    trainer.model.save_pretrained(best_model_dir)
                else:
                    trainer.save_state()
                    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=best_model_dir)
                
                best_eval_file = os.path.join(best_model_dir, "best_trial_eval_results.json")
                with open(best_eval_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Saved best trial eval results to {best_eval_file}")
                
                dataset_name_for_file = os.path.basename(os.path.normpath(data_args.data_path))
                final_eval_file = os.path.join(results_dir, f"{dataset_name_for_file}_eval_results.json")
                os.makedirs(results_dir, exist_ok=True)
                with open(final_eval_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Saved final eval_results.json (matching best_model) to {final_eval_file}")

            return current_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=training_args.optuna_trials)
        logger.info(f"Optuna finished. Best params: {study.best_params}")
        
        logger.info("=" * 80)
        logger.info("OPTUNA TRIALS SUMMARY")
        logger.info("=" * 80)
        trials_summary = []
        for trial in study.trials:
            if trial.state.name == 'COMPLETE':
                trial_info = {
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                trials_summary.append(trial_info)
                logger.info(f"Trial {trial.number}: Score={trial.value:.4f}, Params={trial.params}")
        
        summary_file = results_dir / "optuna_trials_summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                'best_trial_number': study.best_trial.number,
                'best_value': study.best_value,
                'best_params': study.best_params,
                'total_trials': len(study.trials),
                'completed_trials': len([t for t in study.trials if t.state.name == 'COMPLETE']),
                'all_trials': trials_summary
            }, f, indent=2)
        logger.info(f"Saved Optuna trials summary to {summary_file}")
        logger.info("=" * 80)
        
        if model_args.use_wandb:
            wandb.log({"best_optuna_params": study.best_params})

    else:
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
