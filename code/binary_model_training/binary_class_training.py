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
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from imblearn.over_sampling import ADASYN
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
    data_path: str = field(default='../data_preprocessing/balanced_binary_datasets/cassette_vs_alt_three', metadata={"help": "Path to the training data."})
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
    metric_for_best_model: str = field(default="eval_weighted_f1")
    greater_is_better: bool = field(default=True)
    save_total_limit: int = field(default=1)
    seed: int = field(default=42)
    optuna_trials: int = field(default=10, metadata={"help": "Number of trials for Optuna HPO"})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Path to checkpoint to resume training from (for continuing from a previous trial)"})
    start_trial_number: int = field(default=0, metadata={"help": "Starting trial number for Optuna (for continuing from a previous trial)"})


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
        # apply_adasyn: bool = False,
    ):
        super(SupervisedDataset, self).__init__()
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            logging.warning("Perform single sequence classification...")
            texts = [d[1].strip() for d in data]
            unique_labels = sorted(set(d[0].strip() for d in data))
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            self.label_mapping = label_mapping
            self.class_names = unique_labels
            labels = [label_mapping[d[0].strip()] for d in data] 
        elif len(data[0]) == 3:
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[1], d[1]] for d in data]
            labels = [int(d[0]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            if isinstance(texts[0], list):
                texts = [
                    " ".join([load_or_generate_kmer(data_path, [t], kmer)[0] for t in pair]) 
                    for pair in texts
                ]
            else:
                texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        else:
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


def extended_compute_metrics(eval_pred, class_names=None):
    """
    Extended compute_metrics for binary classification.
    For alt_three_vs_alt_five: uses combined score with emphasis on alt_five.
    For other datasets: uses weighted F1.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        class_names: List of class names in order [class_0_name, class_1_name]
    """
    predictions, labels = eval_pred

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    pred_class = np.argmax(predictions, axis=1)
    class_report = classification_report(labels, pred_class, output_dict=True, zero_division=0)

    class_0_f1 = 0.0
    class_1_f1 = 0.0
    
    if "0" in class_report:
        class_0_f1 = class_report["0"].get("f1-score", 0.0)
    elif 0 in class_report:
        class_0_f1 = class_report[0].get("f1-score", 0.0)
    
    if "1" in class_report:
        class_1_f1 = class_report["1"].get("f1-score", 0.0)
    elif 1 in class_report:
        class_1_f1 = class_report[1].get("f1-score", 0.0)
    
    weighted_f1 = class_report.get("weighted avg", {}).get("f1-score", 0.0)
    macro_f1 = class_report.get("macro avg", {}).get("f1-score", 0.0)
    accuracy = class_report.get("accuracy", 0.0)
    
    alt_five_f1 = None
    if class_names:
        if 'alt_five' in class_names:
            alt_five_idx = class_names.index('alt_five')
            alt_five_f1 = class_0_f1 if alt_five_idx == 0 else class_1_f1
        else:
            alt_five_f1 = None
    
    if alt_five_f1 is not None:
        alt_three_f1 = class_1_f1
        combined_score = 0.40 * alt_five_f1 + 0.45 * alt_three_f1 + 0.15 * weighted_f1
    else:
        combined_score = 0.40 * class_0_f1 + 0.45 * class_1_f1 + 0.15 * weighted_f1
    
    return {
        "eval_class_0_f1": class_0_f1,
        "eval_class_1_f1": class_1_f1,
        "accuracy": accuracy,
        "eval_weighted_f1": weighted_f1,
        "eval_macro_f1": macro_f1,
        "eval_combined_score": combined_score
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
    
    # Use provided class names or default
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(labels)))]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {runname_label}")
    
    plt.savefig(results_dir / f"confusion_matrix_{runname_label}.png")
    plt.close()


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
    global os
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Check CUDA availability and verify GPU access
    if not torch.cuda.is_available():
        logging.error("=" * 80)
        logging.error("CRITICAL: CUDA is not available! Training will be very slow on CPU.")
        logging.error("=" * 80)
        logging.error(f"PyTorch version: {torch.__version__}")
        logging.error(f"CUDA compiled: {torch.version.cuda}")
        logging.error(f"CUDA available: {torch.cuda.is_available()}")
        logging.error(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        logging.error(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}")
        import subprocess
        try:
            nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if nvidia_smi.returncode == 0:
                logging.error("nvidia-smi works, but PyTorch cannot access GPU. This is a configuration issue.")
                logging.error("Possible causes:")
                logging.error("  1. PyTorch CUDA version mismatch with host CUDA")
                logging.error("  2. CUDA_VISIBLE_DEVICES environment variable issue")
                logging.error("  3. Missing CUDA libraries in container")
            else:
                logging.error("nvidia-smi failed - no GPU access at all")
        except Exception as e:
            logging.error(f"Could not run nvidia-smi: {e}")
        logging.error("=" * 80)
        raise RuntimeError("CUDA is not available. Cannot train on GPU. Please check Docker GPU configuration.")
    
    # Log GPU information
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"CUDA devices: {torch.cuda.device_count()}")
    logging.info(f"Current device: {torch.cuda.current_device()}")
    logging.info(f"Device name: {torch.cuda.get_device_name(0)}")

    if training_args.fp16 and not torch.cuda.is_available():
        logging.warning("FP16 requested but CUDA not available. Disabling FP16.")
        training_args.fp16 = False
    elif not training_args.fp16 and torch.cuda.is_available():
        logging.info("CUDA available. Enabling FP16 for faster training.")
        training_args.fp16 = True

    dataset_name = os.path.basename(os.path.normpath(data_args.data_path))
    
    if "alt_three_vs_alt_five" in dataset_name:
        training_args.metric_for_best_model = "eval_combined_score"
        logger.info("Using eval_combined_score (40% alt_five + 45% alt_three + 15% weighted F1) for alt_three_vs_alt_five")
    else:
        training_args.metric_for_best_model = "eval_weighted_f1"
        logger.info(f"Using eval_weighted_f1 for {dataset_name}")

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
        training_args.report_to = []  # Empty list = no reporting
        run_name = "no_wandb_run"
        logging.info("WandB disabled. Training logs will not be sent to WandB.")
    else:
        training_args.report_to = ["wandb"]
        run_name = f"DB22_{dataset_name}"
        try:
            wandb.init(entity='sahilthapa35077-university-of-', project="alt3_cass", name=run_name)
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

    # Load dataset
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

    def build_and_train_model(learning_rate, weight_decay, num_train_epochs, batch_size, alpha=None, gamma=2.0, trial_number=None, resume_from_checkpoint=None):
        """Helper function for normal or Optuna-based training loops.
        
        Args:
            trial_number: If provided, creates a unique output directory for this trial to avoid checkpoint conflicts.
            resume_from_checkpoint: If provided, loads model from this checkpoint instead of base model.
        """
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
                        stub_content = '''def _flash_attn_forward(*args, **kwargs):
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
            if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
                logger.info(f"Loading model from checkpoint: {resume_from_checkpoint}")
                model = transformers.AutoModelForSequenceClassification.from_pretrained(
                    resume_from_checkpoint,
                    **model_kwargs
                )
            else:
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
                    if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
                        logger.info(f"Loading model from checkpoint (retry): {resume_from_checkpoint}")
                        model = transformers.AutoModelForSequenceClassification.from_pretrained(
                            resume_from_checkpoint,
                            **model_kwargs
                        )
                    else:
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
            class NoOpFlashAttn:
                def __getattr__(self, name):
                    raise RuntimeError("Flash Attention is disabled. The model should use standard attention.")
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
                class_weights = compute_class_weights(train_dataset.labels, train_dataset.num_labels)
                logger.info(f"Computed class weights (balanced dataset): {class_weights}")
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
                
        class_names = getattr(train_dataset, 'class_names', None)
                
        if model_args.use_class_weights:
            trainer = CustomTrainer(
                model=model,
                tokenizer=tokenizer,
                args=local_training_args,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                compute_metrics=lambda eval_pred: extended_compute_metrics(eval_pred, class_names=class_names),
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
                compute_metrics=lambda eval_pred: extended_compute_metrics(eval_pred, class_names=class_names),
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
                json.dump(results, f)
            
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
            logger.info(f"Starting trial {trial.number}")
            import os as os_module
            dataset_name_for_metric = os_module.path.basename(os_module.path.normpath(data_args.data_path))

            if "constitutive_vs_alt_three" in dataset_name_for_metric:
                learning_rate = trial.suggest_float("learning_rate", 5e-5, 2e-4, log=True)
            elif "constitutive_vs_alt_five" in dataset_name_for_metric:
                learning_rate = trial.suggest_float("learning_rate", 3e-5, 1.5e-4, log=True)
            elif "alt_three_vs_alt_five" in dataset_name_for_metric:
                learning_rate = trial.suggest_float("learning_rate", 5e-5, 1.2e-4, log=True)
            else:
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
            if "constitutive_vs_alt_five" in dataset_name_for_metric:
                weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)
            elif "alt_three_vs_alt_five" in dataset_name_for_metric:
                weight_decay = trial.suggest_float("weight_decay", 5e-5, 1e-2, log=True)
            else:
                weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            if "constitutive_vs_alt_three" in dataset_name_for_metric:
                num_train_epochs = trial.suggest_int("num_train_epochs", 30, 50)
            elif "constitutive_vs_alt_five" in dataset_name_for_metric:
                num_train_epochs = trial.suggest_int("num_train_epochs", 35, 55)
            elif "alt_three_vs_alt_five" in dataset_name_for_metric:
                num_train_epochs = trial.suggest_int("num_train_epochs", 40, 60)
            else:
                num_train_epochs = trial.suggest_int("num_train_epochs", 25, 40)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if "constitutive_vs_alt_three" in dataset_name_for_metric:
                gamma = trial.suggest_float("gamma", 3.0, 5.0)
                alpha = torch.tensor([
                    trial.suggest_float("alpha_class_0", 1.3, 2.2),
                    trial.suggest_float("alpha_class_1", 0.8, 1.4)
                ], dtype=torch.float32).to(device)
            elif "constitutive_vs_alt_five" in dataset_name_for_metric:
                gamma = trial.suggest_float("gamma", 3.5, 6.0)
                alpha = torch.tensor([
                    trial.suggest_float("alpha_class_0", 1.0, 2.0),
                    trial.suggest_float("alpha_class_1", 1.0, 2.0)
                ], dtype=torch.float32).to(device)
            elif "alt_three_vs_alt_five" in dataset_name_for_metric:
                gamma = trial.suggest_float("gamma", 4.0, 6.5)
                alpha = torch.tensor([
                    trial.suggest_float("alpha_class_0", 1.5, 2.5),
                    trial.suggest_float("alpha_class_1", 1.8, 2.5)
                ], dtype=torch.float32).to(device)
            else:
                gamma = trial.suggest_float("gamma", 2.0, 4.0)
                alpha = torch.tensor([
                    trial.suggest_float("alpha_class_0", 0.7, 1.3),
                    trial.suggest_float("alpha_class_1", 0.7, 1.3)
                ], dtype=torch.float32).to(device)

            trial_num = getattr(trial, '_number', trial.number)
            trainer, build_results = build_and_train_model(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                num_train_epochs=num_train_epochs,
                batch_size=batch_size,
                alpha=alpha,
                gamma=gamma,
                trial_number=trial_num,
                resume_from_checkpoint=training_args.resume_from_checkpoint
            )
            best_checkpoint_path = None
            if hasattr(trainer.state, 'best_model_checkpoint') and trainer.state.best_model_checkpoint:
                best_checkpoint_path = trainer.state.best_model_checkpoint
                logger.info(f"Trial {trial_num}: Best checkpoint is {best_checkpoint_path}")
                metric_name = getattr(trainer.args, 'metric_for_best_model', 'eval_combined_score')
                if hasattr(trainer.state, 'best_metric'):
                    logger.info(f"Trial {trial_num}: Best metric ({metric_name}) = {trainer.state.best_metric:.4f}")
            else:
                logger.warning(f"Trial {trial_num}: No best checkpoint found in trainer state. Using final model.")

            metrics = trainer.evaluate(val_dataset)
            logger.info(f"Computed metrics for trial {trial_num}: {metrics}")

            eval_class_0_f1 = metrics.get("eval_class_0_f1", 0)
            eval_class_1_f1 = metrics.get("eval_class_1_f1", 0)
            eval_weighted_f1 = metrics.get("eval_weighted_f1", 0)
            eval_combined_score = metrics.get("eval_combined_score", 0)
            
            if eval_weighted_f1 is None:
                raise ValueError(f"Metrics are missing required keys: {metrics}")

            if "alt_three_vs_alt_five" in dataset_name_for_metric:
                alt_five_f1 = eval_class_0_f1
                current_score = eval_combined_score
                logger.info(f"Trial {trial_num} - alt_five F1 (class 0): {alt_five_f1:.4f}, Combined score: {current_score:.4f}")
            else:
                current_score = eval_weighted_f1
                logger.info(f"Trial {trial_num} - Weighted F1: {current_score:.4f}")
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
                metric_name = "eval_combined_score" if "alt_three_vs_alt_five" in dataset_name_for_metric else "eval_weighted_f1"
                logger.info(f"New best trial found: {trial_num} with {metric_name}: {current_score:.4f}")
                
                best_checkpoint_info = {
                    'trial_number': trial_num,
                    'best_metric': current_score,
                    'best_metric_name': metric_name,
                    'best_checkpoint_path': best_checkpoint_path if best_checkpoint_path else trainer.state.best_model_checkpoint,
                    'accuracy': metrics.get('eval_accuracy', 0),
                    'weighted_f1': metrics.get('eval_weighted_f1', 0),
                    'class_0_f1': metrics.get('eval_class_0_f1', 0),
                    'class_1_f1': metrics.get('eval_class_1_f1', 0),
                }
                
                best_info_file = os_module.path.join(best_model_dir, "best_checkpoint_info.json")
                os_module.makedirs(best_model_dir, exist_ok=True)
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
                
                best_eval_file = os_module.path.join(best_model_dir, "best_trial_eval_results.json")
                with open(best_eval_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Saved best trial eval results to {best_eval_file}")
                
                final_eval_file = os_module.path.join(results_dir, f"{dataset_name_for_metric}_eval_results.json")
                os_module.makedirs(results_dir, exist_ok=True)
                with open(final_eval_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Saved final eval_results.json (matching best_model) to {final_eval_file}")

            return current_score

        study = optuna.create_study(direction="maximize")
        start_trial_offset = training_args.start_trial_number
        
        def make_objective_with_offset(original_objective, offset):
            def wrapped_objective(trial):
                class TrialWrapper:
                    def __init__(self, trial, offset):
                        self._trial = trial
                        self._offset = offset
                    
                    def __getattr__(self, name):
                        if name == 'number':
                            return self._trial.number + self._offset
                        return getattr(self._trial, name)
                
                wrapped_trial = TrialWrapper(trial, offset)
                return original_objective(wrapped_trial)
            return wrapped_objective
        
        if training_args.start_trial_number > 0:
            logger.info(f"Continuing from previous training. Trials will be numbered {training_args.start_trial_number} to {training_args.start_trial_number + training_args.optuna_trials - 1}")
            objective_with_offset = make_objective_with_offset(objective, start_trial_offset)
        else:
            objective_with_offset = objective
        
        study.optimize(objective_with_offset, n_trials=training_args.optuna_trials)
        logger.info(f"Optuna finished. Best params: {study.best_params}")
        if model_args.use_wandb:
            wandb.log({"best_optuna_params": study.best_params})

    else:
        trainer, _ = build_and_train_model(
            learning_rate=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            num_train_epochs=training_args.num_train_epochs,
            batch_size=training_args.per_device_train_batch_size,
        )
    
    if model_args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
