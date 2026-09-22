# -*- coding: utf-8 -*-
# Resumable experiment framework with the high-performing dual-view concept.


import os
import re
import gc
import json
import pickle
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
)

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except Exception:
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from statsmodels.stats.contingency_tables import mcnemar
    MCNEMAR_AVAILABLE = True
except ImportError:
    MCNEMAR_AVAILABLE = False


# ============================================================
# GLOBAL CONFIG
# ============================================================
SEED = 42

DATA_PATH = "./explicit_implicit_dataset.csv"
OUTPUT_DIR = Path("./experiment_results_v3")
BASELINE_RESULT_DIR = OUTPUT_DIR / "baselines"
PROPOSED_DIR = OUTPUT_DIR / "proposed_hybrid"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

for p in [OUTPUT_DIR, BASELINE_RESULT_DIR, PROPOSED_DIR, CHECKPOINT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

LABEL2ID = {
    "Non-Hate": 0,
    "Implicit": 1,
    "Explicit": 2,
}
ID2LABEL = {
    0: "Non-Hate",
    1: "Implicit",
    2: "Explicit",
}
CLASS_NAMES = ["Non-Hate", "Implicit", "Explicit"]
NUM_LABELS = 3

# ---------------- Baseline setup ----------------
BASELINE_MODELS = {
    "DistilBERT": "distilbert-base-uncased",
    "MiniLM": "microsoft/MiniLM-L12-H384-uncased",
    "MobileBERT": "google/mobilebert-uncased",
    "ModernBERT": "answerdotai/ModernBERT-base",
}

BASELINE_MAX_LEN = 192
BASELINE_BATCH_SIZE = 8
BASELINE_EPOCHS = 6
BASELINE_LR = 2e-5
BASELINE_WEIGHT_DECAY = 0.01
BASELINE_PATIENCE = 2

# ---------------- Proposed setup ----------------
PROPOSED_MODEL_NAME = "distilbert-base-uncased"
PROPOSED_MAX_LEN = 128
PROPOSED_BATCH_SIZE = 8
PROPOSED_EPOCHS = 40
PROPOSED_LR = 2e-5
PROPOSED_WEIGHT_DECAY = 0.01
PROPOSED_PATIENCE = 7
PROPOSED_GRAD_ACC = 2
USE_MIXED_PRECISION = True

TTA_STEPS = 5

# Resume / rerun behavior
SKIP_COMPLETED_BASELINES = True
FORCE_RETRAIN_PROPOSED = False
RESUME_BASELINES = True
RESUME_PROPOSED = True
EVALUATE_ONLY_PROPOSED = False

# Set False if you only want baselines
RUN_PROPOSED = True

# Set False if runtime is tight
# Recommended False during the training PBS job. After training finishes,
# set EVALUATE_ONLY_PROPOSED=True and RUN_TTA=True in a shorter evaluation job.
RUN_TTA = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def atomic_torch_save(state, path):
    """Write a checkpoint atomically so a PBS kill cannot corrupt it."""
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

print("=" * 70)
print("HATE SPEECH CLASSIFICATION: COMPACT BASELINES + PROPOSED MODEL")
print("=" * 70)
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 70)


# ============================================================
# DATA
# ============================================================
def create_sample_data():
    texts = [
        "I respect all people equally regardless of their background",
        "They are just not built for this kind of work, if you know what I mean",
        "You are disgusting and should go back to where you came from",
        "Some people just do not understand our culture",
        "Everyone deserves equal opportunities",
        "These people are so lazy and stupid",
        "I love my country and its diversity",
        "Foreigners are ruining our society",
        "We should help those in need regardless of origin",
        "Go back to your own country",
        "Maybe those people should stay where they belong",
        "Obviously they are all criminals, right?",
        "I strongly disagree with their politics but respect their rights",
        "That group is clearly inferior to ours",
        "Some of them are alright, but most are not",
    ]

    labels = [
        "Non-Hate", "Implicit", "Explicit", "Implicit", "Non-Hate",
        "Implicit", "Non-Hate", "Implicit", "Non-Hate", "Explicit",
        "Implicit", "Implicit", "Non-Hate", "Explicit", "Implicit",
    ]

    return pd.DataFrame({"text": texts, "label": labels})


def load_dataset(path):
    try:
        df = pd.read_csv(path)
        print(f"\nLoaded dataset: {path}")
    except Exception as e:
        print(f"\nCould not load dataset: {e}")
        print("Using tiny sample dataset only for code testing.")
        df = create_sample_data()

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip()

    invalid = sorted(set(df["label"]) - set(LABEL2ID))
    if invalid:
        raise ValueError(
            f"Unexpected labels: {invalid}. "
            f"Expected exactly: {list(LABEL2ID.keys())}"
        )

    df["label_id"] = df["label"].map(LABEL2ID).astype(int)

    print("\nDataset statistics")
    print("------------------")
    print("Total samples:", len(df))
    print(df["label"].value_counts())

    return df


df = load_dataset(DATA_PATH)

X = df["text"].values
y = df["label_id"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=SEED,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp,
)

print("\nData split")
print("----------")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def safe_model_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_pickle(obj, path):
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_metrics(labels, preds):
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    macro_f1 = f1_score(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(labels, preds)

    precision, recall, f1_per_class, support = (
        precision_recall_fscore_support(
            labels,
            preds,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
    )

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))

    return {
        "test_f1": float(macro_f1),
        "test_accuracy": float(accuracy),
        "test_precision_macro": macro_precision,
        "test_recall_macro": macro_recall,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "support": support.tolist(),
        "predictions": preds.tolist(),
        "labels": labels.tolist(),
    }


def print_metrics(name, result):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Macro F1 : {result['test_f1']:.4f}")
    print(f"Accuracy : {result['test_accuracy']:.4f}")
    print(f"Precision: {result['test_precision_macro']:.4f}")
    print(f"Recall   : {result['test_recall_macro']:.4f}")

    f1c = result["f1_per_class"]

    print(
        f"Class F1 -> "
        f"Non-Hate {f1c[0]:.4f} | "
        f"Implicit {f1c[1]:.4f} | "
        f"Explicit {f1c[2]:.4f}"
    )

    print(
        classification_report(
            result["labels"],
            result["predictions"],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )


# ============================================================
# BASELINE DATASET
# ============================================================
class TransformerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in enc.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long,
        )

        return item


def make_transformer_loader(
    X_data,
    y_data,
    tokenizer,
    max_len,
    batch_size,
    train=False,
):
    ds = TransformerDataset(
        X_data,
        y_data,
        tokenizer,
        max_len,
    )

    if train:
        counts = np.bincount(
            np.asarray(y_data),
            minlength=NUM_LABELS,
        ).astype(np.float64)

        class_weights = 1.0 / (counts + 1e-6)
        sample_weights = class_weights[np.asarray(y_data)]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# BASELINE EVALUATION
# ============================================================
@torch.no_grad()
def evaluate_baseline(model, loader):
    model.eval()

    preds = []
    labels = []

    for batch in loader:
        batch = {
            k: v.to(DEVICE)
            for k, v in batch.items()
        }

        labs = batch["labels"]

        outputs = model(**batch)
        logits = outputs.logits

        batch_preds = logits.argmax(dim=-1)

        preds.extend(
            batch_preds.detach().cpu().numpy().tolist()
        )
        labels.extend(
            labs.detach().cpu().numpy().tolist()
        )

    return compute_metrics(labels, preds)


# ============================================================
# BASELINE TRAINING
# ============================================================
def train_one_baseline(
    display_name,
    hf_name,
    epochs=BASELINE_EPOCHS,
):
    print("\n" + "#" * 70)
    print(f"TRAINING BASELINE: {display_name}")
    print(f"Checkpoint: {hf_name}")
    print("#" * 70)

    result_path = (
        BASELINE_RESULT_DIR
        / f"{safe_model_name(display_name)}_results.pkl"
    )

    ckpt_path = (
        CHECKPOINT_DIR
        / f"{safe_model_name(display_name)}_best.pt"
    )

    last_ckpt_path = (
        CHECKPOINT_DIR
        / f"{safe_model_name(display_name)}_last.pt"
    )

    if SKIP_COMPLETED_BASELINES and result_path.exists():
        print(
            f"Existing result found for {display_name}. "
            "Skipping training."
        )
        return load_pickle(result_path)

    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_name,
        use_fast=True,
    )

    train_loader = make_transformer_loader(
        X_train,
        y_train,
        tokenizer,
        BASELINE_MAX_LEN,
        BASELINE_BATCH_SIZE,
        train=True,
    )

    val_loader = make_transformer_loader(
        X_val,
        y_val,
        tokenizer,
        BASELINE_MAX_LEN,
        BASELINE_BATCH_SIZE,
        train=False,
    )

    test_loader = make_transformer_loader(
        X_test,
        y_test,
        tokenizer,
        BASELINE_MAX_LEN,
        BASELINE_BATCH_SIZE,
        train=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    n_params = count_trainable_parameters(model)

    print(
        f"Trainable parameters: "
        f"{n_params / 1e6:.2f} M"
    )

    optimizer = AdamW(
        model.parameters(),
        lr=BASELINE_LR,
        weight_decay=BASELINE_WEIGHT_DECAY,
    )

    total_steps = max(
        1,
        len(train_loader) * epochs,
    )

    warmup_steps = int(
        0.10 * total_steps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    start_epoch = 0
    best_val_f1 = -1.0
    patience_counter = 0
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_f1": [],
        "val_accuracy": [],
    }

    if RESUME_BASELINES and last_ckpt_path.exists():
        print(f"Resuming {display_name} from {last_ckpt_path}")
        resume_ckpt = torch.load(
            last_ckpt_path,
            map_location=DEVICE,
            weights_only=False,
        )
        model.load_state_dict(
            resume_ckpt["model_state_dict"],
            strict=True,
        )
        optimizer.load_state_dict(
            resume_ckpt["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            resume_ckpt["scheduler_state_dict"]
        )
        start_epoch = int(resume_ckpt["epoch"]) + 1
        best_val_f1 = float(
            resume_ckpt.get("best_val_f1", -1.0)
        )
        patience_counter = int(
            resume_ckpt.get("patience_counter", 0)
        )
        saved_history = resume_ckpt.get("history", {})
        for key in history:
            history[key] = list(saved_history.get(key, []))
        print(
            f"Completed epochs: {start_epoch}/{epochs} | "
            f"Best Val F1: {best_val_f1:.4f}"
        )

    for epoch in range(start_epoch, epochs):
        print(
            f"\n{display_name} | "
            f"Epoch {epoch + 1}/{epochs}"
        )

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        valid_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = {
                k: v.to(DEVICE)
                for k, v in batch.items()
            }

            labels = batch["labels"]

            optimizer.zero_grad(set_to_none=True)

            try:
                outputs = model(**batch)

                loss = outputs.loss

                if (
                    torch.isnan(loss)
                    or torch.isinf(loss)
                ):
                    print(
                        f"Invalid loss at batch "
                        f"{batch_idx}; skipping."
                    )
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()

                logits = outputs.logits
                preds = logits.argmax(dim=-1)

                correct += (
                    preds == labels
                ).sum().item()

                total += labels.size(0)
                running_loss += loss.item()
                valid_batches += 1

                if (
                    batch_idx > 0
                    and batch_idx % 50 == 0
                ):
                    print(
                        f"Batch "
                        f"{batch_idx}/{len(train_loader)} | "
                        f"Loss {loss.item():.4f} | "
                        f"Acc "
                        f"{correct / max(total, 1):.4f}"
                    )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"OOM at batch {batch_idx}; "
                        "batch skipped."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    cleanup_cuda()
                    continue
                raise

        train_loss = (
            running_loss
            / max(valid_batches, 1)
        )

        train_acc = (
            correct
            / max(total, 1)
        )

        val_result = evaluate_baseline(
            model,
            val_loader,
        )

        history["train_loss"].append(
            float(train_loss)
        )
        history["train_accuracy"].append(
            float(train_acc)
        )
        history["val_f1"].append(
            float(val_result["test_f1"])
        )
        history["val_accuracy"].append(
            float(val_result["test_accuracy"])
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val F1 "
            f"{val_result['test_f1']:.4f} | "
            f"Val Acc "
            f"{val_result['test_accuracy']:.4f}"
        )

        if val_result["test_f1"] > best_val_f1:
            best_val_f1 = val_result["test_f1"]
            patience_counter = 0

            atomic_torch_save(
                {
                    "epoch": epoch,
                    "best_val_f1": best_val_f1,
                    "hf_name": hf_name,
                    "display_name": display_name,
                    "model_state_dict": {
                        k: v.detach().cpu()
                        for k, v
                        in model.state_dict().items()
                    },
                },
                ckpt_path,
            )

            print(
                f"Saved best {display_name} "
                f"checkpoint. "
                f"Val F1={best_val_f1:.4f}"
            )

        else:
            patience_counter += 1

        atomic_torch_save(
            {
                "epoch": epoch,
                "best_val_f1": float(best_val_f1),
                "patience_counter": int(patience_counter),
                "history": history,
                "hf_name": hf_name,
                "display_name": display_name,
                "model_state_dict": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            last_ckpt_path,
        )
        print(f"Saved resume checkpoint after epoch {epoch + 1}.")

        if patience_counter >= BASELINE_PATIENCE:
            print(
                f"Early stopping {display_name} "
                f"at epoch {epoch + 1}."
            )
            break

    if ckpt_path.exists():
        ckpt = torch.load(
            ckpt_path,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            ckpt["model_state_dict"],
            strict=True,
        )

    test_result = evaluate_baseline(
        model,
        test_loader,
    )

    test_result["model"] = display_name
    test_result["checkpoint"] = hf_name
    test_result["parameters"] = int(n_params)
    test_result["parameters_million"] = float(
        n_params / 1e6
    )
    test_result["best_val_f1"] = float(
        best_val_f1
    )
    test_result["history"] = history

    save_pickle(
        test_result,
        result_path,
    )

    print_metrics(
        display_name,
        test_result,
    )

    del model
    del tokenizer
    del train_loader
    del val_loader
    del test_loader

    cleanup_cuda()

    return test_result


# ============================================================
# LINGUISTIC FEATURES FOR PROPOSED MODEL
# ============================================================
HEDGE_WORDS = {
    "maybe",
    "perhaps",
    "some",
    "those",
    "they",
    "them",
    "certain",
    "supposedly",
    "apparently",
    "allegedly",
    "just",
    "only",
    "simply",
}

IRONY_CUES = {
    "obviously",
    "clearly",
    "of course",
    "sure",
    "right",
    "totally",
    "definitely",
    "sarcasm",
    "jk",
    "just kidding",
}

EXPLICIT_SLURS_PROXY = {
    "hate",
    "kill",
    "die",
    "trash",
    "vermin",
    "scum",
    "filthy",
    "disgusting",
    "animal",
    "subhuman",
    "worthless",
}

IDENTITY_TERMS = {
    "immigrant",
    "foreigner",
    "refugee",
    "muslim",
    "jew",
    "christian",
    "black",
    "white",
    "asian",
    "latino",
    "gay",
    "lesbian",
    "trans",
    "woman",
    "man",
    "disabled",
}

analyzer = (
    SentimentIntensityAnalyzer()
    if VADER_AVAILABLE
    else None
)


def extract_enhanced_features(text):
    text = str(text)
    lower = text.lower()

    tokens = re.findall(
        r"\b\w+\b",
        lower,
    )

    n_tokens = len(tokens)

    # Sentiment
    if VADER_AVAILABLE and analyzer is not None:
        s = analyzer.polarity_scores(text)

        pos_s = s["pos"]
        neg_s = s["neg"]
        neu_s = s["neu"]
        compound = s["compound"]

    else:
        pos_s = 0.0
        neg_s = 0.0
        neu_s = 0.0
        compound = 0.0

    # Readability
    if TEXTSTAT_AVAILABLE:
        try:
            flesch = textstat.flesch_reading_ease(
                text
            )
            fog = textstat.gunning_fog(
                text
            )
            coleman = textstat.coleman_liau_index(
                text
            )
        except Exception:
            flesch = 0.0
            fog = 0.0
            coleman = 0.0
    else:
        flesch = 0.0
        fog = 0.0
        coleman = 0.0

    # Lexical
    avg_word_len = (
        np.mean([len(w) for w in tokens])
        if tokens
        else 0.0
    )

    vocab_richness = (
        len(set(tokens))
        / (n_tokens + 1)
    )

    upper_ratio = (
        sum(1 for c in text if c.isupper())
        / (len(text) + 1)
    )

    punct_ratio = (
        sum(
            1
            for c in text
            if c in "!?.,;:"
        )
        / (len(text) + 1)
    )

    exclam = text.count("!")
    question = text.count("?")

    # Lexicon
    hedge_count = (
        sum(
            1
            for w in tokens
            if w in HEDGE_WORDS
        )
        / (n_tokens + 1)
    )

    irony_count = (
        sum(
            1
            for w in tokens
            if w in IRONY_CUES
        )
        / (n_tokens + 1)
    )

    slur_count = (
        sum(
            1
            for w in tokens
            if w in EXPLICIT_SLURS_PROXY
        )
        / (n_tokens + 1)
    )

    identity_count = (
        sum(
            1
            for w in tokens
            if w in IDENTITY_TERMS
        )
        / (n_tokens + 1)
    )

    identity_hate_cooc = float(
        identity_count > 0
        and slur_count > 0
    )

    # POS
    if SPACY_AVAILABLE and nlp is not None:
        try:
            doc = nlp(text[:1000])
            tags = [t.pos_ for t in doc]
            n = len(tags) + 1

            noun_r = (
                sum(
                    1
                    for t in tags
                    if t in ("NOUN", "PROPN")
                )
                / n
            )

            verb_r = (
                sum(
                    1
                    for t in tags
                    if t == "VERB"
                )
                / n
            )

            adj_r = (
                sum(
                    1
                    for t in tags
                    if t == "ADJ"
                )
                / n
            )

            adv_r = (
                sum(
                    1
                    for t in tags
                    if t == "ADV"
                )
                / n
            )

        except Exception:
            noun_r = 0.0
            verb_r = 0.0
            adj_r = 0.0
            adv_r = 0.0
    else:
        noun_r = 0.0
        verb_r = 0.0
        adj_r = 0.0
        adv_r = 0.0

    # Subjectivity
    if TEXTBLOB_AVAILABLE:
        try:
            subjectivity = (
                TextBlob(text)
                .sentiment
                .subjectivity
            )
        except Exception:
            subjectivity = 0.5
    else:
        subjectivity = 0.5

    feats = np.array(
        [
            pos_s,
            neg_s,
            neu_s,
            compound,
            flesch / 100.0,
            fog / 20.0,
            coleman / 20.0,
            n_tokens / 200.0,
            avg_word_len / 10.0,
            vocab_richness,
            exclam / 10.0,
            question / 10.0,
            upper_ratio,
            punct_ratio,
            noun_r,
            verb_r,
            adj_r,
            adv_r,
            subjectivity,
        ],
        dtype=np.float32,
    )

    feats = np.clip(
        feats,
        -10.0,
        10.0,
    )

    feats = np.nan_to_num(
        feats,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )

    return feats


FEATURE_DIM = 19


# ============================================================
# PROPOSED DATASET
# ============================================================
class ProposedDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_len,
        augment=False,
    ):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

        print(
            f"Extracting linguistic features "
            f"for {len(self.texts)} samples..."
        )

        self.features = [
            extract_enhanced_features(t)
            for t in self.texts
        ]

        print(
            "Feature extraction complete. "
            f"Dim={FEATURE_DIM}"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        feats = self.features[idx].copy()

        if (
            self.augment
            and random.random() < 0.30
        ):
            feats += np.random.normal(
                loc=0.0,
                scale=0.01,
                size=feats.shape,
            ).astype(np.float32)

        item = {
            "input_ids":
                enc["input_ids"].squeeze(0),

            "attention_mask":
                enc["attention_mask"].squeeze(0),

            "features":
                torch.tensor(
                    feats,
                    dtype=torch.float32,
                ),

            "labels":
                torch.tensor(
                    self.labels[idx],
                    dtype=torch.long,
                ),
        }

        return item


def make_proposed_loader(
    X_data,
    y_data,
    tokenizer,
    train=False,
):
    ds = ProposedDataset(
        X_data,
        y_data,
        tokenizer,
        PROPOSED_MAX_LEN,
        augment=train,
    )

    if train:
        counts = np.bincount(
            np.asarray(y_data),
            minlength=NUM_LABELS,
        ).astype(np.float64)

        class_weights = (
            1.0
            / (counts + 1e-6)
        )

        sample_weights = (
            class_weights[
                np.asarray(y_data)
            ]
        )

        sampler = WeightedRandomSampler(
            sample_weights,
            len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            ds,
            batch_size=PROPOSED_BATCH_SIZE,
            sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    return DataLoader(
        ds,
        batch_size=PROPOSED_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# PROPOSED MODEL
# ============================================================
class DualViewModel(nn.Module):
    def __init__(
        self,
        feature_dim=FEATURE_DIM,
        dropout_rate=0.30,
        model_name=PROPOSED_MODEL_NAME,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # Mask-aware token attention pooling.
        self.self_attention = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.Tanh(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Feature-conditioned gate: 256 text + 128 linguistic features.
        self.cross_attention = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
        )

        self.fusion_norm = nn.LayerNorm(384)

        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, NUM_LABELS),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(384, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, NUM_LABELS),
            nn.Sigmoid(),
        )

        self.aux_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, NUM_LABELS),
        )

        # Initialize only new heads. Pretrained DistilBERT is untouched.
        custom_modules = [
            self.feature_net,
            self.self_attention,
            self.text_projection,
            self.cross_attention,
            self.fusion_norm,
            self.classifier,
            self.uncertainty_head,
            self.aux_classifier,
        ]
        for module in custom_modules:
            module.apply(self._init_custom_weights)

    @staticmethod
    def _init_custom_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids,
        attention_mask,
        features,
        return_aux=False,
    ):
        feature_dtype = next(self.feature_net.parameters()).dtype
        features = features.to(dtype=feature_dtype)

        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        head_dtype = next(self.self_attention.parameters()).dtype
        seq = out.last_hidden_state.to(dtype=head_dtype)

        scores = self.self_attention(seq).squeeze(-1)
        scores = scores.masked_fill(
            attention_mask == 0,
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(seq * weights.unsqueeze(-1), dim=1)

        text_representation = self.text_projection(pooled)
        feature_representation = self.feature_net(features)
        gate_input = torch.cat(
            [text_representation, feature_representation], dim=1
        )
        text_gate = self.cross_attention(gate_input)
        gated_text = text_representation * (0.5 + text_gate)
        fused = self.fusion_norm(
            torch.cat([gated_text, feature_representation], dim=1)
        )

        logits = self.classifier(fused)
        uncertainty = self.uncertainty_head(fused)

        if return_aux:
            aux_logits = self.aux_classifier(text_representation)
            return logits, uncertainty, weights, aux_logits

        return logits, uncertainty, weights


# ============================================================
# PROPOSED LOSS
# ============================================================
class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        gamma=1.5,
        num_classes=NUM_LABELS,
        label_smoothing=0.05,
        ce_weight=0.70,
        focal_weight=0.30,
        aux_weight=0.20,
    ):
        super().__init__()

        self.gamma = gamma
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.aux_weight = aux_weight

    def forward(
        self,
        logits,
        targets,
        epoch_progress=0.5,
        aux_logits=None,
    ):
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probs = F.softmax(logits.float(), dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.clamp(1e-6, 1.0 - 1e-6)
        focal = ((1.0 - pt).pow(self.gamma) * ce).mean()
        loss = self.ce_weight * ce.mean() + self.focal_weight * focal

        if aux_logits is not None:
            aux = F.cross_entropy(
                aux_logits,
                targets,
                label_smoothing=self.label_smoothing,
            )
            current_aux_weight = self.aux_weight * max(
                0.25, 1.0 - float(epoch_progress)
            )
            loss = loss + current_aux_weight * aux

        return loss


# ============================================================
# LLRD OPTIMIZER
# ============================================================
def build_llrd_optimizer(
    model,
    base_lr=PROPOSED_LR,
    decay=0.9,
    weight_decay=PROPOSED_WEIGHT_DECAY,
    head_mult=5.0,
):
    del decay, head_mult
    return AdamW(
        [
            {"params": model.bert.parameters(), "lr": base_lr},
            {"params": model.feature_net.parameters(), "lr": base_lr * 2.0},
            {"params": model.self_attention.parameters(), "lr": base_lr * 2.0},
            {"params": model.text_projection.parameters(), "lr": base_lr * 1.5},
            {"params": model.cross_attention.parameters(), "lr": base_lr * 1.5},
            {"params": model.fusion_norm.parameters(), "lr": base_lr * 1.5},
            {"params": model.classifier.parameters(), "lr": base_lr * 2.0},
            {"params": model.uncertainty_head.parameters(), "lr": base_lr * 1.5},
            {"params": model.aux_classifier.parameters(), "lr": base_lr * 2.0},
        ],
        weight_decay=weight_decay,
    )


# ============================================================
# PROPOSED EVALUATION
# ============================================================
@torch.no_grad()
def evaluate_proposed(
    model,
    loader,
):
    model.eval()

    preds = []
    labels = []
    uncertainties = []

    for batch in loader:
        ids = (
            batch["input_ids"]
            .to(DEVICE)
        )

        mask = (
            batch["attention_mask"]
            .to(DEVICE)
        )

        feats = (
            batch["features"]
            .to(DEVICE)
        )

        labs = (
            batch["labels"]
            .to(DEVICE)
        )

        logits, unc, _ = model(
            ids,
            mask,
            feats,
        )

        p = logits.argmax(
            dim=1
        )

        preds.extend(
            p.cpu().numpy().tolist()
        )

        labels.extend(
            labs.cpu().numpy().tolist()
        )

        uncertainties.extend(
            unc.mean(dim=1)
            .cpu()
            .numpy()
            .tolist()
        )

    result = compute_metrics(
        labels,
        preds,
    )

    result["uncertainties"] = (
        uncertainties
    )

    return result


@torch.no_grad()
def evaluate_proposed_tta(
    model,
    loader,
    n_steps=TTA_STEPS,
):
    """
    Monte-Carlo dropout style TTA.
    FIXED: labels are collected for the complete test set.
    """
    model.train()

    all_probs = []
    true_labels = None

    for step in range(n_steps):
        probs_this_pass = []
        labels_this_pass = []

        for batch in loader:
            ids = (
                batch["input_ids"]
                .to(DEVICE)
            )

            mask = (
                batch["attention_mask"]
                .to(DEVICE)
            )

            feats = (
                batch["features"]
                .to(DEVICE)
            )

            logits, _, _ = model(
                ids,
                mask,
                feats,
            )

            probs = F.softmax(
                logits,
                dim=-1,
            ).cpu()

            probs_this_pass.append(
                probs
            )

            labels_this_pass.extend(
                batch["labels"]
                .cpu()
                .numpy()
                .tolist()
            )

        all_probs.append(
            torch.cat(
                probs_this_pass,
                dim=0,
            )
        )

        if true_labels is None:
            true_labels = np.asarray(
                labels_this_pass
            )

        print(
            f"TTA pass "
            f"{step + 1}/{n_steps} complete."
        )

    avg_probs = torch.stack(
        all_probs,
        dim=0,
    ).mean(dim=0)

    preds = (
        avg_probs
        .argmax(dim=-1)
        .numpy()
    )

    model.eval()

    result = compute_metrics(
        true_labels,
        preds,
    )

    result["probs"] = (
        avg_probs.numpy()
    )

    return result


# ============================================================
# PROPOSED TRAINING
# ============================================================
def train_proposed_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    epochs,
    checkpoint_path,
):
    last_checkpoint_path = checkpoint_path.with_name(
        "hybrid_proposed_last.pt"
    )
    start_epoch = 0
    best_f1 = -1.0
    patience_counter = 0

    history = defaultdict(
        list
    )
    amp_enabled = USE_MIXED_PRECISION and torch.cuda.is_available()
    scaler = GradScaler(enabled=amp_enabled)

    if (
        RESUME_PROPOSED
        and last_checkpoint_path.exists()
        and not FORCE_RETRAIN_PROPOSED
    ):
        print(
            f"\nResuming proposed model from "
            f"{last_checkpoint_path}"
        )
        resume_ckpt = torch.load(
            last_checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
        model.load_state_dict(
            resume_ckpt["model_state_dict"],
            strict=True,
        )
        optimizer.load_state_dict(
            resume_ckpt["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            resume_ckpt["scheduler_state_dict"]
        )
        if "scaler_state_dict" in resume_ckpt:
            scaler.load_state_dict(
                resume_ckpt["scaler_state_dict"]
            )
        start_epoch = int(resume_ckpt["epoch"]) + 1
        best_f1 = float(resume_ckpt.get("best_f1", -1.0))
        patience_counter = int(
            resume_ckpt.get("patience_counter", 0)
        )
        for key, values in resume_ckpt.get("history", {}).items():
            history[key] = list(values)
        print(
            f"Completed epochs: {start_epoch}/{epochs} | "
            f"Best Val F1: {best_f1:.4f}"
        )

    for epoch in range(start_epoch, epochs):
        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Proposed model | "
            f"Epoch {epoch + 1}/{epochs}"
        )

        print(
            "=" * 70
        )

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0
        valid_batches = 0

        progress = (
            epoch
            / max(
                1,
                epochs - 1,
            )
        )

        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(
            train_loader
        ):
            ids = (
                batch["input_ids"]
                .to(DEVICE)
            )

            mask = (
                batch["attention_mask"]
                .to(DEVICE)
            )

            feats = (
                batch["features"]
                .to(DEVICE)
            )

            labs = (
                batch["labels"]
                .to(DEVICE)
            )

            try:
                with autocast(enabled=amp_enabled):
                    logits, uncertainty, _, aux_logits = model(
                        ids,
                        mask,
                        feats,
                        return_aux=True,
                    )

                    raw_loss = loss_fn(
                        logits,
                        labs,
                        progress,
                        aux_logits,
                    )
                    raw_loss = raw_loss + 0.01 * uncertainty.mean()
                    loss = raw_loss / PROPOSED_GRAD_ACC

                if (
                    torch.isnan(loss)
                    or torch.isinf(loss)
                ):
                    print(
                        f"Invalid loss "
                        f"at batch {i}; "
                        "skipping."
                    )
                    continue

                scaler.scale(loss).backward()

                should_step = (
                    (i + 1) % PROPOSED_GRAD_ACC == 0
                    or (i + 1) == len(train_loader)
                )

                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                preds = logits.argmax(
                    dim=-1
                )

                correct += (
                    preds == labs
                ).sum().item()

                total += labs.size(0)
                total_loss += raw_loss.item()
                valid_batches += 1

                if (
                    i > 0
                    and i % 20 == 0
                ):
                    print(
                        f"Batch "
                        f"{i}/{len(train_loader)} | "
                        f"Loss {raw_loss.item():.4f} | "
                        f"Acc "
                        f"{correct / max(total, 1):.4f}"
                    )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"OOM at batch {i}; "
                        "skipping."
                    )
                    optimizer.zero_grad(
                        set_to_none=True
                    )
                    gc.collect()
                    cleanup_cuda()
                    continue
                raise

        avg_loss = (
            total_loss
            / max(
                valid_batches,
                1,
            )
        )

        avg_acc = (
            correct
            / max(
                total,
                1,
            )
        )

        val = evaluate_proposed(
            model,
            val_loader,
        )

        history[
            "train_loss"
        ].append(
            float(avg_loss)
        )

        history[
            "train_acc"
        ].append(
            float(avg_acc)
        )

        history[
            "val_f1"
        ].append(
            float(
                val["test_f1"]
            )
        )

        history[
            "val_acc"
        ].append(
            float(
                val["test_accuracy"]
            )
        )

        print(
            f"\nEpoch {epoch + 1} | "
            f"Train Loss "
            f"{avg_loss:.4f} | "
            f"Train Acc "
            f"{avg_acc:.4f} | "
            f"Val F1 "
            f"{val['test_f1']:.4f} | "
            f"Val Acc "
            f"{val['test_accuracy']:.4f}"
        )

        f1c = val[
            "f1_per_class"
        ]

        print(
            f"Val class F1 -> "
            f"Non-Hate "
            f"{f1c[0]:.4f} | "
            f"Implicit "
            f"{f1c[1]:.4f} | "
            f"Explicit "
            f"{f1c[2]:.4f}"
        )

        if val["test_f1"] > best_f1:
            best_f1 = (
                val["test_f1"]
            )

            patience_counter = 0

            atomic_torch_save(
                {
                    "model_state_dict":
                        {
                            k:
                            v.detach().cpu()
                            for k, v
                            in model
                            .state_dict()
                            .items()
                        },
                    "epoch": epoch,
                    "best_f1":
                        best_f1,
                    "feature_dim":
                        FEATURE_DIM,
                    "model_name":
                        PROPOSED_MODEL_NAME,
                    "max_len":
                        PROPOSED_MAX_LEN,
                },
                checkpoint_path,
            )

            print(
                f"Saved best proposed "
                f"checkpoint "
                f"(Val F1="
                f"{best_f1:.4f})"
            )

        else:
            patience_counter += 1

        atomic_torch_save(
            {
                "epoch": epoch,
                "best_f1": float(best_f1),
                "patience_counter": int(patience_counter),
                "history": dict(history),
                "feature_dim": FEATURE_DIM,
                "model_name": PROPOSED_MODEL_NAME,
                "max_len": PROPOSED_MAX_LEN,
                "model_state_dict": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            },
            last_checkpoint_path,
        )
        print(f"Saved resume checkpoint after epoch {epoch + 1}.")

        if patience_counter >= PROPOSED_PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1}."
            )
            break

    return (
        best_f1,
        dict(history),
    )


# ============================================================
# RUN PROPOSED MODEL
# ============================================================
def run_proposed():
    result_path = (
        PROPOSED_DIR
        / "proposed_results.pkl"
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / "hybrid_proposed_best.pt"
    )

    if (
        result_path.exists()
        and not FORCE_RETRAIN_PROPOSED
    ):
        print(
            "\nExisting proposed result "
            "found. Skipping retraining."
        )
        return load_pickle(
            result_path
        )

    tokenizer = AutoTokenizer.from_pretrained(
        PROPOSED_MODEL_NAME,
        use_fast=True,
    )

    print(
        "\nBuilding proposed "
        "train loader..."
    )

    train_loader = (
        make_proposed_loader(
            X_train,
            y_train,
            tokenizer,
            train=True,
        )
    )

    print(
        "\nBuilding proposed "
        "validation loader..."
    )

    val_loader = (
        make_proposed_loader(
            X_val,
            y_val,
            tokenizer,
            train=False,
        )
    )

    print(
        "\nBuilding proposed "
        "test loader..."
    )

    test_loader = (
        make_proposed_loader(
            X_test,
            y_test,
            tokenizer,
            train=False,
        )
    )

    model = DualViewModel(
        feature_dim=FEATURE_DIM
    ).to(DEVICE)

    n_params = count_trainable_parameters(
        model
    )

    print(
        f"\nProposed trainable "
        f"parameters: "
        f"{n_params / 1e6:.2f} M"
    )

    history = {}
    best_f1 = -1.0

    if EVALUATE_ONLY_PROPOSED:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"EVALUATE_ONLY_PROPOSED=True, but the best "
                f"checkpoint does not exist: {checkpoint_path}"
            )
        print(
            "\nExisting proposed checkpoint "
            "found. Evaluation-only mode: "
            "loading it and skipping training."
        )

        ckpt = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            ckpt["model_state_dict"],
            strict=False,
        )

        best_f1 = float(
            ckpt.get(
                "best_f1",
                -1.0,
            )
        )

    else:
        loss_fn = (
            ClassBalancedFocalLoss()
            .to(DEVICE)
        )

        optimizer = (
            build_llrd_optimizer(
                model,
                base_lr=PROPOSED_LR,
                weight_decay=
                    PROPOSED_WEIGHT_DECAY,
            )
        )

        total_steps = max(
            1,
            int(np.ceil(len(train_loader) / PROPOSED_GRAD_ACC))
            * PROPOSED_EPOCHS,
        )

        warmup_steps = int(
            0.10 * total_steps
        )

        scheduler = (
            get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=
                    warmup_steps,
                num_training_steps=
                    total_steps,
            )
        )

        best_f1, history = (
            train_proposed_model(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                loss_fn,
                PROPOSED_EPOCHS,
                checkpoint_path,
            )
        )

        if checkpoint_path.exists():
            ckpt = torch.load(
                checkpoint_path,
                map_location=DEVICE,
                weights_only=False,
            )

            model.load_state_dict(
                ckpt[
                    "model_state_dict"
                ],
                strict=False,
            )

            best_f1 = float(
                ckpt.get("best_f1", best_f1)
            )

    print(
        "\nEvaluating proposed model "
        "without TTA..."
    )

    plain = evaluate_proposed(
        model,
        test_loader,
    )

    plain["model"] = "Proposed"
    plain["checkpoint"] = (
        PROPOSED_MODEL_NAME
    )
    plain["parameters"] = int(
        n_params
    )
    plain[
        "parameters_million"
    ] = float(
        n_params / 1e6
    )
    plain["best_val_f1"] = float(
        best_f1
    )
    plain["history"] = history

    print_metrics(
        "Proposed - No TTA",
        plain,
    )

    final_result = dict(
        plain
    )

    if RUN_TTA:
        print(
            "\nEvaluating proposed "
            f"with TTA x{TTA_STEPS}..."
        )

        tta = evaluate_proposed_tta(
            model,
            test_loader,
            n_steps=TTA_STEPS,
        )

        tta["model"] = (
            "Proposed"
        )

        tta["checkpoint"] = (
            PROPOSED_MODEL_NAME
        )

        tta["parameters"] = int(
            n_params
        )

        tta[
            "parameters_million"
        ] = float(
            n_params / 1e6
        )

        tta[
            "best_val_f1"
        ] = float(
            best_f1
        )

        tta["history"] = history

        print_metrics(
            "Proposed - TTA",
            tta,
        )

        final_result = dict(
            tta
        )

        final_result[
            "plain_predictions"
        ] = plain[
            "predictions"
        ]

        final_result[
            "plain_test_f1"
        ] = plain[
            "test_f1"
        ]

        final_result[
            "plain_test_accuracy"
        ] = plain[
            "test_accuracy"
        ]

    save_pickle(
        final_result,
        result_path,
    )

    del model
    del tokenizer
    del train_loader
    del val_loader
    del test_loader

    cleanup_cuda()

    return final_result


# ============================================================
# McNEMAR
# ============================================================
def mcnemar_test(
    preds_a,
    preds_b,
    labels,
):
    if not MCNEMAR_AVAILABLE:
        return {
            "statistic": None,
            "p_value": None,
            "significant": None,
        }

    a_ok = (
        np.asarray(preds_a)
        == np.asarray(labels)
    )

    b_ok = (
        np.asarray(preds_b)
        == np.asarray(labels)
    )

    table = [
        [
            int(
                np.sum(
                    a_ok & b_ok
                )
            ),
            int(
                np.sum(
                    a_ok & ~b_ok
                )
            ),
        ],
        [
            int(
                np.sum(
                    ~a_ok & b_ok
                )
            ),
            int(
                np.sum(
                    ~a_ok & ~b_ok
                )
            ),
        ],
    ]

    result = mcnemar(
        table,
        exact=False,
        correction=True,
    )

    return {
        "statistic":
            float(
                result.statistic
            ),
        "p_value":
            float(
                result.pvalue
            ),
        "significant":
            bool(
                result.pvalue
                < 0.05
            ),
        "table":
            table,
    }


# ============================================================
# COMPARISON TABLE
# ============================================================
def build_comparison_table(results):
    rows = []

    for model_name, r in results.items():
        if r is None:
            continue

        f1c = r.get(
            "f1_per_class",
            [np.nan] * 3,
        )

        rows.append(
            {
                "Model":
                    model_name,

                "Parameters_M":
                    r.get(
                        "parameters_million",
                        np.nan,
                    ),

                "Accuracy":
                    r.get(
                        "test_accuracy",
                        np.nan,
                    ),

                "Macro_F1":
                    r.get(
                        "test_f1",
                        np.nan,
                    ),

                "Macro_Precision":
                    r.get(
                        "test_precision_macro",
                        np.nan,
                    ),

                "Macro_Recall":
                    r.get(
                        "test_recall_macro",
                        np.nan,
                    ),

                "Non_Hate_F1":
                    f1c[0]
                    if len(f1c) > 0
                    else np.nan,

                "Implicit_F1":
                    f1c[1]
                    if len(f1c) > 1
                    else np.nan,

                "Explicit_F1":
                    f1c[2]
                    if len(f1c) > 2
                    else np.nan,
            }
        )

    table = pd.DataFrame(
        rows
    )

    if not table.empty:
        table = table.sort_values(
            by="Macro_F1",
            ascending=False,
        ).reset_index(
            drop=True
        )

    return table


# ============================================================
# IMPROVEMENT ANALYSIS
# ============================================================
def improvement_analysis(results):
    print(
        "\n"
        + "=" * 70
    )
    print(
        "IMPROVEMENT ANALYSIS"
    )
    print(
        "=" * 70
    )

    proposed = results.get(
        "Proposed"
    )

    if proposed is None:
        print(
            "Proposed model result "
            "not available."
        )
        return None

    baseline_scores = []

    for name, result in results.items():
        if name == "Proposed":
            continue

        if result is None:
            continue

        score = result.get(
            "test_f1"
        )

        if score is None:
            continue

        try:
            score = float(
                score
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if np.isfinite(score):
            baseline_scores.append(
                (name, score)
            )

    if not baseline_scores:
        print(
            "No successful baseline "
            "results are available."
        )
        print(
            "Improvement analysis "
            "skipped safely."
        )
        return None

    best_baseline_name, best_baseline_f1 = max(
        baseline_scores,
        key=lambda x: x[1],
    )

    proposed_f1 = float(
        proposed[
            "test_f1"
        ]
    )

    absolute_gain = (
        proposed_f1
        - best_baseline_f1
    )

    relative_gain = (
        absolute_gain
        / max(
            best_baseline_f1,
            1e-12,
        )
        * 100.0
    )

    print(
        f"Best baseline      : "
        f"{best_baseline_name}"
    )

    print(
        f"Best baseline F1   : "
        f"{best_baseline_f1:.4f}"
    )

    print(
        f"Proposed F1        : "
        f"{proposed_f1:.4f}"
    )

    print(
        f"Absolute F1 gain   : "
        f"{absolute_gain:.4f}"
    )

    print(
        f"Relative gain (%)  : "
        f"{relative_gain:.2f}"
    )

    return {
        "best_baseline":
            best_baseline_name,

        "best_baseline_f1":
            best_baseline_f1,

        "proposed_f1":
            proposed_f1,

        "absolute_gain":
            absolute_gain,

        "relative_gain_percent":
            relative_gain,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    results = {}
    failures = {}

    # --------------------------------------------------------
    # 1. Train compact baselines
    # --------------------------------------------------------
    for display_name, hf_name in BASELINE_MODELS.items():
        try:
            result = train_one_baseline(
                display_name,
                hf_name,
                BASELINE_EPOCHS,
            )

            results[
                display_name
            ] = result

        except Exception as e:
            print(
                "\n"
                + "!" * 70
            )

            print(
                f"ERROR training "
                f"{display_name}: {e}"
            )

            print(
                "Continuing with the "
                "remaining models."
            )

            print(
                "!" * 70
            )

            import traceback
            traceback.print_exc()

            failures[
                display_name
            ] = str(e)

            cleanup_cuda()

    # --------------------------------------------------------
    # 2. Proposed model
    # --------------------------------------------------------
    if RUN_PROPOSED:
        try:
            proposed_result = (
                run_proposed()
            )

            results[
                "Proposed"
            ] = proposed_result

        except Exception as e:
            print(
                "\n"
                + "!" * 70
            )

            print(
                f"ERROR training/evaluating "
                f"Proposed model: {e}"
            )

            print(
                "!" * 70
            )

            import traceback
            traceback.print_exc()

            failures[
                "Proposed"
            ] = str(e)

            cleanup_cuda()

    # --------------------------------------------------------
    # 3. Comprehensive table
    # --------------------------------------------------------
    print(
        "\n"
        + "=" * 70
    )

    print(
        "COMPREHENSIVE MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    comparison = (
        build_comparison_table(
            results
        )
    )

    if comparison.empty:
        print(
            "No completed model results "
            "available."
        )
    else:
        pd.set_option(
            "display.max_columns",
            None,
        )

        print(
            comparison.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

        comparison.to_csv(
            OUTPUT_DIR
            / "model_comparison.csv",
            index=False,
        )

    # --------------------------------------------------------
    # 4. Statistical significance:
    #    each baseline vs proposed
    # --------------------------------------------------------
    print(
        "\n"
        + "=" * 70
    )

    print(
        "STATISTICAL SIGNIFICANCE TESTING"
    )

    print(
        "=" * 70
    )

    significance_rows = []

    proposed = results.get(
        "Proposed"
    )

    if proposed is not None:
        p_preds = proposed.get(
            "predictions"
        )

        p_labels = proposed.get(
            "labels"
        )

        for name, r in results.items():
            if name == "Proposed":
                continue

            if r is None:
                continue

            b_preds = r.get(
                "predictions"
            )

            b_labels = r.get(
                "labels"
            )

            if (
                b_preds is None
                or p_preds is None
                or b_labels is None
                or p_labels is None
            ):
                continue

            if not (
                len(b_preds)
                == len(p_preds)
                == len(p_labels)
            ):
                print(
                    f"{name}: skipped "
                    "McNemar because prediction "
                    "lengths differ."
                )
                continue

            mc = mcnemar_test(
                b_preds,
                p_preds,
                p_labels,
            )

            print(
                f"{name} vs Proposed: "
                f"{mc}"
            )

            significance_rows.append(
                {
                    "Baseline":
                        name,

                    "Statistic":
                        mc[
                            "statistic"
                        ],

                    "p_value":
                        mc[
                            "p_value"
                        ],

                    "Significant":
                        mc[
                            "significant"
                        ],
                }
            )

    else:
        print(
            "Proposed result unavailable; "
            "McNemar tests skipped."
        )

    if significance_rows:
        pd.DataFrame(
            significance_rows
        ).to_csv(
            OUTPUT_DIR
            / "mcnemar_results.csv",
            index=False,
        )

    # --------------------------------------------------------
    # 5. Improvement analysis
    # --------------------------------------------------------
    improvement = (
        improvement_analysis(
            results
        )
    )

    # --------------------------------------------------------
    # 6. Confusion matrix for proposed
    # --------------------------------------------------------
    if proposed is not None:
        cm = confusion_matrix(
            proposed["labels"],
            proposed["predictions"],
            labels=[0, 1, 2],
        )

        cm_df = pd.DataFrame(
            cm,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        )

        print(
            "\nProposed confusion matrix"
        )

        print(
            cm_df
        )

        cm_df.to_csv(
            OUTPUT_DIR
            / "proposed_confusion_matrix.csv"
        )

    # --------------------------------------------------------
    # 7. Save summary
    # --------------------------------------------------------
    summary = {
        "completed_models":
            list(
                results.keys()
            ),

        "failures":
            failures,

        "improvement":
            improvement,
    }

    with open(
        OUTPUT_DIR
        / "run_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # 8. Final status
    # --------------------------------------------------------
    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXECUTION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "Completed models:"
    )

    for name in results:
        print(
            f"  [OK] {name}"
        )

    if failures:
        print(
            "\nFailed models:"
        )

        for name, error in failures.items():
            print(
                f"  [FAIL] "
                f"{name}: {error}"
            )

    print(
        "\nSaved outputs in:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print(
        "\nImportant output files:"
    )

    print(
        "  model_comparison.csv"
    )

    print(
        "  mcnemar_results.csv"
    )

    print(
        "  proposed_confusion_matrix.csv"
    )

    print(
        "  run_summary.json"
    )

    print(
        "  baselines/*_results.pkl"
    )

    print(
        "  proposed/proposed_results.pkl"
    )

    print(
        "  checkpoints/*.pt"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nExecution interrupted "
            "by user."
        )

    except Exception as e:
        print(
            f"\nFatal main error: {e}"
        )

        import traceback
        traceback.print_exc()

    finally:
        cleanup_cuda()
        print(
            "\nExecution complete."
        )
