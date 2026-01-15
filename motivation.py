import os
import numpy as np
import torch
import torch.nn.functional as F
import json
import random
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

# Model zoo from biscope_utils.py
MODEL_ZOO = {
    'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'gemma-2b': 'google/gemma-1.1-2b-it',
    'gemma-7b': 'google/gemma-1.1-7b-it', 
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
}


def get_rank_from_logits(logits, labels):
    """
    Compute rank of actual tokens in model's predictions.
    Returns ranks for each position.
    """
    # Ensure logits and labels have the right shape
    # logits: (seq_len, vocab_size) or (batch_size, seq_len, vocab_size)
    # labels: (seq_len,) or (batch_size, seq_len)

    # If logits is 2D, we're good. If 3D with batch_size=1, squeeze it
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    if labels.dim() == 2:
        labels = labels.squeeze(0)

    # Get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 2, f"Expected 2 dimensions in matches tensor, got {matches.shape}"
    ranks, timesteps = matches[:, -1], matches[:, -2]

    # Make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # Convert to 1-indexed rank
    return ranks


def compute_next_token_features(logits, labels):
    """
    Compute Next Token (FCE-style) features: rank and CE loss.
    Aligned with biscope_utils.py's compute_fce_loss.
    """
    # FCE: shift to predict next token
    # logits shape: (1, seq_len, vocab_size), labels shape: (seq_len,)
    fce_logits = logits[0, :-1, :]  # Positions [0, 1, ..., seq_len-2]
    fce_labels = labels[1:]          # Positions [1, 2, ..., seq_len-1]
    # This aligns: logits[i] predicts labels[i+1]

    # Compute CE loss
    ce_loss = CrossEntropyLoss(reduction='none')(fce_logits, fce_labels)
    avg_ce = ce_loss.mean().item()

    # Compute rank
    ranks = get_rank_from_logits(fce_logits, fce_labels)
    avg_rank = ranks.mean().item()

    return avg_rank, avg_ce


def compute_last_token_features(logits, labels):
    """
    Compute Last Token (BCE-style) features: rank and CE loss.
    Aligned with biscope_utils.py's compute_bce_loss.
    """
    # BCE: no shift, same position prediction
    # logits shape: (1, seq_len, vocab_size), labels shape: (seq_len,)
    # Use SAME labels as FCE: labels[1:]
    bce_logits = logits[0, 1:, :]   # Positions [1, 2, ..., seq_len-1]
    bce_labels = labels[1:]          # Positions [1, 2, ..., seq_len-1]
    # This aligns: logits[i] predicts labels[i] (same position)

    # Compute CE loss
    ce_loss = CrossEntropyLoss(reduction='none')(bce_logits, bce_labels)
    avg_ce = ce_loss.mean().item()

    # Compute rank
    ranks = get_rank_from_logits(bce_logits, bce_labels)
    avg_rank = ranks.mean().item()

    return avg_rank, avg_ce


def detect_single_sample(args, model, tokenizer, sample, device='cuda'):
    """
    Process a single sample and extract 4 features:
    [next_token_rank, last_token_rank, next_token_ce, last_token_ce]
    """
    # Tokenize with clipping
    if args.sample_clip:
        text_ids = tokenizer(sample, return_tensors='pt', 
                            max_length=args.sample_clip, 
                            truncation=True).input_ids.to(device)
    else:
        text_ids = tokenizer(sample, return_tensors='pt').input_ids.to(device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids=text_ids)
        logits = outputs.logits
        labels = text_ids[0]
    
    # Compute features
    next_rank, next_ce = compute_next_token_features(logits, labels)
    last_rank, last_ce = compute_last_token_features(logits, labels)
    
    return [next_rank, last_rank, next_ce, last_ce]


def parse_dataset_arg(ds):
    """
    Parse dataset string: {paraphrased or nonparaphrased}_{task}_{generative_model}
    Returns: (dataset_type, task, generative_model)
    """
    parts = ds.split('_')
    if len(parts) < 3 or parts[0] not in ['paraphrased', 'nonparaphrased']:
        raise ValueError("Dataset must be in format {paraphrased or nonparaphrased}_{task}_{generative_model}")
    return parts[0], parts[1], '_'.join(parts[2:])


def load_dataset(args, dataset_type, task, generative_model):
    """
    Load human and GPT data from either HuggingFace or local files.
    Reuse logic from biscope_utils.py data_generation()
    """
    if args.use_hf_dataset:
        from datasets import load_dataset
        ds = load_dataset("HanxiGuo/BiScope_Data", split="train")
        paraphrased_flag = True if dataset_type == "paraphrased" else False
        
        human_data = ds.filter(lambda x: x["task"] == task and x["source"].lower() == "human")
        human_data = [s["text"] for s in human_data]
        
        gpt_data = ds.filter(lambda x: x["task"] == task and
                            x["paraphrased"] == paraphrased_flag and
                            x["source"].lower() == generative_model.lower())
        gpt_data = [s["text"] for s in gpt_data]
    else:
        # Load from local JSON files
        if dataset_type == 'paraphrased':
            base_dir = "./Paraphrased_Dataset"
        else:
            base_dir = "./Dataset"

        # Load human data
        with open(f'./Dataset/{task}/{task}_human.json', 'r') as f:
            human_data = json.load(f)

        # Parse based on task type (same as biscope_utils.py)
        if task == 'Arxiv':
            human_data = [s['abs'] for s in human_data]
        elif task == 'Code':
            human_data = [s[0] + s[1] for s in human_data]
        elif task in ['Essay', 'Creative']:
            human_data = [s.get('essay', s) for s in human_data]
        elif task == 'Yelp':
            human_data = [s for s in human_data]

        # Load GPT data
        with open(f'{base_dir}/{task}/{task}_{generative_model}.json', 'r') as f:
            gpt_data = json.load(f)

    return human_data, gpt_data


def data_generation(args):
    """
    Generate features for all samples in the dataset.
    Returns: (human_features, gpt_features) as numpy arrays of shape (N, 4)
    """
    # Parse dataset argument
    dataset_type, task, generative_model = parse_dataset_arg(args.dataset)

    # Load model
    if args.detect_model not in MODEL_ZOO:
        raise ValueError(f"Unknown detection model: {args.detect_model}")

    print(f"Loading detection model: {args.detect_model}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ZOO[args.detect_model],
        torch_dtype=torch.float16,
        device_map='auto'
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ZOO[args.detect_model],
        padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    human_data, gpt_data = load_dataset(args, dataset_type, task, generative_model)
    print(f"Human samples: {len(human_data)}, GPT samples: {len(gpt_data)}")

    # Use the first 300 samples for each dataset
    human_data = human_data[:300]
    gpt_data = gpt_data[:300]

    # Generate features for human data
    print("Extracting features from human samples...")
    human_features = []
    for sample in tqdm(human_data):
        features = detect_single_sample(args, model, tokenizer, sample)
        human_features.append(features)
    human_features = np.array(human_features)

    # Generate features for GPT data
    print("Extracting features from GPT samples...")
    gpt_features = []
    for sample in tqdm(gpt_data):
        features = detect_single_sample(args, model, tokenizer, sample)
        gpt_features.append(features)
    gpt_features = np.array(gpt_features)

    return human_features, gpt_features


def evaluate(human, gpt):
    """
    Evaluate F1 score using Random Forest classifier with 5-fold CV.
    human, gpt: numpy arrays of features
    """
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    features = np.concatenate([human, gpt], axis=0)
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)
    labels = np.concatenate([np.zeros(human.shape[0]), np.ones(gpt.shape[0])], axis=0)
    scores = cross_val_score(classifier, features, labels, cv=5, scoring='f1')
    return scores.mean()


def plot_motivation(human_features, gpt_features, save_path):
    """
    Create 2x3 subplot figure matching the provided image.
    human_features, gpt_features: (N, 4) arrays
    Columns: [next_token_rank, last_token_rank, next_token_ce, last_token_ce]
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Extract features
    human_next_rank = human_features[:, 0]
    human_last_rank = human_features[:, 1]
    human_next_ce = human_features[:, 2]
    human_last_ce = human_features[:, 3]

    gpt_next_rank = gpt_features[:, 0]
    gpt_last_rank = gpt_features[:, 1]
    gpt_next_ce = gpt_features[:, 2]
    gpt_last_ce = gpt_features[:, 3]

    # Colors
    human_color = '#5DADE2'  # Teal/Blue
    ai_color = '#F39C12'     # Orange

    # (a) Next Token Rank - Histogram
    f1_a = evaluate(human_next_rank, gpt_next_rank)
    axs[0, 0].hist(human_next_rank, bins=30, alpha=0.7, label='Human Text', color=human_color)
    axs[0, 0].hist(gpt_next_rank, bins=30, alpha=0.7, label='AI Text', color=ai_color)
    axs[0, 0].set_xlabel('Rank')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].set_title(f'(a) Next Token Rank, F1={f1_a:.2f}')
    axs[0, 0].legend()

    # (b) Last Token Rank - Histogram
    f1_b = evaluate(human_last_rank, gpt_last_rank)
    axs[0, 1].hist(human_last_rank, bins=30, alpha=0.7, label='Human Text', color=human_color)
    axs[0, 1].hist(gpt_last_rank, bins=30, alpha=0.7, label='AI Text', color=ai_color)
    axs[0, 1].set_xlabel('Rank')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].set_title(f'(b) Last Token Rank, F1={f1_b:.2f}')
    axs[0, 1].legend()

    # (c) Both Rank - Scatter
    f1_c = evaluate(human_features[:, :2], gpt_features[:, :2])
    axs[0, 2].scatter(human_next_rank, human_last_rank, alpha=0.6,
                     label='Human Text', color=human_color, s=20)
    axs[0, 2].scatter(gpt_next_rank, gpt_last_rank, alpha=0.6,
                     label='AI Text', color=ai_color, s=20)
    axs[0, 2].set_xlabel('Next Token Rank')
    axs[0, 2].set_ylabel('Last Token Rank')
    axs[0, 2].set_title(f'(c) Both Rank, F1={f1_c:.2f}')
    axs[0, 2].legend()

    # (d) Next Token CE - Histogram
    f1_d = evaluate(human_next_ce, gpt_next_ce)
    axs[1, 0].hist(human_next_ce, bins=30, alpha=0.7, label='Human Text', color=human_color)
    axs[1, 0].hist(gpt_next_ce, bins=30, alpha=0.7, label='AI Text', color=ai_color)
    axs[1, 0].set_xlabel('Cross Entropy Loss')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].set_title(f'(d) Next Token CE, F1={f1_d:.2f}')
    axs[1, 0].legend()

    # (e) Last Token CE - Histogram
    f1_e = evaluate(human_last_ce, gpt_last_ce)
    axs[1, 1].hist(human_last_ce, bins=30, alpha=0.7, label='Human Text', color=human_color)
    axs[1, 1].hist(gpt_last_ce, bins=30, alpha=0.7, label='AI Text', color=ai_color)
    axs[1, 1].set_xlabel('Cross Entropy Loss')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].set_title(f'(e) Last Token CE, F1={f1_e:.2f}')
    axs[1, 1].legend()

    # (f) Both CE - Scatter
    f1_f = evaluate(human_features[:, 2:], gpt_features[:, 2:])
    axs[1, 2].scatter(human_next_ce, human_last_ce, alpha=0.6,
                     label='Human Text', color=human_color, s=20)
    axs[1, 2].scatter(gpt_next_ce, gpt_last_ce, alpha=0.6,
                     label='AI Text', color=ai_color, s=20)
    axs[1, 2].set_xlabel('Next Token CE Loss')
    axs[1, 2].set_ylabel('Last Token CE Loss')
    axs[1, 2].set_title(f'(f) Both CE, F1={f1_f:.2f}')
    axs[1, 2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate motivation figure for BiScope paper')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sample_clip', type=int, default=2000,
                       help='Max token length for samples')
    parser.add_argument('--detect_model', type=str, required=True,
                       help='Detection model key (e.g., llama2-7b)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Format: {paraphrased or nonparaphrased}_{task}_{generative_model}')
    parser.add_argument('--use_hf_dataset', type=bool, default=False,
                       help='Load dataset from Hugging Face')
    parser.add_argument('--phase', type=str, default='collect',
                       choices=['collect', 'plot', 'all'],
                       help='Phase: collect features, plot only, or both')
    parser.add_argument('--output_dir', type=str, default='./motivation_results',
                       help='Output directory for results')
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse dataset name for file naming
    dataset_type, task, generative_model = parse_dataset_arg(args.dataset)
    save_prefix = f"{args.dataset}_{args.detect_model}"

    human_path = os.path.join(args.output_dir, f"{save_prefix}_human.npy")
    gpt_path = os.path.join(args.output_dir, f"{save_prefix}_gpt.npy")
    fig_path = os.path.join(args.output_dir, f"{save_prefix}_motivation.png")

    # Execute based on phase
    if args.phase in ['collect', 'all']:
        print("=== Phase: Feature Collection ===")
        human_features, gpt_features = data_generation(args)
        np.save(human_path, human_features)
        np.save(gpt_path, gpt_features)
        print(f"Features saved: {human_path}, {gpt_path}")
        print(f"Human features shape: {human_features.shape}")
        print(f"GPT features shape: {gpt_features.shape}")

    if args.phase in ['plot', 'all']:
        print("=== Phase: Plotting ===")
        human_features = np.load(human_path)
        gpt_features = np.load(gpt_path)
        print(f"Loaded human features: {human_features.shape}")
        print(f"Loaded GPT features: {gpt_features.shape}")
        plot_motivation(human_features, gpt_features, fig_path)


if __name__ == '__main__':
    main()

