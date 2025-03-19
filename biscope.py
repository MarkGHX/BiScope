import os
import argparse
import random
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from biscope_utils import data_generation
import torch

def parse_dataset_arg(ds):
    """
    Parse dataset string with expected format:
      {paraphrased or nonparaphrased}_{task}_{generative_model}
    For example: "paraphrased_Arxiv_gpt-3.5-turbo"
    Returns a tuple: (dataset_type, task, generative_model)
    """
    parts = ds.split('_')
    if len(parts) < 3 or parts[0] not in ['paraphrased', 'nonparaphrased']:
        raise ValueError("Dataset must be in format {paraphrased or nonparaphrased}_{task}_{generative_model}")
    return parts[0], parts[1], '_'.join(parts[2:])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--sample_clip', type=int, default=2000, help="Max token length for samples")
    parser.add_argument('--summary_model', type=str, default='none', help="Summary model key or 'none'")
    parser.add_argument('--detect_model', type=str, required=True, help="Detection model key")
    parser.add_argument('--train_dataset', type=str, required=True,
                        help='Format: {paraphrased or nonparaphrased}_{task}_{generative_model}')
    parser.add_argument('--test_dataset', type=str, required=True,
                        help='Format: {paraphrased or nonparaphrased}_{task}_{generative_model}')
    parser.add_argument('--use_hf_dataset', type=bool, default=False, help="Load dataset from Hugging Face")
    args = parser.parse_args()
    
    if args.use_hf_dataset:
        print("Using Hugging Face datasets...")
    else:
        print("Using local datasets...")

    # Set seeds.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Parse dataset arguments.
    train_type, train_task, train_gen = parse_dataset_arg(args.train_dataset)
    test_type, test_task, test_gen = parse_dataset_arg(args.test_dataset)
    
    # Create a base output directory that includes both train and test dataset strings.
    base_out_dir = os.path.join('./results', 
        f"{args.train_dataset}_vs_{args.test_dataset}_{args.detect_model}_{args.summary_model}_clip{args.sample_clip}")
    os.makedirs(base_out_dir, exist_ok=True)
    
    # Create separate subdirectories for train and test features.
    # If train and test datasets are identical, use the same directory.
    if args.train_dataset == args.test_dataset:
        train_dir = test_dir = base_out_dir
    else:
        train_dir = os.path.join(base_out_dir, "train")
        test_dir  = os.path.join(base_out_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    
    # Generate features for the training dataset.
    print("Generating train features...")
    data_generation(args, train_dir, train_type, train_task, train_gen)
    
    # Load train features.
    with open(os.path.join(train_dir, f"{train_task}_human_features.pkl"), 'rb') as f:
        train_human = np.array(pickle.load(f))
    with open(os.path.join(train_dir, f"{train_task}_GPT_features.pkl"), 'rb') as f:
        train_gpt = np.array(pickle.load(f))
    
    # If training and testing datasets are identical, use 5-fold CV.
    if args.train_dataset == args.test_dataset:
        feats = np.concatenate([train_human, train_gpt], axis=0)
        labels = np.concatenate([np.zeros(len(train_human)), np.ones(len(train_gpt))], axis=0)
        clf = RandomForestClassifier(n_estimators=100, random_state=args.seed)
        scores = cross_val_score(clf, feats, labels, cv=5, scoring='f1')
        print("5-fold CV F1 scores:", scores, "Average:", scores.mean())
        with open(os.path.join(base_out_dir, 'cv_scores.txt'), 'w') as f:
            f.write(" ".join(map(str, scores)))
    else:
        # For different train and test datasets, two cases are handled:
        # Case 1: Cross-model OOD setting: same task but different generative model/paraphrase status.
        if train_task == test_task:
            print("Evaluating cross-model OOD setting (same task):")
            # Train on human and GPT training features.
            train_feats = np.concatenate([train_human, train_gpt], axis=0)
            train_labels = np.concatenate([np.zeros(len(train_human)), np.ones(len(train_gpt))], axis=0)
            clf = RandomForestClassifier(n_estimators=100, random_state=args.seed)
            clf.fit(train_feats, train_labels)
            # Generate test features for GPT only.
            print("Generating test GPT features...")
            data_generation(args, test_dir, test_type, test_task, test_gen)
            with open(os.path.join(test_dir, f"{test_task}_GPT_features.pkl"), 'rb') as f:
                test_gpt = np.array(pickle.load(f))
            # In this setting, test labels are all 1 (GPT).
            test_labels = np.ones(len(test_gpt))
            preds = clf.predict(test_gpt)
            acc = np.mean(preds == test_labels)
            print("Test accuracy (using only GPT test data):", acc)
            with open(os.path.join(base_out_dir, 'test_accuracy.txt'), 'w') as f:
                f.write(str(acc))
        # Case 2: Cross-domain OOD setting: task changes.
        else:
            print("Evaluating cross-domain OOD setting (different task):")
            data_generation(args, test_dir, test_type, test_task, test_gen)
            with open(os.path.join(test_dir, f"{test_task}_human_features.pkl"), 'rb') as f:
                test_human = np.array(pickle.load(f))
            with open(os.path.join(test_dir, f"{test_task}_GPT_features.pkl"), 'rb') as f:
                test_gpt = np.array(pickle.load(f))
            train_feats = np.concatenate([train_human, train_gpt], axis=0)
            train_labels = np.concatenate([np.zeros(len(train_human)), np.ones(len(train_gpt))], axis=0)
            test_feats = np.concatenate([test_human, test_gpt], axis=0)
            test_labels = np.concatenate([np.zeros(len(test_human)), np.ones(len(test_gpt))], axis=0)
            clf = RandomForestClassifier(n_estimators=100, random_state=args.seed)
            clf.fit(train_feats, train_labels)
            preds = clf.predict(test_feats)
            f1 = f1_score(test_labels, preds)
            print("Test F1 score:", f1)
            with open(os.path.join(base_out_dir, 'test_f1.txt'), 'w') as f:
                f.write(str(f1))

if __name__ == '__main__':
    main()
