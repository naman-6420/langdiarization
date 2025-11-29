import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data_load import RawFeatures, collate_fn_cnn_atten, get_atten_mask
from model import X_Attention_E2E_LD
from model_evaluation import compute_far_frr

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_output(outputs, seq_len):
    """Concatenates the outputs of a batch, respecting individual sequence lengths."""
    output_list = []
    for i in range(len(seq_len)):
        length = seq_len[i]
        output_list.append(outputs[i, :length, :])
    return torch.cat(output_list, dim=0)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def change_labels_to_string(label_list, lang_char):
    """Converts numeric labels back to string representation for evaluation."""
    mapping = {0: 'S', 1: lang_char, 2: 'E'}
    return [mapping.get(label, '?') for label in label_list]


def evaluation_metric(labels, predicted):
    """Prints classification report and confusion matrix."""
    # FINAL FIX: Removed 'width' argument to support older versions of scikit-learn
    report = metrics.classification_report(labels, predicted, zero_division=0)
    cm = metrics.confusion_matrix(labels, predicted)
    print("\n--- Validation Metrics ---")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("--------------------------\n")

def main():
    parser = argparse.ArgumentParser(description='Training script for Attention E2E model')
    parser.add_argument('--savedir', type=str, required=True, help='Directory to save the trained model')
    parser.add_argument('--train', type=str, required=True, help='Path to training data TSV file')
    parser.add_argument('--test', type=str, required=True, help='Path to testing data TSV file')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--lang', type=int, default=3, help='Number of language classes')
    parser.add_argument('--model', type=str, default='my_attention_model', help='Name for the saved model file')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--maxlength', type=int, default=666, help='Max sequence length for positional encoding')
    parser.add_argument('--lmbda', type=float, default=0.5, help='Lambda for joint loss')
    parser.add_argument('--fll', type=str, required=True, help='First language literal (e.g., "G" or "T")')
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'Current device: {device}')
    os.makedirs(args.savedir, exist_ok=True)

    model = X_Attention_E2E_LD(n_lang=args.lang, feat_dim=256, n_heads=4, d_k=256, d_v=256,
                               d_ff=2048, max_seq_len=args.maxlength, device=device)
    model.to(device)

    loss_func_CRE = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1).to(device)
    loss_func_xv = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1).to(device)

    train_set = RawFeatures(args.train)
    valid_set = RawFeatures(args.test)

    train_data = DataLoader(dataset=train_set, batch_size=args.batch, pin_memory=True,
                            num_workers=8, shuffle=True, collate_fn=collate_fn_cnn_atten)
    valid_data = DataLoader(dataset=valid_set, batch_size=args.batch, pin_memory=True,
                            shuffle=False, collate_fn=collate_fn_cnn_atten)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        pbar = tqdm(train_data, desc=f"Training Epoch {epoch+1}")
        for utt, labels, cnn_labels, feat_lens, _ in pbar:
            utt_ = utt.to(device=device, dtype=torch.float)

            transformer_seq_lens = [l // 10 for l in feat_lens]
            atten_mask = get_atten_mask(transformer_seq_lens, utt_.size(0)).to(device)
            
            labels = labels.to(device=device, dtype=torch.long)
            cnn_labels = cnn_labels.to(device=device, dtype=torch.long)

            outputs, cnn_outputs = model(utt_, transformer_seq_lens, atten_mask)

            outputs = get_output(outputs, transformer_seq_lens)
            
            b, s, _ = utt_.shape
            n_lang = cnn_outputs.shape[-1]
            cnn_outputs_seq = cnn_outputs.view(b, s // 10, n_lang)

            cnn_outputs_unpadded = get_output(cnn_outputs_seq, transformer_seq_lens)
            
            loss_trans = loss_func_CRE(outputs, labels)
            loss_xv = loss_func_xv(cnn_outputs_unpadded, cnn_labels)
            loss = args.lmbda * loss_trans + (1 - args.lmbda) * loss_xv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(Loss=f"{loss.item():.4f}", LR=f"{get_lr(optimizer):.6f}")
        
        scheduler.step()

        # Validation
        model.eval()
        correct, total_frames = 0, 0
        all_labels_numeric, all_preds_numeric = [], []

        with torch.no_grad():
            for utt, labels, _, feat_lens, _ in tqdm(valid_data, desc="Validating"):
                utt_ = utt.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)
                
                transformer_seq_lens = [l // 10 for l in feat_lens]
                atten_mask = get_atten_mask(transformer_seq_lens, utt_.size(0)).to(device)

                outputs, _ = model(x=utt_, seq_len=transformer_seq_lens, atten_mask=atten_mask)
                outputs = get_output(outputs, transformer_seq_lens)
                predicted = torch.argmax(outputs, -1)
                
                total_frames += labels.size(-1)
                correct += (predicted == labels).sum().item()
                
                all_labels_numeric.extend(labels.cpu().numpy())
                all_preds_numeric.extend(predicted.cpu().numpy())
        
        acc = correct / total_frames if total_frames > 0 else 0.0
        print(f'Validation Accuracy: {acc * 100:.4f}%')

        evaluation_metric(
            change_labels_to_string(all_labels_numeric, args.fll), 
            change_labels_to_string(all_preds_numeric, args.fll)
        )

        if acc > best_acc:
            print(f'New best accuracy: {acc*100:.4f}%. Saving model...')
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.savedir, f'{args.model}.ckpt'))

    print(f'\nFinished Training. Final Best Accuracy: {best_acc * 100:.4f}%')

if __name__ == "__main__":
    main()

