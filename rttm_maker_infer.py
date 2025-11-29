import os
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import math
import sys
import fairseq
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    import librosa
except ImportError:
    librosa = None

# We import from data_load, assuming it has the same content as data_load_without_wandb
from model import X_Base_E2E_LD, X_Attention_E2E_LD
from data_load_without_wandb import get_atten_mask

def get_output(outputs, seq_len):
    output_list = []
    for i in range(len(seq_len)):
        output_list.append(outputs[i, :seq_len[i], :])
    return torch.cat(output_list, dim=0)

def predictions_to_segments(predictions, frame_duration_ms=200):
    segments = []
    if not predictions: return segments
    current_label, start_time = predictions[0], 0
    for i, label in enumerate(predictions[1:], 1):
        if label != current_label:
            segments.append((current_label, start_time, i * frame_duration_ms))
            current_label, start_time = label, i * frame_duration_ms
    segments.append((current_label, start_time, len(predictions) * frame_duration_ms))
    return segments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_audio', type=str, required=True)
    parser.add_argument('--ground_truth_tsv', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, choices=['base', 'attention'])
    parser.add_argument('--w2v_path', type=str, required=True)
    parser.add_argument('--lang', type=int, default=3)
    parser.add_argument('--fll_name', type=str, default='Lang1')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_labels_txt', type=str, required=True)
    parser.add_argument('--output_rttm', type=str, required=True)
    parser.add_argument('--segment_output_dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- FIX 1: Base Model Path Override ---
    base_model_dir = os.path.dirname(args.w2v_path)
    base_model_path = os.path.join(base_model_dir, "CLSRIL-23.pt")
    
    if not os.path.exists(base_model_path):
        base_model_path = args.w2v_path 

    print(f"Loading Wav2Vec2 from: {args.w2v_path}...")
    try:
        model_w2v, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [args.w2v_path],
            arg_overrides={"data": "/tmp", "w2v_path": base_model_path}
        )
        model_w2v = model_w2v[0].eval().to(device)
        feature_extractor = model_w2v.w2v_encoder.w2v_model if hasattr(model_w2v, 'w2v_encoder') else model_w2v
    except Exception as e:
        print(f"Error loading Wav2Vec2: {e}")
        return

    # Load Diarization Model
    print(f"Loading {args.model_type} model...")
    ModelClass = X_Base_E2E_LD if args.model_type == 'base' else X_Attention_E2E_LD
    model_head = ModelClass(n_lang=args.lang, feat_dim=256, n_heads=4, d_k=256, d_v=256, d_ff=2048, max_seq_len=666, device=device)
    model_head.load_state_dict(torch.load(args.model_path, map_location=device))
    model_head.to(device).eval()

    # Load Audio
    print(f"Processing audio: {args.input_audio}")
    try:
        wav, sr = sf.read(args.input_audio)
        if sr != 16000 and librosa: wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        wav_tensor = torch.from_numpy(wav).float().to(device).unsqueeze(0)
    except Exception as e:
        print(f"Error reading audio: {e}")
        return

    # Inference
    with torch.no_grad():
        features = feature_extractor(wav_tensor, mask=False, features_only=True)['x']
        # Pad features to be divisible by 10
        if features.shape[1] % 10 != 0:
            pad_len = 10 - (features.shape[1] % 10)
            padding = torch.zeros(features.shape[0], pad_len, features.shape[2]).to(device)
            features = torch.cat([features, padding], dim=1)
        
        seq_len = [features.shape[1] // 10]
        
        # --- FIX 2: Correct get_atten_mask call ---
        # Your definition is: def get_atten_mask(seq_lens, batch_size)
        # So we pass: (seq_len, 1)
        atten_mask = get_atten_mask(seq_len, 1).to(device)
        
        outputs, _ = model_head(x=features, seq_len=seq_len, atten_mask=atten_mask)
        predicted = torch.argmax(get_output(outputs, seq_len), -1).cpu().numpy().tolist()

    # Save Output
    try:
        os.makedirs(os.path.dirname(args.output_labels_txt), exist_ok=True)
        with open(args.output_labels_txt, 'w') as f: f.write("".join(map(str, predicted)))
        print(f"Saved labels to: {args.output_labels_txt}")
    except Exception as e:
        print(f"Error saving labels: {e}")
    
    # Save RTTM
    hyp_segments = predictions_to_segments(predicted, 200)
    rttm_lines = []
    labels = {0: 'Silence', 1: args.fll_name, 2: 'English'} if args.lang == 3 else {0: 'English', 1: args.fll_name}
    base_name = os.path.basename(args.input_audio).replace('.wav', '')
    
    for c, s, e in hyp_segments:
        if labels.get(c) != 'Silence':
            rttm_lines.append(f"SPEAKER {base_name} 1 {s/1000.0:.3f} {(e-s)/1000.0:.3f} <NA> <NA> {labels.get(c)} <NA> <NA>\n")
    
    try:
        with open(args.output_rttm, 'w') as f: f.writelines(rttm_lines)
        print(f"Saved RTTM to: {args.output_rttm}")
    except Exception as e:
        print(f"Error saving RTTM: {e}")

    # Segmentation
    if args.segment_output_dir:
        os.makedirs(args.segment_output_dir, exist_ok=True)
        counts = defaultdict(int)
        for i, (c, s, e) in enumerate(hyp_segments):
            lbl = labels.get(c, f"Unknown{c}")
            counts[lbl] += 1
            out_dir = os.path.join(args.segment_output_dir, lbl)
            os.makedirs(out_dir, exist_ok=True)
            try:
                sf.write(os.path.join(out_dir, f"{base_name}_{i:03d}_{lbl}_{counts[lbl]:02d}.wav"), 
                         wav[int(s/1000*sr):int(e/1000*sr)], sr if sr else 16000)
            except Exception as e:
                print(f"Error saving clip: {e}")

    print("Success.")

if __name__ == "__main__":
    main()
