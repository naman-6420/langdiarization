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

try:
    import librosa
except ImportError:
    print("Warning: librosa not installed. Automatic resampling will not be available.", file=sys.stderr)
    librosa = None

# Imports from your existing project files
from model import X_Base_E2E_LD, X_Attention_E2E_LD
from data_load_without_wandb import get_atten_mask 

# --- Helper Functions (from JER/DER script) ---

def get_output(outputs, seq_len):
    """Helper function to unpad the model output."""
    output_list = []
    for i in range(len(seq_len)):
        length = seq_len[i]
        output_list.append(outputs[i, :length, :])
    return torch.cat(output_list, dim=0)

def predictions_to_segments(predictions, frame_duration_ms=200):
    """Converts a sequence of frame-by-frame predictions into a list of segments."""
    segments = []
    if not predictions:
        return segments
    current_label = predictions[0]
    start_time = 0
    for i, label in enumerate(predictions[1:], 1):
        if label != current_label:
            end_time = i * frame_duration_ms
            segments.append((current_label, start_time, end_time))
            current_label = label
            start_time = end_time
    end_time = len(predictions) * frame_duration_ms
    segments.append((current_label, start_time, end_time))
    return segments

def get_label_at_time(segments, time_ms):
    """Helper for DER: finds the label at a specific millisecond."""
    for label, start, end in segments:
        if start <= time_ms < end:
            return label
    return None

def calculate_jer_components(ref_segments, hyp_segments):
    """Calculates total intersection and union duration for each label class."""
    intersection = defaultdict(float)
    union = defaultdict(float)
    all_labels = set(seg[0] for seg in ref_segments) | set(seg[0] for seg in hyp_segments)
    
    for label in all_labels:
        ref_label_segs = [seg for seg in ref_segments if seg[0] == label]
        hyp_label_segs = [seg for seg in hyp_segments if seg[0] == label]
        
        total_ref_dur = sum(end - start for _, start, end in ref_label_segs)
        total_hyp_dur = sum(end - start for _, start, end in hyp_label_segs)
        
        intersect_dur = 0
        for r_start, r_end in [(s, e) for _, s, e in ref_label_segs]:
            for h_start, h_end in [(s, e) for _, s, e in hyp_label_segs]:
                overlap_start = max(r_start, h_start)
                overlap_end = min(r_end, h_end)
                if overlap_end > overlap_start:
                    intersect_dur += (overlap_end - overlap_start)
                    
        intersection[label] += intersect_dur
        union[label] += total_ref_dur + total_hyp_dur - intersect_dur
    return intersection, union

def calculate_der(ref_segments, hyp_segments, file_duration_ms):
    """Calculates Diarization Error Rate (DER) components."""
    timestamps = set([0, file_duration_ms])
    for _, start, end in ref_segments:
        timestamps.add(start)
        timestamps.add(end)
    for _, start, end in hyp_segments:
        timestamps.add(start)
        timestamps.add(end)
    
    sorted_timestamps = sorted(list(timestamps))
    
    fa_error = 0.0
    miss_error = 0.0
    conf_error = 0.0
    
    for i in range(len(sorted_timestamps) - 1):
        start = sorted_timestamps[i]
        end = sorted_timestamps[i+1]
        duration = end - start
        
        if duration == 0:
            continue
            
        mid_point = (start + end) / 2
        ref_label = get_label_at_time(ref_segments, mid_point)
        hyp_label = get_label_at_time(hyp_segments, mid_point)
        
        if ref_label != hyp_label:
            if ref_label == 0: # Silence is class 0
                fa_error += duration
            elif hyp_label == 0: # Silence is class 0
                miss_error += duration
            else:
                conf_error += duration
                
    return fa_error, miss_error, conf_error

def find_ground_truth(tsv_file, audio_filename, lang):
    """Finds the ground truth label string from a TSV file."""
    audio_basename = os.path.basename(audio_filename)
    
    with open(tsv_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            
            # Check if the audio_basename is a substring of the feature path
            # This handles both .wav and .npy filenames
            if audio_basename.split('.')[0] in parts[0]:
                label_str = parts[1].strip()
                if lang == 3: # MSCS
                    label_map = {'S': 0, 'E': 2}
                    default_lang = 1
                else: # MUCS
                    label_map = {'E': 0}
                    default_lang = 1
                
                label_numeric = []
                for char in label_str:
                    label_numeric.append(label_map.get(char, default_lang))
                
                return label_numeric
                
    return None


# --- Main Inference & Evaluation Script ---
def main():
    parser = argparse.ArgumentParser(description='Full Inference & Evaluation Script (HPC Version)')
    parser.add_argument('--input_audio', type=str, required=True, help='Path to the single .wav file to process.')
    parser.add_argument('--ground_truth_tsv', type=str, required=True, help='Path to the ground truth .tsv file (e.g., WAVData_DEV.tsv).')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.ckpt).')
    parser.add_argument('--model_type', type=str, required=True, choices=['base', 'attention'], help='Type of model: "base" or "attention".')
    parser.add_argument('--w2v_path', type=str, required=True, help='Path to the finetuned Wav2Vec2 feature extractor model (.pt).')
    parser.add_argument('--lang', type=int, default=3, help='Number of classes (3 for MSCS, 2 for MUCS).')
    parser.add_argument('--fll_name', type=str, default='Lang1', help='Primary language name (e.g., "Gujarati", "Hindi").')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID. Use -1 for CPU.')
    parser.add_argument('--output_labels_txt', type=str, required=True, help='Path to save the output label sequence (.txt).')
    parser.add_argument('--output_rttm', type=str, required=True, help='Path to save the output RTTM file.')
    parser.add_argument('--segment_output_dir', type=str, default=None, help='Optional: Directory to save the segmented audio clips.')
    
    args = parser.parse_args()

    # --- 1. Setup Device ---
    if args.device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- 2. Load Wav2Vec2 Feature Extractor ---
    print(f"Loading Wav2Vec2 feature extractor from: {args.w2v_path}...")
    try:
        model_w2v, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.w2v_path])
        model_w2v = model_w2v[0]
        model_w2v.eval()
        model_w2v.to(device)
        
        if hasattr(model_w2v, 'w2v_encoder'):
             feature_extractor = model_w2v.w2v_encoder.w2v_model
             print("   -> Loaded finetuned model, accessing w2v_encoder.w2v_model")
        else:
             feature_extractor = model_w2v
             print("   -> Loaded base model, using it directly.")
             
    except Exception as e:
        print(f"Error loading Wav2Vec2 model: {e}")
        return

    # --- 3. Load Trained Diarization Model (The "Head") ---
    print(f"Loading {args.model_type} model from: {args.model_path}...")
    if args.model_type == 'base':
        model_head = X_Base_E2E_LD(n_lang=args.lang, feat_dim=256, n_heads=4, d_k=256, d_v=256,
                                  d_ff=2048, max_seq_len=666, device=device)
    else: # attention
        model_head = X_Attention_E2E_LD(n_lang=args.lang, feat_dim=256, n_heads=4, d_k=256, d_v=256,
                                        d_ff=2048, max_seq_len=666, device=device)
    
    model_head.load_state_dict(torch.load(args.model_path, map_location=device))
    model_head.to(device)
    model_head.eval()

    # --- 4. Load Audio ---
    print(f"Loading audio file: {args.input_audio}")
    original_wav, original_sr = None, None
    try:
        wav, sr = sf.read(args.input_audio)
        original_wav, original_sr = wav, sr
        
        if sr != 16000:
            print(f"Warning: Sample rate is {sr} Hz. Resampling to 16000 Hz...")
            if librosa:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            else:
                print("Error: librosa is not installed. Please install it (`pip install librosa`) for automatic resampling.")
                return
        if wav.ndim > 1:
            wav = wav.mean(axis=1) # Convert stereo to mono
        
        wav_tensor = torch.from_numpy(wav).float().to(device)
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
            
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
        
    # --- 5. Load Ground Truth Label ---
    print(f"Loading ground truth from: {args.ground_truth_tsv}")
    label_numeric = find_ground_truth(args.ground_truth_tsv, args.input_audio, args.lang)
    if label_numeric is None:
        print(f"Error: Could not find ground truth for {args.input_audio} in {args.ground_truth_tsv}")
        return
    print("   -> Ground truth found.")
    
    # --- 6. Run Inference ---
    print("Running inference...")
    predicted_numeric = []
    with torch.no_grad():
        features = feature_extractor(wav_tensor, mask=False, features_only=True)['x']

        feat_len = features.shape[1]
        if feat_len % 10 != 0:
            padding_needed = 10 - (feat_len % 10)
            padding = torch.zeros(features.shape[0], padding_needed, features.shape[2]).to(device)
            features = torch.cat([features, padding], dim=1)
        
        true_seq_len = [feat_len // 10]
        padded_seq_len = features.shape[1] // 10
        atten_mask = get_atten_mask(true_seq_len, 1, padded_seq_len).to(device)

        outputs, _ = model_head(x=features, seq_len=true_seq_len, atten_mask=atten_mask)
        
        outputs_unpadded = get_output(outputs, true_seq_len)
        predicted_numeric = torch.argmax(outputs_unpadded, -1).cpu().numpy().tolist()

    # --- 7. Create Label Map ---
    if args.lang == 3:
        label_map = {0: 'Silence', 1: args.fll_name, 2: 'English'}
    else: # MUCS
        label_map = {0: 'English', 1: args.fll_name}

    # --- 8. Save Label String Output ---
    output_string = "".join(map(str, predicted_numeric))
    try:
        os.makedirs(os.path.dirname(args.output_labels_txt), exist_ok=True)
        with open(args.output_labels_txt, 'w') as f:
            f.write(output_string)
        print(f"\nSuccessfully saved label sequence to: {args.output_labels_txt}")
    except Exception as e:
        print(f"\nError saving output file: {e}")

    # --- 9. Generate Segments ---
    # Trim labels to match predictions (which are based on feature length)
    max_len = len(predicted_numeric)
    label_numeric = label_numeric[:max_len]
    file_duration_ms = max_len * 200

    ref_segments = predictions_to_segments(label_numeric, 200)
    hyp_segments = predictions_to_segments(predicted_numeric, 200)

    # --- 10. Save RTTM File ---
    print(f"Saving RTTM file to: {args.output_rttm}")
    filename_base = os.path.basename(args.input_audio).replace('.wav', '')
    all_rttm_entries = []
    for label_code, start_ms, end_ms in hyp_segments:
        lang_name = label_map.get(label_code, f'Class{label_code}')
        if lang_name == 'Silence': # Don't write silence to RTTM
            continue
        start_sec = start_ms / 1000.0
        duration_sec = (end_ms - start_ms) / 1000.0
        rttm_line = f"SPEAKER {filename_base} 1 {start_sec:.3f} {duration_sec:.3f} <NA> <NA> {lang_name} <NA> <NA>\n"
        all_rttm_entries.append(rttm_line)
    
    with open(args.output_rttm, 'w') as f:
        f.writelines(all_rttm_entries)
    print("   -> RTTM file saved.")

    # --- 11. Calculate and Print JER ---
    print("\n--- Jaccard Error Rate (JER) ---")
    total_jer_num = 0.0
    total_jer_den = 0.0
    intersection, union = calculate_jer_components(ref_segments, hyp_segments)
    
    for label_code, u_duration in sorted(union.items()):
        lang_name = label_map.get(label_code, f'Class{label_code}')
        if u_duration > 0:
            jer = 1.0 - (intersection[label_code] / u_duration)
            print(f"JER for Class '{lang_name}': {jer * 100:.2f}%")
            if lang_name != 'Silence':
                total_jer_num += jer * u_duration
                total_jer_den += u_duration
    
    if total_jer_den > 0:
        weighted_avg_jer = total_jer_num / total_jer_den
        print(f"Weighted Average JER (speech only): {weighted_avg_jer * 100:.2f}%")

    # --- 12. Calculate and Print DER ---
    print("\n--- Diarization Error Rate (DER) ---")
    fa, miss, conf = calculate_der(ref_segments, hyp_segments, file_duration_ms)
    total_der_duration = file_duration_ms
    
    if total_der_duration > 0:
        der = (fa + miss + conf) / total_der_duration
        fa_pct = (fa / total_der_duration) * 100
        miss_pct = (miss / total_der_duration) * 100
        conf_pct = (conf / total_der_duration) * 100
        
        print(f"Total DER: {der * 100:.2f}%")
        print("Breakdown:")
        print(f"  False Alarm (Silence as Speech): {fa_pct:.2f}%")
        print(f"  Missed Speech (Speech as Silence): {miss_pct:.2f}%")
        print(f"  Language Confusion (LangA as LangB): {conf_pct:.2f}%")
    else:
        print("No duration to calculate DER.")

    # --- 13. Segment and Save Audio Clips ---
    if args.segment_output_dir:
        print(f"\nSegmenting audio and saving to: {args.segment_output_dir}")
        os.makedirs(args.segment_output_dir, exist_ok=True)
        
        samples_per_ms = original_sr / 1000.0
        
        count_map = defaultdict(int)
        for i, (label_code, start_ms, end_ms) in enumerate(hyp_segments):
            label_name = label_map.get(label_code, f'Unknown{label_code}')
            count_map[label_name] += 1
            
            lang_audio_dir = os.path.join(args.segment_output_dir, label_name)
            os.makedirs(lang_audio_dir, exist_ok=True)
            
            start_sample = int(start_ms * samples_per_ms)
            end_sample = int(end_ms * samples_per_ms)
            
            sub_audio = original_wav[start_sample:end_sample]
            
            clip_filename = f"{filename_base}_{i+1:03d}_{label_name}_{count_map[label_name]:02d}.wav"
            output_path = os.path.join(lang_audio_dir, clip_filename)
            
            try:
                sf.write(output_path, sub_audio, original_sr)
            except Exception as e:
                print(f"Error writing audio segment {output_path}: {e}")
            
        print(f"Successfully saved {len(hyp_segments)} audio clips into sub-folders.")

if __name__ == "__main__":
    main()
